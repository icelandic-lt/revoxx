"""Audio playback with hardware-synchronized position updates.

This module implements the playback system with struct-based
shared memory for inter-process communication.
"""

import time
import numpy as np
import sounddevice as sd
from typing import Optional, Any
import multiprocessing as mp
from multiprocessing.synchronize import Event
import traceback

from .audio_buffer import AudioBuffer
from .shared_state import SharedState
from .level_calculator import LevelCalculator
from .queue_manager import AudioQueueManager
from ..utils.config import AudioConfig
from ..utils.audio_utils import calculate_blocksize
from ..utils.process_cleanup import ProcessCleanupManager
from ..utils.device_manager import get_device_manager
from ..constants import UIConstants


class AudioPlayer:
    """Audio player with synchronized position updates."""

    def __init__(self, config: AudioConfig, shared_state_name: str):
        """Initialize audio player.

        Args:
            config: Audio configuration
            shared_state_name: Name of shared memory block
        """
        self.config = config

        # Attach to existing shared state
        self.shared_state = SharedState(create=False)
        self.shared_state.attach_to_existing(shared_state_name)

        # Playback state
        self.audio_buffer: Optional[AudioBuffer] = None
        self.audio_data: Optional[np.ndarray] = None
        self.current_position = 0
        self.stream: Optional[sd.OutputStream] = None
        self._stop_requested = False

        # Calculate blocksize from response time setting
        self.blocksize = calculate_blocksize(
            config.sync_response_time_ms, config.sample_rate
        )

        # Level calculator for meter updates
        self.level_calculator = LevelCalculator(config.sample_rate)

    def set_output_device(self, device_name: Optional[str]) -> None:
        """Update output device name used for future streams."""
        self.config.output_device = device_name

    def start_playback(
        self, audio_data: np.ndarray, sample_rate: int, audio_buffer: AudioBuffer
    ) -> None:
        """Start playback.

        Args:
            audio_data: Audio samples to play
            sample_rate: Sample rate in Hz
            audio_buffer: Audio buffer containing the shared memory
        """
        # Stop any current playback
        self.stop_playback()
        # Give the audio system time to release resources (empirically determined)
        time.sleep(UIConstants.AUDIO_PROCESS_SLEEP)

        # Use provided SHM buffer with normalized data
        self.audio_buffer = audio_buffer
        self.audio_data = audio_buffer.get_array()  # Zero-copy

        # Reset positions
        self.current_position = 0
        self._stop_requested = False

        # Update level calculator sample rate if needed
        self.level_calculator.update_sample_rate(sample_rate)
        self.level_calculator.reset()

        # Update shared state with initial position
        self.shared_state.start_playback(len(audio_data), sample_rate)
        self.shared_state.update_playback_position(0, 0.0)

        # Create output stream with callback
        # Optional routing to a specific physical output channel: we emulate mapping
        # by opening a stream with enough channels and writing only to the target one.
        output_mapping = getattr(self, "_output_channel_mapping", None)
        target_channel_index = 0
        num_stream_channels = 1
        if isinstance(output_mapping, list) and len(output_mapping) == 1:
            try:
                target_channel_index = int(output_mapping[0])
                num_stream_channels = max(1, target_channel_index + 1)
            except (ValueError, TypeError, IndexError):
                # Invalid channel mapping format
                target_channel_index = 0
                num_stream_channels = 1

        # Store for callback use
        self._playback_output_channel_index = target_channel_index

        # Convert device name to index for current device list
        device_index = None
        if self.config.output_device is not None:
            try:
                device_manager = get_device_manager()
                device_index = device_manager.get_device_index_by_name(
                    self.config.output_device
                )
            except (ImportError, RuntimeError):
                pass

        # Open stream with fallback to default device
        try:
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                blocksize=self.blocksize,
                device=device_index,
                channels=num_stream_channels,
                dtype="float32",  # Always use float32 for sounddevice
                callback=self._audio_callback,
                finished_callback=self._finished_callback,
            )
        except (sd.PortAudioError, OSError):
            try:
                self.stream = sd.OutputStream(
                    samplerate=sample_rate,
                    blocksize=self.blocksize,
                    device=None,
                    channels=num_stream_channels,
                    dtype="float32",
                    callback=self._audio_callback,
                    finished_callback=self._finished_callback,
                )
            except (sd.PortAudioError, OSError) as e:
                print(f"Error opening OutputStream: {e}")
                self._stop_requested = True
                return
        self.stream.start()

    def stop_playback(self) -> None:
        """Stop playback and clean up."""
        # Set stop flag first
        self._stop_requested = True

        # Store stream reference locally to avoid race conditions
        stream = self.stream
        if stream:
            try:
                stream.stop()
                stream.close()
            except (sd.PortAudioError, RuntimeError) as e:
                # Handle sounddevice specific errors
                print(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None

        self.shared_state.stop_playback()

        # Clean up shared buffer
        if self.audio_buffer:
            self.audio_buffer.close()
            # Don't unlink here - the buffer was created by main process
            self.audio_buffer = None
            self.audio_data = None

    def handle_command(
        self, command: dict, attached_buffer: Optional["AudioBuffer"] = None
    ) -> Optional["AudioBuffer"]:
        """Handle a command from the control queue.

        Args:
            command: Command dictionary with 'action' key
            attached_buffer: Currently attached buffer (for cleanup)

        Returns:
            Updated attached_buffer or None if released
        """
        action = command.get("action")

        if action == "play":
            # Clean up previous buffer if exists
            if attached_buffer:
                attached_buffer.close()

            # Get and attach to new buffer
            buffer_metadata = command.get("buffer_metadata")
            if buffer_metadata:
                attached_buffer = AudioBuffer.attach_to_existing(
                    buffer_metadata["name"],
                    tuple(buffer_metadata["shape"]),
                    np.dtype(buffer_metadata["dtype"]),
                )

                # Start playback
                audio_data = attached_buffer.get_array()
                sample_rate = command.get("sample_rate", self.config.sample_rate)
                self.start_playback(audio_data, sample_rate, attached_buffer)

                return attached_buffer

        elif action == "stop":
            self.stop_playback()
            # Clean up attached buffer
            if attached_buffer:
                attached_buffer.close()
            return None

        elif action == "set_output_device":
            device_name = command.get("device_name", None)
            self.set_output_device(device_name)

        elif action == "set_output_channel_mapping":
            mapping = command.get("mapping", None)
            self._update_channel_mapping(mapping)

        elif action == "refresh_devices":
            try:
                device_manager = get_device_manager()
                device_manager.refresh()
            except (ImportError, RuntimeError):
                pass

        else:
            return attached_buffer  # Unknown command, keep buffer

        return attached_buffer  # Keep current buffer for most commands

    def _update_channel_mapping(self, mapping: Optional[list]) -> None:
        """Update the output channel mapping configuration.

        Args:
            mapping: List of channel indices or None for default
        """
        try:
            if isinstance(mapping, list):
                mapping = [int(x) for x in mapping]
                self._output_channel_mapping = mapping
            else:
                self._output_channel_mapping = None
        except (ValueError, TypeError):
            self._output_channel_mapping = None

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Optional[sd.CallbackFlags],
    ) -> None:
        """Audio stream callback with hardware timing.

        Args:
            outdata: Output buffer to fill
            frames: Number of frames to provide
            time_info: Hardware timing information from sounddevice
            status: Callback status flags
        """
        if status:
            print(f"Playback callback status: {status}")

        # Early exit if stop requested
        if self._stop_requested:
            outdata.fill(0)
            raise sd.CallbackStop()

        # Update playback state
        self._update_playback_state(time_info)

        # Process audio frames
        if self.audio_data is None:
            outdata.fill(0)
            return

        frames_processed = self._process_audio_frames(outdata, frames)

        if frames_processed == 0:
            # End of audio reached
            outdata.fill(0)
            self.shared_state.stop_playback()
            self._stop_requested = True
            raise sd.CallbackStop()

    def _update_playback_state(self, time_info: Any) -> None:
        """Update shared state with current playback position.

        Args:
            time_info: Hardware timing information
        """
        self.shared_state.update_playback_position(
            self.current_position, time_info.outputBufferDacTime
        )
        # Explicitly mark PLAYING to avoid early IDLE reads
        self.shared_state.set_playback_state(status=2)

    def _process_audio_frames(self, outdata: np.ndarray, frames: int) -> int:
        """Process and output audio frames.

        Args:
            outdata: Output buffer to fill
            frames: Number of frames requested

        Returns:
            Number of frames actually processed
        """
        remaining = len(self.audio_data) - self.current_position
        if remaining <= 0:
            return 0

        # Copy audio data
        to_copy = min(frames, remaining)
        audio_chunk = self.audio_data[
            self.current_position : self.current_position + to_copy
        ]

        # Route audio to appropriate channel
        self._route_audio_to_channel(outdata, audio_chunk, to_copy)

        # Update level meter
        if to_copy > 0:
            self._update_level_meter(audio_chunk)

        # Fill rest with silence if needed
        if to_copy < frames:
            outdata[to_copy:] = 0

        # Update position and check for near-end
        self.current_position += to_copy
        self._check_playback_near_end()

        return to_copy

    def _route_audio_to_channel(
        self, outdata: np.ndarray, audio_chunk: np.ndarray, frames: int
    ) -> None:
        """Route audio to the appropriate output channel.

        Args:
            outdata: Output buffer
            audio_chunk: Audio data to route
            frames: Number of frames to write
        """
        out_channel_index = getattr(self, "_playback_output_channel_index", 0)

        # Only clear buffer if using multichannel output
        if outdata.shape[1] > 1:
            outdata.fill(0)

        # Guard channel index within bounds
        if 0 <= out_channel_index < outdata.shape[1]:
            outdata[:frames, out_channel_index] = audio_chunk

    def _update_level_meter(self, audio_chunk: np.ndarray) -> None:
        """Update level meter with current audio chunk.

        Args:
            audio_chunk: Audio data to analyze
        """
        rms_db, peak_db, peak_hold_db = self.level_calculator.process(
            audio_chunk.reshape(-1, 1), 1  # Reshape for mono
        )
        self.shared_state.update_level_meter(
            rms_db=rms_db,
            peak_db=peak_db,
            peak_hold_db=peak_hold_db,
            frame_count=self.level_calculator.get_frame_count(),
        )

    def _check_playback_near_end(self) -> None:
        """Check if playback is near the end and update state accordingly."""
        # detect if next callback will exceed audio length
        next_position = self.current_position + self.blocksize

        if self.current_position < len(self.audio_data) <= next_position:
            # Signal that we're in the last buffer before completion
            self.shared_state.mark_playback_finishing()

        if self.current_position >= len(self.audio_data):
            # Signal that playback is completed
            self.shared_state.mark_playback_completed()
            self._stop_requested = True

    def _finished_callback(self) -> None:
        """Called when stream finishes."""
        self.shared_state.stop_playback()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_playback()
        if self.shared_state:
            self.shared_state.close()


def playback_process(
    config: AudioConfig,
    control_queue: mp.Queue,
    shared_state_name: str,
    manager_dict: dict,
    shutdown_event: Event,
) -> None:
    """Process function for audio playback with hardware synchronization.

    Args:
        config: Audio configuration
        control_queue: Queue for control commands
        shared_state_name: Name of shared memory block
        manager_dict: Shared manager dict
        shutdown_event: End process ?
    """
    # Setup signal handling for child process
    cleanup = ProcessCleanupManager(cleanup_callback=None, debug=False)
    cleanup.ignore_signals_in_child()

    # Create AudioQueueManager with existing queues
    queue_manager = AudioQueueManager(
        record_queue=None,  # Not used in playback process
        playback_queue=control_queue,
        audio_queue=None,  # Not used in playback process
    )

    player = None
    attached_buffer: Optional[AudioBuffer] = None

    try:
        # Create player with shared state
        player = AudioPlayer(config, shared_state_name)

        while True:
            try:
                # Get next command with timeout
                command = queue_manager.get_playback_command(timeout=0.1)
            except TypeError as e:
                print(f"Playback process received invalid command: {e}")
                continue

            if command is None:
                # Check shutdown event
                if shutdown_event.is_set():
                    break
                continue

            if command.get("action") == "quit":
                break

            attached_buffer = player.handle_command(command, attached_buffer)

    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        print("")
    except Exception:
        traceback.print_exc()

    finally:
        # Cleanup
        if player:
            player.cleanup()
        if attached_buffer:
            attached_buffer.close()
