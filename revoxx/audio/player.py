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
from .worker_state import WorkerState
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
        self._playback_output_channel_index = None
        self.config = config

        # Attach to existing shared state
        self.shared_state = SharedState(create=False)
        self.shared_state.attach_to_existing(shared_state_name)

        # Playback state - explicit state machine
        self._state = WorkerState.IDLE
        self.audio_buffer: Optional[AudioBuffer] = None
        self.audio_data: Optional[np.ndarray] = None
        self.current_position = 0
        self.stream: Optional[sd.OutputStream] = None
        self._start_sample = 0
        self._end_sample: Optional[int] = None

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
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        audio_buffer: AudioBuffer,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
    ) -> None:
        """Start playback.

        Args:
            audio_data: Audio samples to play
            sample_rate: Sample rate in Hz
            audio_buffer: Audio buffer containing the shared memory
            start_sample: Starting sample position (default 0)
            end_sample: Ending sample position (default None = play to end)
        """
        self._stop_if_active()
        self._prepare_playback_state(audio_data, audio_buffer, start_sample, end_sample)
        self._init_level_calculator(sample_rate)
        self._init_shared_state(sample_rate)

        target_channel, num_channels = self._calculate_channel_mapping()
        self._playback_output_channel_index = target_channel

        device_index = self._get_output_device_index()
        if not self._open_output_stream(sample_rate, num_channels, device_index):
            return

        self.stream.start()
        self._state = WorkerState.ACTIVE

    def _stop_if_active(self) -> None:
        """Stop any current playback if player is in active state.

        Ensures clean transition by stopping the stream and allowing
        time for audio resources to be released before starting new playback.
        """
        if self._state == WorkerState.ACTIVE:
            self._stop_stream()
            time.sleep(UIConstants.AUDIO_PROCESS_SLEEP)

    def _prepare_playback_state(
        self,
        audio_data: np.ndarray,
        audio_buffer: AudioBuffer,
        start_sample: int,
        end_sample: Optional[int],
    ) -> None:
        """Set up buffer references and playback position bounds.

        Attaches to the shared memory buffer and configures the sample
        range for partial playback support.

        Args:
            audio_data: Source audio array (used for length calculation)
            audio_buffer: Shared memory buffer to attach
            start_sample: First sample to play (clamped to >= 0)
            end_sample: Last sample to play (None = end of audio)
        """
        self.audio_buffer = audio_buffer
        self.audio_data = audio_buffer.get_array()
        self._start_sample = max(0, start_sample)
        self._end_sample = end_sample if end_sample is not None else len(audio_data)
        self.current_position = self._start_sample

    def _init_level_calculator(self, sample_rate: int) -> None:
        """Reset level calculator for new playback session.

        Args:
            sample_rate: Sample rate of audio to be played
        """
        self.level_calculator.update_sample_rate(sample_rate)
        self.level_calculator.reset()

    def _init_shared_state(self, sample_rate: int) -> None:
        """Initialize shared state for cross-process position synchronization.

        Sets up the shared memory structure so the UI process can read
        the current playback position in real-time.

        Args:
            sample_rate: Sample rate for time calculations
        """
        effective_length = self._end_sample - self._start_sample
        self.shared_state.start_playback(effective_length, sample_rate)
        self.shared_state.update_playback_position(0, 0.0)

    def _calculate_channel_mapping(self) -> tuple:
        """Determine output channel routing configuration.

        Supports routing mono audio to a specific physical output channel
        on multi-channel interfaces (e.g., channel 3 of an 8-channel device).

        Returns:
            Tuple of (target_channel_index, num_stream_channels) where
            target_channel_index is the 0-based output channel and
            num_stream_channels is the total channels to open.
        """
        output_mapping = getattr(self, "_output_channel_mapping", None)
        if isinstance(output_mapping, list) and len(output_mapping) == 1:
            try:
                target = int(output_mapping[0])
                return target, max(1, target + 1)
            except (ValueError, TypeError, IndexError):
                pass
        return 0, 1

    def _get_output_device_index(self) -> Optional[int]:
        """Resolve configured device name to sounddevice index.

        Returns:
            Device index if found, None for system default.
        """
        if self.config.output_device is None:
            return None
        try:
            device_manager = get_device_manager()
            return device_manager.get_device_index_by_name(self.config.output_device)
        except (ImportError, RuntimeError):
            return None

    def _open_output_stream(
        self, sample_rate: int, num_channels: int, device_index: Optional[int]
    ) -> bool:
        """Create and open the audio output stream.

        Attempts to open the configured device first, falling back to
        the system default if that fails.

        Args:
            sample_rate: Playback sample rate
            num_channels: Number of output channels to open
            device_index: Target device index (None for default)

        Returns:
            True if stream opened successfully, False on failure.
        """
        stream_params = {
            "samplerate": sample_rate,
            "blocksize": self.blocksize,
            "channels": num_channels,
            "dtype": "float32",
            "callback": self._audio_callback,
            "finished_callback": self._finished_callback,
        }

        try:
            self.stream = sd.OutputStream(device=device_index, **stream_params)
            return True
        except (sd.PortAudioError, OSError):
            pass

        try:
            self.stream = sd.OutputStream(device=None, **stream_params)
            return True
        except (sd.PortAudioError, OSError) as e:
            print(f"Error opening OutputStream: {e}")
            self._state = WorkerState.IDLE
            return False

    def _stop_stream(self) -> None:
        """Stop the audio stream without changing state or cleaning up buffers.

        This is the low-level stream stop used internally.
        """
        stream = self.stream
        if stream:
            try:
                stream.stop()
                stream.close()
            except (sd.PortAudioError, RuntimeError) as e:
                print(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None

    def stop_playback(self) -> None:
        """Stop playback and clean up.

        This is idempotent - safe to call multiple times.
        """
        # Only stop if actually playing
        if self._state != WorkerState.ACTIVE:
            return

        self._stop_stream()
        self._state = WorkerState.IDLE

        self.shared_state.stop_playback()

        # Clean up shared buffer
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer = None
            self.audio_data = None

    def handle_command(
        self, command: dict, attached_buffer: Optional["AudioBuffer"] = None
    ) -> Optional["AudioBuffer"]:
        """Handle a command from the control queue.

        State-based command handling ensures idempotent operations:
        - "stop" when IDLE is ignored
        - "play" when PLAYING first stops, then starts

        Args:
            command: Command dictionary with 'action' key
            attached_buffer: Currently attached buffer (for cleanup)

        Returns:
            Updated attached_buffer or None if released
        """
        action = command.get("action")

        if action == "play":
            return self._handle_play_command(command, attached_buffer)
        elif action == "stop":
            return self._handle_stop_command(attached_buffer)
        elif action == "set_output_device":
            self.set_output_device(command.get("device_name"))
        elif action == "set_output_channel_mapping":
            self._update_channel_mapping(command.get("mapping"))
        elif action == "refresh_devices":
            self._refresh_devices()

        return attached_buffer

    def _handle_play_command(
        self, command: dict, attached_buffer: Optional["AudioBuffer"]
    ) -> Optional["AudioBuffer"]:
        """Process play command and start audio playback.

        Cleans up any existing buffer, attaches to the new shared memory
        buffer from the command, and initiates playback.

        Args:
            command: Play command with buffer_metadata and playback params
            attached_buffer: Previous buffer to clean up

        Returns:
            New attached buffer, or None if no valid metadata provided.
        """
        if attached_buffer:
            attached_buffer.close()

        buffer_metadata = command.get("buffer_metadata")
        if not buffer_metadata:
            return None

        attached_buffer = AudioBuffer.attach_to_existing(
            buffer_metadata["name"],
            tuple(buffer_metadata["shape"]),
            np.dtype(buffer_metadata["dtype"]),
        )

        audio_data = attached_buffer.get_array()
        sample_rate = command.get("sample_rate", self.config.sample_rate)
        start_sample = command.get("start_sample", 0)
        end_sample = command.get("end_sample", None)

        self.start_playback(
            audio_data, sample_rate, attached_buffer, start_sample, end_sample
        )
        return attached_buffer

    def _handle_stop_command(
        self, attached_buffer: Optional["AudioBuffer"]
    ) -> Optional["AudioBuffer"]:
        """Process stop command and halt playback.

        Idempotent: only performs cleanup if actually playing.

        Args:
            attached_buffer: Buffer to potentially clean up

        Returns:
            None if stopped and cleaned up, original buffer if already idle.
        """
        if self._state != WorkerState.ACTIVE:
            return attached_buffer

        self.stop_playback()
        if attached_buffer:
            attached_buffer.close()
        return None

    def _refresh_devices(self) -> None:
        """Refresh the audio device list."""
        try:
            device_manager = get_device_manager()
            device_manager.refresh()
        except (ImportError, RuntimeError):
            pass

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

        # Early exit if not in playing state
        if self._state != WorkerState.ACTIVE:
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
            self._state = WorkerState.IDLE
            raise sd.CallbackStop()

    def _update_playback_state(self, time_info: Any) -> None:
        """Update shared state with current playback position.

        Args:
            time_info: Hardware timing information
        """
        # Report position relative to start_sample for correct progress display
        relative_position = self.current_position - self._start_sample
        self.shared_state.update_playback_position(
            relative_position, time_info.outputBufferDacTime
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
        # Use end_sample instead of full audio length
        effective_end = (
            self._end_sample if self._end_sample is not None else len(self.audio_data)
        )
        remaining = effective_end - self.current_position
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
        # Use effective end position
        effective_end = (
            self._end_sample if self._end_sample is not None else len(self.audio_data)
        )

        # detect if next callback will exceed audio length
        next_position = self.current_position + self.blocksize

        if self.current_position < effective_end <= next_position:
            # Signal that we're in the last buffer before completion
            self.shared_state.mark_playback_finishing()

        if self.current_position >= effective_end:
            # Signal that playback is completed
            self.shared_state.mark_playback_completed()
            self._state = WorkerState.IDLE

    def _finished_callback(self) -> None:
        """Called when stream finishes."""
        self._state = WorkerState.IDLE
        self.shared_state.stop_playback()

    def cleanup(self) -> None:
        """Clean up resources.

        This forces cleanup regardless of state, used at process exit.
        """
        # Force stop stream regardless of state
        self._stop_stream()
        self._state = WorkerState.IDLE

        # Clean up buffer
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer = None
            self.audio_data = None

        if self.shared_state:
            self.shared_state.stop_playback()
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
