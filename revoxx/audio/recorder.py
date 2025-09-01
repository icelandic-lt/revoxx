"""Audio recorder with hardware-synchronized position updates.

This module implements recording with struct-based shared memory
for inter-process communication.
"""

import sys
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Any
import multiprocessing as mp
from multiprocessing.synchronize import Event
import traceback
import soundfile as sf

from .shared_state import SharedState, SHARED_STATUS_INVALID
from .level_calculator import LevelCalculator
from .queue_manager import AudioQueueManager
from ..utils.config import AudioConfig
from ..utils.audio_utils import calculate_blocksize
from ..utils.process_cleanup import ProcessCleanupManager
from ..utils.device_manager import get_device_manager


class AudioRecorder:
    """Audio recorder with struct-based synchronized position updates."""

    def __init__(
        self,
        config: AudioConfig,
        shared_state_name: str,
        queue_manager=None,
        manager_dict: Optional[dict] = None,
    ):
        """Initialize synchronized audio recorder.

        Args:
            config: Audio configuration
            shared_state_name: Name of shared memory block
            queue_manager: AudioQueueManager for queue communication
            manager_dict: Shared manager dict
        """
        self.config = config
        self.queue_manager = queue_manager
        self.manager_dict = manager_dict

        # Attach to existing shared state
        self.shared_state = SharedState(create=False)
        self.shared_state.attach_to_existing(shared_state_name)

        # Recording state
        self.is_recording = False
        self.audio_chunks = []
        self.stream: Optional[sd.InputStream] = None
        self.current_position = 0

        # Level calculator for meter updates
        self.level_calculator = LevelCalculator(config.sample_rate)

        # Calculate blocksize from response time setting
        self.blocksize = calculate_blocksize(
            config.sync_response_time_ms, config.sample_rate
        )

    def set_input_device(self, device_name: Optional[str]) -> None:
        """Update input device name used for future streams."""
        self.config.input_device = device_name

    def start_recording(self) -> bool:
        """Start synchronized recording.

        Returns:
            bool: True if stream started successfully, False otherwise
        """
        if not self._validate_and_prepare():
            return False

        open_channels = self._determine_channel_configuration()

        self.stream = self._create_stream(self.config.sample_rate, open_channels)
        if not self.stream:
            return False

        self.stream.start()
        return True

    def _validate_and_prepare(self) -> bool:
        """Validate recording state and prepare for recording.

        Returns:
            True if validation successful and ready to record
        """
        # Check recording state
        recording_state = self.shared_state.get_recording_state()
        if recording_state.get("status", 0) == SHARED_STATUS_INVALID:
            print("ERROR: Recording state not initialized", file=sys.stderr)
            return False

        # Get and validate audio settings
        settings = self.shared_state.get_audio_settings()
        if settings.get("status", 0) == SHARED_STATUS_INVALID:
            print(
                "ERROR: Audio settings not initialized (invalid status)",
                file=sys.stderr,
            )
            print(f"ERROR: Settings: {settings}", file=sys.stderr)
            return False

        # Update configuration if sample rate changed
        sample_rate = settings["sample_rate"]
        if sample_rate != self.config.sample_rate:
            self.config.sample_rate = sample_rate
            self.level_calculator.update_sample_rate(sample_rate)
            self.blocksize = calculate_blocksize(
                self.config.sync_response_time_ms, sample_rate
            )

        # Reset recording state
        self.is_recording = True
        self.audio_chunks = []
        self.current_position = 0

        # Update shared state
        self.shared_state.start_recording(sample_rate)
        return True

    def _determine_channel_configuration(self) -> int:
        """Determine how many channels to open based on device and mapping.

        Returns:
            Number of channels to open for the input stream
        """
        # Get device max channels
        max_channels = self._get_device_max_channels()

        # Get configured mapping
        input_mapping = getattr(self, "_input_channel_mapping", None)

        if not input_mapping:
            # Simple case - use configured channels limited by device
            self._input_channel_pick = None
            if max_channels and max_channels > 0:
                return min(self.config.channels, max_channels)
            return self.config.channels

        # Channel mapping specified - filter and validate
        filtered = self._filter_channel_mapping(input_mapping, max_channels)

        if filtered:
            self._input_channel_pick = filtered
            if max_channels:
                return min(len(filtered), max_channels)
            return len(filtered)

        # Fallback to default
        self._input_channel_pick = None
        return self.config.channels

    def _get_device_max_channels(self) -> Optional[int]:
        """Get maximum input channels for the configured device.

        Returns:
            Maximum number of input channels or None if unknown
        """
        try:
            # Convert device name to index if needed
            device_index = None
            if self.config.input_device is not None:
                try:
                    device_manager = get_device_manager()
                    device_index = device_manager.get_device_index_by_name(
                        self.config.input_device
                    )
                except (ImportError, RuntimeError):
                    pass

            if device_index is not None:
                dev_info = sd.query_devices(device_index)
            else:
                dev_info = sd.query_devices(None)

            if isinstance(dev_info, dict):
                return int(dev_info.get("max_input_channels", 0))
            return None
        except (sd.PortAudioError, ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _filter_channel_mapping(
        mapping: list, max_channels: Optional[int]
    ) -> Optional[list]:
        """Filter channel mapping to valid channels.

        Args:
            mapping: Requested channel mapping
            max_channels: Maximum available channels

        Returns:
            Filtered list of valid channel indices or None if all invalid
        """
        if not isinstance(mapping, list) or len(mapping) == 0:
            return None

        try:
            if max_channels is not None:
                # Filter to available channels
                filtered = [int(i) for i in mapping if 0 <= int(i) < max_channels]
            else:
                # No max known, convert to ints
                filtered = [int(i) for i in mapping]

            return filtered if filtered else None
        except (ValueError, TypeError):
            return None

    def _create_stream(
        self, sample_rate: int, channels: int
    ) -> Optional[sd.InputStream]:
        """Create input stream with fallback to default device.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of channels to open

        Returns:
            InputStream instance or None if failed
        """
        stream_params = {
            "samplerate": sample_rate,
            "blocksize": self.blocksize,
            "channels": channels,
            "dtype": self.config.dtype,
            "callback": self._audio_callback,
        }

        # Convert device name to index for current device list
        device_index = None
        if self.config.input_device is not None:
            try:
                device_manager = get_device_manager()
                device_index = device_manager.get_device_index_by_name(
                    self.config.input_device
                )
                if device_index is None:
                    print(
                        f"Device '{self.config.input_device}' not found, falling back to default",
                        file=sys.stderr,
                    )
            except (ImportError, RuntimeError):
                pass

        # Try with configured device
        if device_index is not None:
            try:
                stream_params["device"] = device_index
                return sd.InputStream(**stream_params)
            except (sd.PortAudioError, OSError):
                pass  # Fall through to default

        # Try with system default
        try:
            stream_params["device"] = None
            return sd.InputStream(**stream_params)
        except (sd.PortAudioError, OSError) as e:
            print(f"Error opening InputStream: {e}", file=sys.stderr)
            self.is_recording = False
            # Signal error to main process
            if self.manager_dict is not None:
                try:
                    self.manager_dict["last_input_error"] = str(e)
                except (KeyError, TypeError):
                    pass
            return None

    def _process_input_channels(self, indata: np.ndarray) -> np.ndarray:
        """Process input data according to channel mapping configuration.

        Args:
            indata: Raw input data from audio callback

        Returns:
            Processed audio data with appropriate channel selection/mixing
        """
        if not hasattr(self, "_input_channel_pick") or not self._input_channel_pick:
            return indata.copy()

        # Guard indices vs delivered channel count
        available = indata.shape[1] if indata.ndim == 2 else 1
        safe_indices = [i for i in self._input_channel_pick if 0 <= i < available]

        if len(safe_indices) == 0:
            # Fallback to first available channels matching config.channels
            if indata.ndim == 2 and available >= self.config.channels:
                picked = indata[:, : self.config.channels]
            else:
                picked = indata
        else:
            picked = indata[:, safe_indices] if indata.ndim == 2 else indata

        # If more than one channel selected for mono config, average to mono
        if self.config.channels == 1 and picked.ndim == 2 and picked.shape[1] > 1:
            picked = picked.mean(axis=1, keepdims=True).astype(indata.dtype)

        return picked.copy()

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.shared_state.stop_recording()

        # Concatenate all chunks
        if self.audio_chunks:
            return np.concatenate(self.audio_chunks)
        return np.array([])

    def save_recording(self, audio_data: np.ndarray, filepath: Path) -> None:
        """Save audio data to file.

        Args:
            audio_data: Audio samples to save
            filepath: Path to save file
        """
        # Get current settings from shared state
        settings = self.shared_state.get_audio_settings()

        # Check if settings are correctly initialized
        if settings.get("status", 0) == SHARED_STATUS_INVALID:
            print(
                "ERROR: Audio settings not initialized in save_recording (invalid status)",
                file=sys.stderr,
            )
            return

        sample_rate = settings["sample_rate"]
        bit_depth = settings["bit_depth"]

        # Determine subtype based on format and bit depth
        if filepath.suffix.lower() == ".flac":
            # For FLAC, explicitly set subtype based on bit depth
            if bit_depth == 24:
                sf.write(str(filepath), audio_data, sample_rate, subtype="PCM_24")
            else:
                sf.write(str(filepath), audio_data, sample_rate, subtype="PCM_16")
        elif self.config.subtype:
            # For WAV files, use configured subtype
            sf.write(
                str(filepath), audio_data, sample_rate, subtype=self.config.subtype
            )
        else:
            # Default behavior
            sf.write(str(filepath), audio_data, sample_rate)

    def handle_command(self, command: dict) -> Optional[Any]:
        """Handle a command from the control queue.

        Args:
            command: Command dictionary with 'action' key

        Returns:
            Result of the command, or None if unknown command
        """
        action = command.get("action")

        if action == "start":
            return self.start_recording()

        elif action == "stop":
            audio_data = self.stop_recording()

            # Save if path provided (compatibility with current architecture)
            save_path = self._get_save_path()
            if save_path and len(audio_data) > 0:
                self.save_recording(audio_data, Path(save_path))
                self._clear_save_path()
            return audio_data

        elif action == "set_input_device":
            device_name = command.get("device_name", None)
            self.set_input_device(device_name)
            return True

        elif action == "set_input_channel_mapping":
            mapping = command.get("mapping", None)
            self._update_channel_mapping(mapping)
            return True

        elif action == "refresh_devices":
            try:
                device_manager = get_device_manager()
                device_manager.refresh()
                print("[Recorder] Device list refreshed", file=sys.stderr)
            except (ImportError, RuntimeError) as e:
                print(f"[Recorder] Error refreshing devices: {e}", file=sys.stderr)
            return True

        else:
            return None  # Unknown command

    def _update_channel_mapping(self, mapping: Optional[list]) -> None:
        """Update the input channel mapping configuration.

        Args:
            mapping: List of channel indices or None for default
        """
        try:
            if isinstance(mapping, list):
                # Validate that all values can be converted to int
                mapping = [int(x) for x in mapping]
                self._input_channel_mapping = mapping
            else:
                self._input_channel_mapping = None
        except (ValueError, TypeError):
            self._input_channel_mapping = None

    def _is_audio_queue_active(self) -> bool:
        """Check if audio queue should receive data.
        Same functionality as in ProcessManager, that we cannot access from
        this OS process ctx.

        Returns:
            True if audio queue is active, False otherwise
        """
        if not self.manager_dict:
            return False
        try:
            return self.manager_dict.get("audio_queue_active", False)
        except (AttributeError, KeyError):
            return False

    def _get_save_path(self) -> Optional[str]:
        """Get the current save path for recording.
        Same functionality as ProcessManager.get_save_path(), which we cannot
        access from this separate OS process.

        Returns:
            Path to save recording or None
        """
        if not self.manager_dict:
            return None
        try:
            return self.manager_dict.get("save_path")
        except (AttributeError, KeyError):
            return None

    def _clear_save_path(self) -> None:
        """Clear the save path after recording is saved.
        Same functionality as ProcessManager.set_save_path(None), which we cannot
        access from this separate OS process.
        """
        if self.manager_dict:
            try:
                self.manager_dict["save_path"] = None
            except (AttributeError, KeyError):
                pass

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Audio stream callback with hardware timing.

        Args:
            indata: Input buffer with audio data
            frames: Number of frames received
            time_info: Hardware timing information
            status: Callback status flags
        """
        if status:
            # Log any callback issues (e.g., InputOverflow means data was lost)
            print(f"Recording callback status: {status}", file=sys.stderr)

        if self.is_recording:
            # Store audio chunk
            try:
                processed_audio = self._process_input_channels(indata)
                self.audio_chunks.append(processed_audio)
            except (ValueError, MemoryError):
                # On any channel/memory error, append zeros to keep timing consistent
                try:
                    zeros = np.zeros_like(indata)
                    self.audio_chunks.append(zeros)
                except MemoryError:
                    pass

            # Update shared state with hardware timing
            self.shared_state.update_recording_position(
                self.current_position, time_info.inputBufferAdcTime
            )

            # Calculate and update level meter
            rms_db, peak_db, peak_hold_db = self.level_calculator.process(
                indata, self.config.channels
            )
            self.shared_state.update_level_meter(
                rms_db=rms_db,
                peak_db=peak_db,
                peak_hold_db=peak_hold_db,
                frame_count=self.level_calculator.get_frame_count(),
            )

            # Send to visualization queue if active
            if self._is_audio_queue_active():
                self.queue_manager.put_audio_data(indata.copy())

            # Update position
            self.current_position += frames

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.shared_state:
            self.shared_state.close()


def record_process(
    config: AudioConfig,
    audio_queue: mp.Queue,
    shared_state_name: str,
    control_queue: mp.Queue,
    manager_dict: dict,
    shutdown_event: Event,
) -> None:
    """Process function for audio recording with hardware synchronization.

    Args:
        config: Audio configuration
        audio_queue: Queue for audio visualization
        shared_state_name: Name of shared memory block
        control_queue: Queue for control commands
        manager_dict: Shared manager dict (for save_path compatibility)
        shutdown_event: Signal for shutting down process
    """
    # Setup signal handling for child process
    cleanup = ProcessCleanupManager(cleanup_callback=None, debug=False)
    cleanup.ignore_signals_in_child()

    # Create AudioQueueManager with existing queues
    queue_manager = AudioQueueManager(
        record_queue=control_queue,
        playback_queue=None,  # Not used in record process
        audio_queue=audio_queue,
    )

    recorder = None

    try:
        recorder = AudioRecorder(config, shared_state_name, queue_manager, manager_dict)

        while True:
            # Get next command
            try:
                command = queue_manager.get_record_command(timeout=0.1)
            except TypeError as e:
                print(f"Record process received invalid command: {e}")
                continue

            if command is None:
                # Check shutdown event
                if shutdown_event.is_set():
                    break
                continue

            # Handle quit command directly
            if command.get("action") == "quit":
                break

            result = recorder.handle_command(command)
            if result is None:
                print(f"Warning: Unsupported action: {command.get('action')}")

    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        print("")
    except Exception as e:
        print(f"[RECORD_PROCESS] Fatal error: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        if recorder:
            recorder.cleanup()
