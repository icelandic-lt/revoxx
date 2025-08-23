"""Audio controller for recording and playback operations."""

import queue
import time
import threading
from pathlib import Path
from typing import TYPE_CHECKING
import sounddevice as sd

from ..constants import UIConstants
from ..utils.device_manager import get_device_manager

if TYPE_CHECKING:
    from ..app import Revoxx


class AudioController:
    """Handles audio recording and playback operations.

    This controller manages:
    - Recording start/stop
    - Playback control
    - Monitoring mode
    - Audio capture and processing
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the audio controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self.is_monitoring = False
        self.saved_spectrogram_state = None
        self.saved_level_meter_state = None
        self.transfer_thread = None

    def toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.app.state.recording.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        """Start recording."""
        self._start_audio_capture("recording")

    def stop_recording(self) -> None:
        """Stop recording."""
        self._stop_audio_capture("recording")

    def play_current(self) -> None:
        """Play current recording."""
        if not self.app.state.is_ready_to_play():
            self.app.window.show_message("No recording available")
            return

        # Stop monitoring if active
        if self.is_monitoring:
            self.stop_monitoring_mode()

        # Stop playback exactly like Left/Right keys do
        sd.stop()  # Immediate stop in main process
        self.stop_synchronized_playback()
        if hasattr(self.app.window, "mel_spectrogram"):
            self.app.window.mel_spectrogram.stop_playback()

        # Reset meter via shared state before starting a new playback
        try:
            self.app.shared_state.reset_level_meter()
        except AttributeError:
            # Shared state or method might not exist
            pass

        # Give the playback process time to handle the stop command
        time.sleep(UIConstants.PLAYBACK_STOP_DELAY)
        # Also clear playback status to IDLE
        try:
            self.app.shared_state.stop_playback()
        except AttributeError:
            # Shared state or method might not exist
            pass

        current_label = self.app.state.recording.current_label
        current_take = self.app.state.recording.get_current_take(current_label)

        # If default output device is in effect and not yet notified, inform user once
        if self.app._default_output_in_effect and not self.app._notified_default_output:
            try:
                self.app.window.show_message(
                    "Using system default output device (no saved/available selection)"
                )
            except AttributeError:
                # Window or method might not exist
                pass
            self.app._notified_default_output = True

        # Additionally, warn once if last stream open failed or device likely unavailable
        if hasattr(self.app, "last_output_error") and self.app.last_output_error:
            self.app.window.set_status(
                "Output device unavailable. Using system default if possible."
            )
            self.app.last_output_error = False

        # Quick rescan before playback
        try:
            device_manager = get_device_manager()
            device_manager.refresh()
        except (ImportError, RuntimeError):
            # Device manager might not be available or initialized
            pass

        # Verify selected output device still exists (if set)
        if self.app.config.audio.output_device is not None:
            device_manager = get_device_manager()
            available = [d["index"] for d in device_manager.get_output_devices()]
            if self.app.config.audio.output_device not in available:
                self.app.window.set_status(
                    "Selected output device not found. Using system default."
                )
                try:
                    self.app.playback_queue.put(
                        {"action": "set_output_device", "index": None}, block=False
                    )
                except (queue.Full, AttributeError):
                    # Queue might be full or not exist
                    pass

        # Hardware synchronized playback
        filepath = self.app.file_manager.get_recording_path(current_label, current_take)
        if filepath.exists():
            # Load audio
            audio_data, sr = self.app.file_manager.load_audio(filepath)
            duration = len(audio_data) / sr

            # Reset meter when starting playback of a file
            if (
                hasattr(self.app.window, "embedded_level_meter")
                and self.app.window.embedded_level_meter
            ):
                try:
                    self.app.window.embedded_level_meter.reset()
                except AttributeError:
                    # Level meter might not exist or have reset method
                    pass

            # Create shared audio buffer using buffer manager
            audio_buffer = self.app.buffer_manager.create_buffer(audio_data)

            # Send play command with buffer metadata
            self.app.playback_queue.put(
                {
                    "action": "play",
                    "buffer_metadata": audio_buffer.get_metadata(),
                    "sample_rate": sr,
                }
            )

            # Close our reference but don't unlink - buffer manager handles lifecycle
            audio_buffer.close()

            # Start animations
            if hasattr(self.app.window, "mel_spectrogram"):
                self.app.window.mel_spectrogram.start_playback(duration, sr)

            # Update level meter for playback if visible
            level_meter_visible = (
                self.app.window.level_meter_var.get()
                if hasattr(self.app.window, "level_meter_var")
                else False
            )
            if level_meter_visible:
                # Schedule periodic updates during playback
                self.start_playback_level_monitoring(filepath)

    def stop_synchronized_playback(self) -> None:
        """Stop synchronized playback."""
        self.app.playback_queue.put({"action": "stop"})
        # Also reset level meter when playback stops
        if (
            hasattr(self.app.window, "embedded_level_meter")
            and self.app.window.embedded_level_meter
        ):
            try:
                self.app.window.embedded_level_meter.reset()
            except AttributeError:
                # Level meter might not exist or have reset method
                pass

    def toggle_monitoring(self) -> None:
        """Toggle monitoring mode - shows both level meter and mel spectrogram."""
        if self.is_monitoring:
            self.stop_monitoring_mode()
        else:
            self.start_monitoring_mode()

    def start_monitoring_mode(self) -> None:
        """Start monitoring mode using record process without saving."""
        self._start_audio_capture("monitoring")

    def stop_monitoring_mode(self) -> None:
        """Stop monitoring mode - restore UI state."""
        self._stop_audio_capture("monitoring")

    def start_playback_level_monitoring(self, filepath: Path) -> None:
        """Start monitoring audio levels during playback.

        Args:
            filepath: Path to the audio file being played
        """
        # This is a placeholder for level monitoring during playback
        # The actual implementation would need to monitor the shared state
        # or audio output to update the level meter

    def _start_audio_capture(self, mode: str) -> None:
        """Start audio capture in recording or monitoring mode.

        Args:
            mode: Must be either 'recording' or 'monitoring'
        """
        if mode not in ("recording", "monitoring"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'recording' or 'monitoring'"
            )

        is_recording = mode == "recording"

        # Stop any active monitoring or playback
        if self.is_monitoring:
            self.stop_monitoring_mode()
        if is_recording:
            self.stop_synchronized_playback()

        # Recording-specific setup
        if is_recording:
            # Update state
            self.app.state.recording.is_recording = True
            current_label = self.app.state.recording.current_label

            if not current_label:
                return

            # Get next available take number (considers trash) and update state
            take_num = self.app.file_manager.get_next_take_number(current_label)
            # Don't update takes count here - it will be updated after recording is saved
            # The dialogs should show the current count, not the future count
            save_path = self.app.file_manager.get_recording_path(
                current_label, take_num
            )
            self.app.manager_dict["save_path"] = str(save_path)
        else:
            # Monitoring-specific setup
            self.is_monitoring = True
            # Save current UI state
            self.saved_spectrogram_state = self.app.state.ui.spectrogram_visible
            self.saved_level_meter_state = (
                self.app.window.level_meter_var.get()
                if hasattr(self.app.window, "level_meter_var")
                else False
            )

            # Show both visualizations
            if not self.app.state.ui.spectrogram_visible:
                self.app.display_controller.toggle_mel_spectrogram()
            if (
                hasattr(self.app.window, "level_meter_var")
                and not self.app.window.level_meter_var.get()
            ):
                self.app.window._toggle_level_meter_callback()
                self.app.settings_manager.update_setting("show_level_meter", True)
                self.app.root.update_idletasks()

            # Reset level meter when entering monitoring mode
            if (
                hasattr(self.app.window, "embedded_level_meter")
                and self.app.window.embedded_level_meter
            ):
                try:
                    self.app.window.embedded_level_meter.reset()
                except AttributeError:
                    # Level meter might not exist or have reset method
                    pass

        # Clear and start spectrogram
        if hasattr(self.app.window, "mel_spectrogram"):
            self.app.window.mel_spectrogram.clear()
            self.app.window.mel_spectrogram.start_recording(
                self.app.config.audio.sample_rate
            )

        # Update info overlay
        if self.app.window.info_overlay.visible:
            recording_params = {
                "sample_rate": self.app.config.audio.sample_rate,
                "bit_depth": self.app.config.audio.bit_depth,
                "channels": self.app.config.audio.channels,
            }
            self.app.window.info_overlay.show(
                recording_params,
                is_recording=True,
                is_monitoring=(mode == "monitoring"),
            )

        # If default input device is in effect and not yet notified, inform user once
        if self.app._default_input_in_effect and not self.app._notified_default_input:
            try:
                self.app.window.show_message(
                    "Using system default input device (no saved/available selection)"
                )
            except AttributeError:
                # Window or method might not exist
                pass
            self.app._notified_default_input = True

        # Preflight: quick rescan and availability check
        try:
            device_manager = get_device_manager()
            device_manager.refresh()
        except (ImportError, RuntimeError):
            # Device manager might not be available or initialized
            pass

        # Verify selected input device still exists (if set)
        if self.app.config.audio.input_device is not None:
            device_manager = get_device_manager()
            available = [d["index"] for d in device_manager.get_input_devices()]
            if self.app.config.audio.input_device not in available:
                # Device missing → message and fallback to default for this run
                self.app.window.set_status(
                    "Selected input device not found. Using system default."
                )
                # Do not change persisted selection; just let record process try with None
                self.app.record_queue.put({"action": "set_input_device", "index": None})

        # Start audio capture
        self.app.record_queue.put({"action": "start"})

        # Update UI
        if is_recording:
            self.app.display_controller.update_display()
        else:
            self.app.window.set_status("Monitoring input levels...")
            if hasattr(self.app.window, "monitoring_var"):
                self.app.window.monitoring_var.set(True)

    def _stop_audio_capture(self, mode: str) -> None:
        """Stop audio capture in recording or monitoring mode.

        Args:
            mode: Must be either 'recording' or 'monitoring'
        """
        if mode not in ("recording", "monitoring"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'recording' or 'monitoring'"
            )

        is_recording = mode == "recording"

        # Stop audio capture & spectrogram
        self.app.record_queue.put({"action": "stop"})
        if hasattr(self.app.window, "mel_spectrogram"):
            self.app.window.mel_spectrogram.stop_recording()

        # Recording-specific cleanup
        if is_recording:
            # Update state
            self.app.state.recording.is_recording = False

            # Update displayed take
            current_label = self.app.state.recording.current_label
            if current_label:
                current_take = self.app.state.recording.get_take_count(current_label)
                self.app.state.recording.set_displayed_take(current_label, current_take)

                # Wait a bit for the file to be saved by the recording process
                # then load and display the recording
                self.app.root.after(
                    UIConstants.POST_RECORDING_DELAY_MS,
                    lambda: self._after_recording_saved(current_label),
                )

            # Update display
            self.app.display_controller.update_display()

            # Update info overlay if visible to show the new recording
            if self.app.window.info_overlay.visible:
                # Wait a bit for the file to be saved
                self.app.root.after(
                    UIConstants.POST_RECORDING_DELAY_MS,
                    self.app.display_controller.update_info_overlay,
                )
        else:
            # Monitoring-specific cleanup
            self.is_monitoring = False

            # Restore UI state
            if (
                self.saved_spectrogram_state is not None
                and self.saved_spectrogram_state
                != self.app.state.ui.spectrogram_visible
            ):
                self.app.display_controller.toggle_mel_spectrogram()

            if (
                hasattr(self.app.window, "level_meter_var")
                and self.saved_level_meter_state is not None
            ):
                current_state = self.app.window.level_meter_var.get()
                if self.saved_level_meter_state != current_state:
                    self.app.window._toggle_level_meter_callback()

            # Clear saved states
            self.saved_spectrogram_state = None
            self.saved_level_meter_state = None

            # Update UI
            self.app.window.set_status("Ready")
            if hasattr(self.app.window, "monitoring_var"):
                self.app.window.monitoring_var.set(False)

            # Show previous recording if one exists
            self.app.display_controller.show_saved_recording()

            # Update info overlay if visible to show the current recording
            if self.app.window.info_overlay.visible:
                self.app.display_controller.update_info_overlay()

    def _save_last_recording_position(self, take_number: int) -> None:
        """Save the current recording position to the session.

        Args:
            take_number: The take number that was just recorded
        """
        if self.app.current_session and take_number > 0:
            self.app.current_session.last_recorded_index = (
                self.app.state.recording.current_index
            )
            self.app.current_session.last_recorded_take = take_number
            self.app.current_session.save()

    def _after_recording_saved(self, label: str) -> None:
        """Called after a recording has been saved to disk.

        Args:
            label: The label of the recording that was saved
        """
        if not self.app.active_recordings:
            return

        # Invalidate cache since we have a new recording
        if self.app.active_recordings:
            self.app.active_recordings.on_recording_completed(label)
            # Update takes from active recordings
            self.app.state.recording.takes = self.app.active_recordings.get_all_takes()

        # Update the displayed take to the new recording
        current_label = self.app.state.recording.current_label
        if current_label == label:
            if self.app.active_recordings:
                highest_take = self.app.active_recordings.get_highest_take(
                    current_label
                )
                self.app.state.recording.set_displayed_take(current_label, highest_take)
            else:
                highest_take = 1
                self.app.state.recording.set_displayed_take(current_label, highest_take)

            # Save the last recording position to session
            self._save_last_recording_position(highest_take)

            # Show the new recording
            self.app.display_controller.show_saved_recording()

            # Update take status
            self.app.navigation_controller.update_take_status()

    def start_audio_queue_processing(self):
        """Start processing audio queue for real-time display."""
        self.app.manager_dict["audio_queue_active"] = True

        # Start a transfer thread
        def audio_transfer_thread():
            try:
                while True:
                    # Check active flag with guard; manager may already be gone
                    try:
                        active = self.app.manager_dict.get("audio_queue_active", False)
                    except Exception:
                        break
                    if not active:
                        break

                    try:
                        # Get audio data from queue with timeout
                        audio_data = self.app.audio_queue.get(timeout=0.1)

                        # Process based on type
                        if isinstance(audio_data, dict):
                            if audio_data.get("type") == "audio":
                                # Transfer to mel spectrogram widget for display
                                if (
                                    hasattr(self.app.window, "mel_spectrogram")
                                    and self.app.window.mel_spectrogram
                                ):
                                    audio_array = audio_data.get("data")
                                    if audio_array is not None:
                                        self.app.root.after(
                                            0,
                                            lambda data=audio_array: self.app.window.mel_spectrogram.update_audio(
                                                data
                                            ),
                                        )
                            elif audio_data.get("type") == "level":
                                # Process level meter data
                                if hasattr(self.app.window, "embedded_level_meter"):
                                    level = audio_data.get("level", 0.0)
                                    # Use thread-safe method to update
                                    self.app.root.after(
                                        0,
                                        self.app.window.embedded_level_meter.update_level,
                                        level,
                                    )
                        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
                            # Legacy format: (audio_array, sample_rate)
                            audio_array, sample_rate = audio_data
                            if (
                                hasattr(self.app.window, "mel_spectrogram")
                                and self.app.window.mel_spectrogram
                            ):
                                self.app.root.after(
                                    0,
                                    lambda data=audio_array: self.app.window.mel_spectrogram.update_audio(
                                        data
                                    ),
                                )
                        else:
                            # Direct numpy array
                            if (
                                hasattr(self.app.window, "mel_spectrogram")
                                and self.app.window.mel_spectrogram
                            ):
                                self.app.root.after(
                                    0,
                                    lambda data=audio_data: self.app.window.mel_spectrogram.update_audio(
                                        data
                                    ),
                                )
                    except queue.Empty:
                        continue
                    except queue.Full:
                        # Mel spectrogram queue is full, skip this data
                        continue
                    except (BrokenPipeError, OSError, EOFError):
                        # Queue was closed, exit cleanly
                        break
                    except Exception as e:
                        if "closed" not in str(e).lower():
                            print(f"Error in audio transfer thread: {e}")
                        break
            except (BrokenPipeError, OSError, EOFError):
                # IPC endpoints closed during shutdown; exit quietly
                pass

        self.transfer_thread = threading.Thread(target=audio_transfer_thread)
        self.transfer_thread.daemon = True
        self.transfer_thread.start()

    def update_audio_queue_state(self):
        """Update audio queue processing state based on what's visible."""
        level_meter_visible = (
            hasattr(self.app.window, "level_meter_var")
            and self.app.window.level_meter_var.get()
        )
        needs_audio = self.app.state.ui.spectrogram_visible or level_meter_visible

        self.app.manager_dict["audio_queue_active"] = needs_audio
