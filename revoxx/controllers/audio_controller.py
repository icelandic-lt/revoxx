"""Audio controller for orchestrating recording and playback operations.

This controller acts as the high-level orchestrator for all audio-related
operations in the application. It coordinates between different subsystems
and manages the overall audio workflow.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable
import sounddevice as sd

from ..constants import UIConstants, MsgType
from ..utils.device_manager import get_device_manager
from ..audio.audio_queue_processor import AudioQueueProcessor

if TYPE_CHECKING:
    from ..app import Revoxx


class AudioController:
    """High-level orchestrator for audio operations.

    This controller is responsible for:
    - **Recording Control**: Starting/stopping recordings, managing recording state
    - **Playback Control**: Playing audio files, managing synchronized playback
    - **Monitoring Mode**: Live audio input monitoring with visualizations
    - **Device Management**: Verifying and managing audio input/output devices
    - **Mode Coordination**: Switching between recording/monitoring/playback modes
    - **UI State Management**: Preserving and restoring UI state during mode changes

    The controller delegates low-level audio queue processing to AudioQueueProcessor
    and coordinates with other controllers (FileManager, DisplayController, etc.)
    for a complete audio workflow.

    Attributes:
        app: Reference to the main application
        is_monitoring: Whether monitoring mode is active
        saved_meters_state: Preserved meters visibility state
        audio_queue_processor: Handles audio data queue processing
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the audio controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self.is_monitoring = False
        self.saved_meters_state = None
        self.audio_queue_processor = AudioQueueProcessor(app)

    @staticmethod
    def _refresh_device_manager():
        """Refresh the audio device manager to detect any hardware changes.

        The device manager caches the list of available audio devices. This method
        forces it to re-scan the system for any newly connected or disconnected
        audio devices.

        Returns:
            DeviceManager instance with refreshed device list, or None if the
            device manager service is unavailable.
        """
        try:
            device_manager = get_device_manager()
            device_manager.refresh()
            return device_manager
        except (ImportError, RuntimeError):
            # Device manager might not be available or initialized
            return None

    def _verify_output_device(self) -> None:
        """Verify output device is still available, fallback to default if not."""
        if self.app.config.audio.output_device is None:
            return

        device_manager = self._refresh_device_manager()
        if not device_manager:
            return

        # Check if device name is still available
        available_names = [d["name"] for d in device_manager.get_output_devices()]

        if self.app.config.audio.output_device not in available_names:
            # Device disappeared - this can happen on Linux with USB audio devices
            print(
                f"ERROR: Output device '{self.app.config.audio.output_device}' disappeared from system"
            )
            print(f"Available devices: {available_names}")
            print("Falling back to system default audio device")
            self.app.display_controller.set_status(
                "Selected output device not found. Using system default.", MsgType.ERROR
            )
            self.app.queue_manager.set_output_device(None)

    def _verify_input_device(self) -> None:
        """Verify input device is still available, fallback to default if not."""
        if self.app.config.audio.input_device is None:
            return

        device_manager = self._refresh_device_manager()
        if not device_manager:
            return

        # Check if device name is still available
        available_names = [d["name"] for d in device_manager.get_input_devices()]
        if self.app.config.audio.input_device not in available_names:
            self.app.display_controller.set_status(
                "Selected input device not found. Using system default.", MsgType.ERROR
            )
            self.app.queue_manager.set_input_device(None)

    def stop_all_playback_activities(
        self, callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Stop all current playback and monitoring activities.

        Ensure consistent playback stopping behavior across navigation, recording,
        and playback.

        Args:
            callback: Optional callback to execute after playback is fully stopped
        """
        if self.is_monitoring:
            self.stop_monitoring_mode()

        sd.stop()  # Immediate stop in main process
        self.stop_synchronized_playback()
        self.app.display_controller.stop_spectrogram_playback()

        try:
            self.app.shared_state.reset_level_meter()
        except AttributeError:
            pass

        # Schedule clearing playback status after a short delay
        def clear_and_callback():
            try:
                self.app.shared_state.stop_playback()
            except AttributeError:
                pass

            if callback:
                callback()

        # Use non-blocking delay if we have a window
        if hasattr(self.app, "window") and self.app.window:
            self.app.window.window.after(
                UIConstants.PLAYBACK_STOP_DELAY_MS, clear_and_callback
            )
        else:
            # In tests or without window, execute immediately
            clear_and_callback()

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

    def _execute_playback(self, filepath: Path) -> None:
        """Execute playback of audio file.

        Args:
            filepath: Path to the audio file to play
        """
        if not filepath.exists():
            return

        # Load audio
        audio_data, sr = self.app.file_manager.load_audio(filepath)
        duration = len(audio_data) / sr

        self.app.display_controller.reset_level_meters()

        # Create shared audio buffer using buffer manager
        audio_buffer = self.app.buffer_manager.create_buffer(audio_data)
        metadata = audio_buffer.get_metadata()

        # Send play command with buffer metadata
        self.app.queue_manager.start_playback(buffer_metadata=metadata, sample_rate=sr)

        # Close our reference but don't unlink - buffer manager handles lifecycle
        audio_buffer.close()

        self.app.display_controller.start_spectrogram_playback(duration, sr)

        # Note: Level meter updates during playback happen automatically
        # via shared memory from the playback process (AudioPlayer._update_level_meter)

    def play_current(self) -> None:
        """Play current recording."""
        if not self.app.state.is_ready_to_play():
            self.app.display_controller.set_status(
                "No recording available", MsgType.TEMPORARY
            )
            return

        self.stop_all_playback_activities()

        current_label = self.app.state.recording.current_label
        current_take = self.app.state.recording.get_current_take(current_label)

        # Handle device notifications
        self.app.notify_if_default_device("output")

        # Refresh and verify output device
        self._verify_output_device()

        # Execute playback
        filepath = self.app.file_manager.get_recording_path(current_label, current_take)
        self._execute_playback(filepath)

    def stop_synchronized_playback(self) -> None:
        """Stop audio playback that is synchronized with visual elements.

        This stops playback in the background audio process which is synchronized
        with the spectrogram display and position cursor. The synchronization happens
        via shared memory between the playback process and the UI.

        Side effects:
        - Sends stop command to playback process via queue
        - Resets the level meter display to idle state

        Note: The corresponding start is handled by play_current() which initiates
        synchronized playback of the current recording.
        """
        self.app.queue_manager.stop_playback()
        # Also reset level meter when playback stops
        self.app.display_controller.reset_level_meters()

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

    def _start_audio_capture(self, mode: str) -> None:
        """Start audio capture in recording or monitoring mode.

        This method coordinates the startup of audio capture across multiple components:
        1. Stops any conflicting audio operations (playback, other recordings)
        2. Configures the UI for the selected mode (shows spectrogram, level meter)
        3. Sends start command to the recording process via queue
        4. Updates application state to reflect active capture

        Recording mode: Captures audio to file for saving
        Monitoring mode: Captures audio for level display only (no file saved)

        Args:
            mode: Must be either 'recording' or 'monitoring'
        """
        if mode not in ("recording", "monitoring"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'recording' or 'monitoring'"
            )

        # Prepare for audio capture
        self._prepare_for_audio_capture(mode)

        # Mode-specific setup
        if mode == "recording":
            if not self._setup_recording_mode():
                return  # No current label, can't record
        else:
            self._setup_monitoring_mode()

        # Execute the actual audio capture start
        # Ensure spectrograms are ready before starting (especially for monitoring)
        self.app.display_controller.when_spectrograms_ready(
            lambda: self._do_start_audio_capture(mode)
        )

    def _prepare_for_audio_capture(self, mode: str) -> None:
        """Prepare system for audio capture by stopping conflicting operations.

        Audio capture requires exclusive access to the audio input device.
        This method stops any operations that would conflict:
        - Active monitoring (if starting recording)
        - Active playback (can't record and play simultaneously)
        - Previous recording sessions

        This ensures clean state before starting new capture.
        """
        if self.is_monitoring:
            self.stop_monitoring_mode()

        if mode == "recording":
            self.stop_synchronized_playback()

    def _setup_recording_mode(self) -> bool:
        """Set up recording mode for capturing audio to file.

        Prepares the recording environment by:
        1. Setting recording state flags
        2. Getting next available take number for the current label
        3. Creating file path for the new recording
        4. Storing path in shared dict for the recording process

        Returns:
            True if setup successful, False if no current label
        """
        self.app.state.recording.is_recording = True
        current_label = self.app.state.recording.current_label

        if not current_label:
            return False

        # Get next available take number and set save path
        take_num = self.app.file_manager.get_next_take_number(current_label)
        save_path = self.app.file_manager.get_recording_path(current_label, take_num)
        self.app.process_manager.set_save_path(str(save_path))
        return True

    def _setup_monitoring_mode(self) -> None:
        """Set up monitoring mode for live audio level display.

        Monitoring mode shows real-time audio input levels without saving to file.
        This is useful for:
        - Checking microphone levels before recording
        - Adjusting input gain
        - Testing audio setup

        The method preserves current UI state so it can be restored when
        monitoring stops.
        """
        self.is_monitoring = True

        # Save current UI state
        self._save_ui_state_for_monitoring()

        # Show monitoring visualizations
        self._show_monitoring_visualizations()

        self._reset_level_meter()

    def _save_ui_state_for_monitoring(self) -> None:
        """Save current UI state before entering monitoring mode.

        Monitoring mode temporarily shows the spectrogram and level meter
        even if they were hidden. This method saves the current visibility
        state so the UI can be restored to user preferences when monitoring
        ends.
        """
        # Save current meters state from main window to restore later
        if self.app.window and hasattr(self.app.window, "meters_visible"):
            self.saved_meters_state = self.app.window.meters_visible
        else:
            self.saved_meters_state = False

    def _show_monitoring_visualizations(self) -> None:
        """Enable visualizations for monitoring mode."""
        # Show meters in main window if not visible
        if self.app.window and not self.app.window.meters_visible:
            self.app.display_controller.toggle_meters()
            self.app.window.window.update_idletasks()

    def _reset_level_meter(self) -> None:
        """Reset level meter when entering monitoring mode."""
        self.app.display_controller.reset_level_meters()

    def _do_start_audio_capture(self, mode: str) -> None:
        """Execute the actual audio capture start.

        This is the final step that actually begins audio capture by:
        1. Clearing the spectrogram display for fresh data
        2. Starting spectrogram recording at current sample rate
        3. Sending start command to the recording process
        4. Updating status message in UI
        """
        self.app.display_controller.start_spectrogram_recording(
            self.app.config.audio.sample_rate
        )

        self._update_info_panel_for_capture(mode)

        # Handle device notifications and verify availability
        self.app.notify_if_default_device("input")
        self._verify_input_device()

        self.app.queue_manager.start_recording()

        if mode == "recording":
            self.app.display_controller.update_display()
        else:
            self.app.display_controller.set_status(
                "Monitoring input levels...", MsgType.ACTIVE
            )
            self.app.display_controller.set_monitoring_var(True)

    def _update_info_panel_for_capture(self, mode: str) -> None:
        """Update info panel for audio capture."""
        recording_params = {
            "sample_rate": self.app.config.audio.sample_rate,
            "bit_depth": self.app.config.audio.bit_depth,
            "channels": self.app.config.audio.channels,
        }

        self.app.display_controller.update_info_panels_with_params(recording_params)

    def _stop_audio_capture(self, mode: str) -> None:
        """Stop audio capture in recording or monitoring mode.

        Args:
            mode: Must be either 'recording' or 'monitoring'
        """
        if mode not in ("recording", "monitoring"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'recording' or 'monitoring'"
            )

        # Execute the actual audio capture stop
        self._do_stop_audio_capture()

        # Mode-specific cleanup
        if mode == "recording":
            self._cleanup_recording_mode()
        else:
            self._cleanup_monitoring_mode()

    def _do_stop_audio_capture(self) -> None:
        """Execute the actual audio capture stop."""
        self.app.queue_manager.stop_recording()
        self.app.display_controller.stop_spectrogram_recording()

    def _cleanup_recording_mode(self) -> None:
        """Clean up after recording mode ends.

        Resets recording-related state flags and updates UI to reflect
        that recording has stopped. This ensures the application state
        is consistent and ready for the next operation.
        """
        self.app.state.recording.is_recording = False

        current_label = self.app.state.recording.current_label
        if current_label:
            current_take = self.app.state.recording.get_take_count(current_label)
            self.app.state.recording.set_displayed_take(current_label, current_take)
            self.app.window.window.after(
                UIConstants.POST_RECORDING_DELAY_MS,
                lambda: self._after_recording_saved(current_label),
            )

        self.app.display_controller.update_display()
        if self.app.display_controller.is_info_panel_visible():
            self.app.window.window.after(
                UIConstants.POST_RECORDING_DELAY_MS,
                self.app.display_controller.update_info_panel,
            )

    def _cleanup_monitoring_mode(self) -> None:
        """Clean up after monitoring mode ends.

        Resets monitoring flag and restores UI elements (spectrogram,
        level meter) to their state before monitoring started. This
        ensures the UI returns to user preferences.
        """
        self.is_monitoring = False
        self._restore_ui_state_after_monitoring()

        self.saved_meters_state = None

        self.app.display_controller.set_status("", MsgType.DEFAULT)
        self.app.display_controller.set_monitoring_var(False)

        self.app.display_controller.show_saved_recording()
        if self.app.display_controller.is_info_panel_visible():
            self.app.display_controller.update_info_panel()

    def _restore_ui_state_after_monitoring(self) -> None:
        """Restore UI state after monitoring mode."""
        # Restore meters state (both are controlled together)
        # Restore meters state if it was changed during monitoring
        if (
            self.saved_meters_state is not None
            and self.app.window
            and hasattr(self.app.window, "meters_visible")
            and self.saved_meters_state != self.app.window.meters_visible
        ):
            self.app.display_controller.toggle_meters()

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
        """Handle post-save operations after a recording is written to disk.

        This method is called after the audio file has been successfully saved.
        It updates the UI to show the newly saved recording and refreshes the
        file list to include it.

        Args:
            label: The label of the recording that was saved
        """
        if not self.app.active_recordings:
            return

        # Invalidate cache since we have a new recording
        if self.app.active_recordings:
            self.app.active_recordings.on_recording_completed(label)
            self.app.state.recording.takes = self.app.active_recordings.get_all_takes()

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

            self.app.navigation_controller.update_take_status()

    def start_audio_queue_processing(self) -> None:
        """Start processing audio queue for real-time display.

        Delegates to AudioQueueProcessor for the actual processing logic.
        """
        self.audio_queue_processor.start()

    def update_audio_queue_state(self) -> None:
        """Update audio queue processing state based on what's visible.

        Delegates to AudioQueueProcessor to update the processing state.
        """
        self.audio_queue_processor.update_state()

    def stop_audio_queue_processing(self) -> None:
        """Stop audio queue processing.

        Delegates to AudioQueueProcessor to stop the processing thread.
        """
        self.audio_queue_processor.stop()
