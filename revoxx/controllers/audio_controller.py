"""Audio controller for orchestrating recording and playback operations.

This controller acts as the high-level orchestrator for all audio-related
operations in the application. It coordinates between different subsystems
and manages the overall audio workflow.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING
import sounddevice as sd

from ..constants import UIConstants
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
        saved_spectrogram_state: Preserved spectrogram visibility state
        saved_level_meter_state: Preserved level meter visibility state
        audio_queue_processor: Handles audio data queue processing
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

        available = [d["index"] for d in device_manager.get_output_devices()]
        if self.app.config.audio.output_device not in available:
            self.app.window.set_status(
                "Selected output device not found. Using system default."
            )
            self.app.queue_manager.set_output_device(None)

    def _verify_input_device(self) -> None:
        """Verify input device is still available, fallback to default if not."""
        if self.app.config.audio.input_device is None:
            return

        device_manager = self._refresh_device_manager()
        if not device_manager:
            return

        available = [d["index"] for d in device_manager.get_input_devices()]
        if self.app.config.audio.input_device not in available:
            self.app.window.set_status(
                "Selected input device not found. Using system default."
            )
            self.app.queue_manager.set_input_device(None)

    def stop_all_playback_activities(self) -> None:
        """Stop all current playback and monitoring activities.

        Ensure consistent playback stopping behavior across navigation, recording,
        and playback.
        """
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
            pass

        # Give the playback process time to handle the stop command
        time.sleep(UIConstants.PLAYBACK_STOP_DELAY)
        # Also clear playback status to IDLE
        try:
            self.app.shared_state.stop_playback()
        except AttributeError:
            pass

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
        metadata = audio_buffer.get_metadata()

        # Send play command with buffer metadata
        self.app.queue_manager.start_playback(buffer_metadata=metadata, sample_rate=sr)

        # Close our reference but don't unlink - buffer manager handles lifecycle
        audio_buffer.close()

        # Start animations
        if hasattr(self.app.window, "mel_spectrogram"):
            self.app.window.mel_spectrogram.start_playback(duration, sr)

        # Note: Level meter updates during playback happen automatically
        # via shared memory from the playback process (AudioPlayer._update_level_meter)

    def play_current(self) -> None:
        """Play current recording."""
        if not self.app.state.is_ready_to_play():
            self.app.window.show_message("No recording available")
            return

        # Stop all current playback/monitoring
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
        self._do_start_audio_capture(mode)

    def _prepare_for_audio_capture(self, mode: str) -> None:
        """Prepare system for audio capture by stopping conflicting operations.

        Audio capture requires exclusive access to the audio input device.
        This method stops any operations that would conflict:
        - Active monitoring (if starting recording)
        - Active playback (can't record and play simultaneously)
        - Previous recording sessions

        This ensures clean state before starting new capture.
        """
        # Stop any active monitoring
        if self.is_monitoring:
            self.stop_monitoring_mode()

        # Stop playback only for recording
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
        self.app.manager_dict["save_path"] = str(save_path)
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

        # Reset level meter
        self._reset_level_meter()

    def _save_ui_state_for_monitoring(self) -> None:
        """Save current UI state before entering monitoring mode.

        Monitoring mode temporarily shows the spectrogram and level meter
        even if they were hidden. This method saves the current visibility
        state so the UI can be restored to user preferences when monitoring
        ends.
        """
        self.saved_spectrogram_state = self.app.state.ui.spectrogram_visible
        self.saved_level_meter_state = (
            self.app.window.level_meter_var.get()
            if hasattr(self.app.window, "level_meter_var")
            else False
        )

    def _show_monitoring_visualizations(self) -> None:
        """Enable visualizations for monitoring mode."""
        # Show spectrogram if not visible
        if not self.app.state.ui.spectrogram_visible:
            self.app.display_controller.toggle_mel_spectrogram()

        # Show level meter if not visible
        if (
            hasattr(self.app.window, "level_meter_var")
            and not self.app.window.level_meter_var.get()
        ):
            self.app.window.toggle_level_meter_callback()
            self.app.settings_manager.update_setting("show_level_meter", True)
            self.app.root.update_idletasks()

    def _reset_level_meter(self) -> None:
        """Reset level meter when entering monitoring mode."""
        if (
            hasattr(self.app.window, "embedded_level_meter")
            and self.app.window.embedded_level_meter
        ):
            try:
                self.app.window.embedded_level_meter.reset()
            except AttributeError:
                # Level meter might not exist or have reset method
                pass

    def _do_start_audio_capture(self, mode: str) -> None:
        """Execute the actual audio capture start.

        This is the final step that actually begins audio capture by:
        1. Clearing the spectrogram display for fresh data
        2. Starting spectrogram recording at current sample rate
        3. Sending start command to the recording process
        4. Updating status message in UI
        """
        # Clear and start spectrogram
        if hasattr(self.app.window, "mel_spectrogram"):
            self.app.window.mel_spectrogram.clear()
            self.app.window.mel_spectrogram.start_recording(
                self.app.config.audio.sample_rate
            )

        # Update info panel
        self._update_info_panel_for_capture(mode)

        # Handle device notifications and verify availability
        self.app.notify_if_default_device("input")
        self._verify_input_device()

        # Start audio capture
        self.app.queue_manager.start_recording()

        # Update UI based on mode
        if mode == "recording":
            self.app.display_controller.update_display()
        else:
            self.app.window.set_status("Monitoring input levels...")
            if hasattr(self.app.window, "monitoring_var"):
                self.app.window.monitoring_var.set(True)

    def _update_info_panel_for_capture(self, mode: str) -> None:
        """Update info panel for audio capture."""
        if self.app.window.info_panel_visible:
            recording_params = {
                "sample_rate": self.app.config.audio.sample_rate,
                "bit_depth": self.app.config.audio.bit_depth,
                "channels": self.app.config.audio.channels,
            }
            self.app.window.update_info_panel(recording_params)

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
        if hasattr(self.app.window, "mel_spectrogram"):
            self.app.window.mel_spectrogram.stop_recording()

    def _cleanup_recording_mode(self) -> None:
        """Clean up after recording mode ends.

        Resets recording-related state flags and updates UI to reflect
        that recording has stopped. This ensures the application state
        is consistent and ready for the next operation.
        """
        self.app.state.recording.is_recording = False

        current_label = self.app.state.recording.current_label
        if current_label:
            # Update displayed take & Schedule post-recording actions
            current_take = self.app.state.recording.get_take_count(current_label)
            self.app.state.recording.set_displayed_take(current_label, current_take)
            self.app.root.after(
                UIConstants.POST_RECORDING_DELAY_MS,
                lambda: self._after_recording_saved(current_label),
            )

        self.app.display_controller.update_display()
        if self.app.window.info_panel_visible:
            self.app.root.after(
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

        # Clear saved states
        self.saved_spectrogram_state = None
        self.saved_level_meter_state = None

        # Update UI
        self.app.window.set_status("Ready")
        if hasattr(self.app.window, "monitoring_var"):
            self.app.window.monitoring_var.set(False)

        self.app.display_controller.show_saved_recording()
        if self.app.window.info_panel_visible:
            self.app.display_controller.update_info_panel()

    def _restore_ui_state_after_monitoring(self) -> None:
        """Restore UI state after monitoring mode."""
        # Restore spectrogram state
        if (
            self.saved_spectrogram_state is not None
            and self.saved_spectrogram_state != self.app.state.ui.spectrogram_visible
        ):
            self.app.display_controller.toggle_mel_spectrogram()

        # Restore level meter state
        if (
            hasattr(self.app.window, "level_meter_var")
            and self.saved_level_meter_state is not None
        ):
            current_state = self.app.window.level_meter_var.get()
            if self.saved_level_meter_state != current_state:
                self.app.window.toggle_level_meter_callback()

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
