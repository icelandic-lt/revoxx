"""Navigation controller for utterance and take management."""

from typing import TYPE_CHECKING

from ..constants import FileConstants, MsgType

if TYPE_CHECKING:
    from ..app import Revoxx


class NavigationController:
    """Handles navigation through utterances and takes.

    This controller manages:
    - Navigation between utterances
    - Browsing through different takes
    - Finding specific utterances
    - Resuming at last recording
    - Display position tracking
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the navigation controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app

    def navigate(self, direction: int) -> None:
        """Navigate to next/previous utterance.

        Args:
            direction: Navigation direction (+1 for next, -1 for previous)
        """
        # Stop any current activity
        if self.app.state.recording.is_recording:
            self.app.audio_controller.stop_recording()

        # Stop all playback activities (same as when starting new playback)
        self.app.audio_controller.stop_all_playback_activities()

        # Clear undo history and selection when switching clips
        self.app.edit_controller.clear_undo_history()
        self.app.display_controller.clear_selections()

        # Check if we have a session loaded
        if not self.app.active_recordings:
            # No session loaded, can't navigate
            return

        new_index = self.app.active_recordings.navigate(
            self.app.state.recording.current_index, direction
        )

        if new_index is None:
            return  # No more utterances in that direction

        # Update to new index
        self.app.state.recording.current_index = new_index

        # Set to the highest available take for this utterance
        current_label = self.app.state.recording.current_label
        if current_label:
            if self.app.active_recordings:
                highest_take = self.app.active_recordings.get_highest_take(
                    current_label
                )
            else:
                highest_take = 0
            self.app.state.recording.set_displayed_take(current_label, highest_take)

        # Show saved recording if available
        self.app.display_controller.show_saved_recording()

        # Update display
        self.app.display_controller.update_display()

        # Update take status
        self.update_take_status()

        # Update info panel if visible
        if self.app.window.info_panel_visible:
            self.app.display_controller.update_info_panel()

        # Recalculate font size after navigation
        self.app.display_controller.recalculate_window_font("main")

    def browse_takes(self, direction: int) -> None:
        """Browse through different takes.

        Args:
            direction: Browse direction (+1 for next take, -1 for previous take)
        """
        current_label = self.app.state.recording.current_label
        if not current_label:
            return

        # Stop all playback activities
        self.app.audio_controller.stop_all_playback_activities()

        # Clear undo history when switching takes
        self.app.edit_controller.clear_undo_history()

        # Get current take and all existing takes
        current_take = self.app.state.recording.get_current_take(current_label)
        if not self.app.active_recordings:
            existing_takes = []
        else:
            existing_takes = self.app.active_recordings.get_existing_takes(
                current_label
            )

        if not existing_takes:
            return

        # Find current position in the list
        try:
            current_index = existing_takes.index(current_take)
        except ValueError:
            # Current take not in list, find nearest
            current_index = 0
            for i, take in enumerate(existing_takes):
                if take > current_take:
                    current_index = max(0, i - 1)
                    break
                else:
                    current_index = i

        # Calculate new index
        new_index = current_index + direction

        # Check bounds and get new take
        if 0 <= new_index < len(existing_takes):
            new_take = existing_takes[new_index]
            self.app.state.recording.set_displayed_take(current_label, new_take)
            # Level meter reset is already handled by stop_all_playback_activities above
            self.app.display_controller.show_saved_recording()
            self.update_take_status()

            # Update info overlay if visible
            if self.app.window.info_panel_visible:
                self.app.display_controller.update_info_panel()

    def find_utterance(self, index: int) -> None:
        """Navigate directly to a specific utterance by index.

        Args:
            index: The utterance index to jump to
        """
        # Stop any current activity
        if self.app.state.recording.is_recording:
            self.app.audio_controller.stop_recording()

        self.app.audio_controller.stop_synchronized_playback()
        # Mel spectrogram widget might not exist for the first 100ms ...
        if self.app.window.mel_spectrogram:
            self.app.window.mel_spectrogram.stop_playback()

        # Stop second window spectrogram playback if active
        if self.app.has_active_second_window:
            second = self.app.window_manager.get_window("second")
            if second and second.mel_spectrogram:
                second.mel_spectrogram.stop_playback()

        # Reset level meter and selections
        try:
            self.app.shared_state.reset_level_meter()
        except AttributeError:
            # Shared state or method might not exist
            pass

        self.app.display_controller.clear_selections()

        # Update index
        if 0 <= index < len(self.app.state.recording.utterances):
            self.app.state.recording.current_index = index

            # Set to the highest available take for this utterance
            current_label = self.app.state.recording.current_label
            if current_label:
                highest_take = self.app.active_recordings.get_highest_take(
                    current_label
                )
                self.app.state.recording.set_displayed_take(current_label, highest_take)

            # Show saved recording if available
            self.app.display_controller.show_saved_recording()

            # Update display
            self.app.display_controller.update_display()

            # Update take status
            self.update_take_status()

            # Update info overlay if visible
            if self.app.window.info_panel_visible:
                self.app.display_controller.update_info_panel()

    def resume_session_position(self) -> None:
        """Resume at the last viewed position, falling back to last recorded."""
        if not self.app.current_session:
            return

        session = self.app.current_session
        total = len(self.app.state.recording.utterances)

        # Try last viewed position first
        if (
            session.last_viewed_index is not None
            and 0 <= session.last_viewed_index < total
        ):
            self.find_utterance(session.last_viewed_index)
            current_label = self.app.state.recording.current_label
            self.app.display_controller.set_status(f"Resumed at: {current_label}")
            return

        # Fall back to last recorded position
        self._navigate_to_last_recorded()

    def go_to_last_recorded(self) -> None:
        """Navigate to the last recorded utterance."""
        if not self.app.current_session:
            return

        if self._navigate_to_last_recorded():
            return

        self.app.display_controller.set_status(
            "No recorded utterance to navigate to", MsgType.TEMPORARY
        )

    def _navigate_to_last_recorded(self) -> bool:
        """Navigate to the last recorded utterance and set its take.

        Returns:
            True if navigation was performed, False otherwise
        """
        session = self.app.current_session
        if not session:
            return False

        if session.last_recorded_index is None or session.last_recorded_take is None:
            return False

        total = len(self.app.state.recording.utterances)
        if not (0 <= session.last_recorded_index < total):
            return False

        self.find_utterance(session.last_recorded_index)

        current_label = self.app.state.recording.current_label
        existing_takes = self.app.active_recordings.get_existing_takes(current_label)

        if session.last_recorded_take in existing_takes:
            current_take = self.app.state.recording.get_current_take(current_label)
            if current_take != session.last_recorded_take:
                self.app.state.recording.set_displayed_take(
                    current_label, session.last_recorded_take
                )
                self.app.display_controller.show_saved_recording()
                self.update_take_status()

        self.app.display_controller.set_status(
            f"Last recording: {current_label}", MsgType.TEMPORARY
        )
        return True

    def get_display_position(self, actual_index: int) -> int:
        """Get the display position for an actual index.

        Args:
            actual_index: The actual index in the utterances list

        Returns:
            The display position (1-based) in the current order
        """
        if not self.app.active_recordings:
            # Without active_recordings, display position is actual position + 1
            return actual_index + 1
        return self.app.active_recordings.get_display_position(actual_index)

    def update_take_status(self) -> None:
        """Update the take status display with relative position."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            return

        # Update label with filename if we have a recording
        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take > 0:
            filename = f"take_{current_take:03d}{FileConstants.AUDIO_FILE_EXTENSION}"
            self.app.window.update_label_with_filename(current_label, filename)
            # Update second window if active
            if self.app.has_active_second_window:
                second = self.app.window_manager.get_window("second")
                if second:
                    second.update_label_with_filename(current_label, filename)
        else:
            self.app.window.update_label_with_filename(current_label)
            # Update second window if active
            if self.app.has_active_second_window:
                second = self.app.window_manager.get_window("second")
                if second:
                    second.update_label_with_filename(current_label)

        status_text = self.app.display_controller.format_take_status(current_label)
        self.app.display_controller.set_status(status_text, MsgType.DEFAULT)
        self.app.display_controller.update_flag_indicator(current_label)

    def after_recording_saved(self, label: str) -> None:
        """Called after a recording has been saved to disk.

        Args:
            label: The label of the recording that was saved
        """
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
            else:
                highest_take = 0
            if highest_take > 0:
                self.app.state.recording.set_displayed_take(current_label, highest_take)

                # Save this as the last recorded utterance in the session
                self.app.current_session.last_recorded_index = (
                    self.app.state.recording.current_index
                )
                self.app.current_session.last_recorded_take = highest_take
                self.app.current_session.save()

        # Show the saved recording
        self.app.display_controller.show_saved_recording()

        # Update take status display
        self.update_take_status()

        # Update info panel if visible
        if self.app.window.info_panel_visible:
            self.app.display_controller.update_info_panel()
