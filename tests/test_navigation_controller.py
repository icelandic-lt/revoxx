"""Tests for the NavigationController."""

import unittest
from unittest.mock import Mock, patch

from revoxx.controllers.navigation_controller import NavigationController


class TestNavigationController(unittest.TestCase):
    """Test cases for NavigationController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()

        # Mock state
        self.mock_app.state = Mock()
        self.mock_app.state.recording = Mock()
        self.mock_app.state.recording.is_recording = False
        self.mock_app.state.recording.current_index = 0
        self.mock_app.state.recording.current_label = "test_label"
        self.mock_app.state.recording.utterances = [
            "utterance1",
            "utterance2",
            "utterance3",
        ]
        self.mock_app.state.recording.get_current_take = Mock(return_value=1)
        self.mock_app.state.recording.set_displayed_take = Mock()
        self.mock_app.state.recording.takes = {}

        # Mock window
        self.mock_app.window = Mock()
        self.mock_app.window.mel_spectrogram = Mock()
        self.mock_app.window.info_overlay = Mock()
        self.mock_app.window.info_overlay.visible = False
        self.mock_app.window.update_label_with_filename = Mock()
        self.mock_app.window.set_status = Mock()

        # Mock audio controller
        self.mock_app.audio_controller = Mock()
        self.mock_app.audio_controller.stop_all_playback_activities = Mock()
        self.mock_app.audio_controller.stop_recording = Mock()
        self.mock_app.audio_controller.stop_synchronized_playback = Mock()

        # Mock active recordings
        self.mock_app.active_recordings = Mock()
        self.mock_app.active_recordings.navigate = Mock(return_value=1)
        self.mock_app.active_recordings.get_highest_take = Mock(return_value=2)
        self.mock_app.active_recordings.get_existing_takes = Mock(
            return_value=[1, 2, 3]
        )
        self.mock_app.active_recordings.get_display_position = Mock(return_value=1)
        self.mock_app.active_recordings.get_all_takes = Mock(
            return_value={"test_label": 3}
        )
        self.mock_app.active_recordings.on_recording_completed = Mock()

        # Mock session
        self.mock_app.current_session = Mock()
        self.mock_app.current_session.last_recorded_index = None
        self.mock_app.current_session.last_recorded_take = None
        self.mock_app.current_session.save = Mock()

        # Mock shared state
        self.mock_app.shared_state = Mock()

        # Mock other methods (legacy - kept for compatibility)
        self.mock_app._show_saved_recording = Mock()
        self.mock_app._update_display = Mock()
        self.mock_app._update_info_overlay = Mock()

        # Mock display controller for new architecture
        self.mock_app.display_controller = Mock()
        self.mock_app.display_controller.show_saved_recording = Mock()
        self.mock_app.display_controller.update_display = Mock()
        self.mock_app.display_controller.update_info_overlay = Mock()

        self.controller = NavigationController(self.mock_app)

    def test_navigate_forward(self):
        """Test navigating to the next utterance."""
        self.controller.navigate(1)

        # Verify navigation
        self.mock_app.audio_controller.stop_all_playback_activities.assert_called_once()
        self.mock_app.active_recordings.navigate.assert_called_once_with(0, 1)
        self.assertEqual(self.mock_app.state.recording.current_index, 1)
        self.mock_app.active_recordings.get_highest_take.assert_called_once()
        self.mock_app.state.recording.set_displayed_take.assert_called_once_with(
            "test_label", 2
        )
        self.mock_app.display_controller.show_saved_recording.assert_called_once()
        self.mock_app.display_controller.update_display.assert_called_once()

    def test_navigate_backward(self):
        """Test navigating to the previous utterance."""
        self.mock_app.state.recording.current_index = 2
        self.mock_app.active_recordings.navigate.return_value = 1

        self.controller.navigate(-1)

        # Verify navigation
        self.mock_app.active_recordings.navigate.assert_called_once_with(2, -1)
        self.assertEqual(self.mock_app.state.recording.current_index, 1)

    def test_navigate_stops_recording_if_active(self):
        """Test that navigation stops recording if it's active."""
        self.mock_app.state.recording.is_recording = True

        self.controller.navigate(1)

        self.mock_app.audio_controller.stop_recording.assert_called_once()

    def test_navigate_returns_if_no_more_utterances(self):
        """Test that navigation returns early if no more utterances."""
        self.mock_app.active_recordings.navigate.return_value = None

        self.controller.navigate(1)

        # Should return without updating index
        self.assertEqual(self.mock_app.state.recording.current_index, 0)
        self.mock_app._update_display.assert_not_called()

    def test_browse_takes_forward(self):
        """Test browsing to the next take."""
        self.mock_app.state.recording.get_current_take.return_value = 1

        self.controller.browse_takes(1)

        # Verify take browsing
        self.mock_app.audio_controller.stop_all_playback_activities.assert_called_once()
        # get_existing_takes is called twice: once in browse_takes and once in update_take_status
        self.assertEqual(
            self.mock_app.active_recordings.get_existing_takes.call_count, 2
        )
        self.mock_app.state.recording.set_displayed_take.assert_called_once_with(
            "test_label", 2
        )
        self.mock_app.display_controller.show_saved_recording.assert_called_once()

    def test_browse_takes_backward(self):
        """Test browsing to the previous take."""
        self.mock_app.state.recording.get_current_take.return_value = 2

        self.controller.browse_takes(-1)

        self.mock_app.state.recording.set_displayed_take.assert_called_once_with(
            "test_label", 1
        )

    def test_browse_takes_no_more_forward(self):
        """Test browsing when no more takes forward."""
        self.mock_app.state.recording.get_current_take.return_value = 3

        self.controller.browse_takes(1)

        self.mock_app.display_controller.set_status.assert_called_once_with(
            "No more takes forward"
        )
        self.mock_app.state.recording.set_displayed_take.assert_not_called()

    def test_browse_takes_no_more_backward(self):
        """Test browsing when no more takes backward."""
        self.mock_app.state.recording.get_current_take.return_value = 1

        self.controller.browse_takes(-1)

        self.mock_app.display_controller.set_status.assert_called_once_with(
            "No more takes backward"
        )
        self.mock_app.state.recording.set_displayed_take.assert_not_called()

    def test_browse_takes_with_no_recordings(self):
        """Test browsing takes when no recordings exist."""
        self.mock_app.active_recordings.get_existing_takes.return_value = []

        self.controller.browse_takes(1)

        self.mock_app.state.recording.set_displayed_take.assert_not_called()

    def test_find_utterance_valid_index(self):
        """Test finding a specific utterance with valid index."""
        self.controller.find_utterance(2)

        # Verify finding
        self.mock_app.audio_controller.stop_synchronized_playback.assert_called_once()
        self.assertEqual(self.mock_app.state.recording.current_index, 2)
        self.mock_app.active_recordings.get_highest_take.assert_called_once()
        self.mock_app.display_controller.show_saved_recording.assert_called_once()
        self.mock_app.display_controller.update_display.assert_called_once()

    def test_find_utterance_invalid_index(self):
        """Test finding utterance with invalid index."""
        self.controller.find_utterance(10)  # Out of range

        # Should not update index
        self.assertEqual(self.mock_app.state.recording.current_index, 0)
        self.mock_app._update_display.assert_not_called()

    def test_find_utterance_stops_recording(self):
        """Test that find_utterance stops recording if active."""
        self.mock_app.state.recording.is_recording = True

        self.controller.find_utterance(1)

        self.mock_app.audio_controller.stop_recording.assert_called_once()

    def test_resume_at_last_recording_with_valid_session(self):
        """Test resuming at last recording with valid session data."""
        self.mock_app.current_session.last_recorded_index = 1
        self.mock_app.current_session.last_recorded_take = 2

        with patch.object(self.controller, "find_utterance") as mock_find:
            self.controller.resume_at_last_recording()
            mock_find.assert_called_once_with(1)

        self.mock_app.display_controller.set_status.assert_called_with(
            "Resumed at last recording: test_label"
        )

    def test_resume_at_last_recording_no_session(self):
        """Test resuming when no session exists."""
        self.mock_app.current_session = None

        self.controller.resume_at_last_recording()

        # Should return early
        self.mock_app.display_controller.set_status.assert_not_called()

    def test_resume_at_last_recording_no_last_index(self):
        """Test resuming when no last recorded index."""
        self.mock_app.current_session.last_recorded_index = None

        self.controller.resume_at_last_recording()

        # Should return early
        self.mock_app.display_controller.set_status.assert_not_called()

    def test_get_display_position(self):
        """Test getting display position for an index."""
        result = self.controller.get_display_position(5)

        self.mock_app.active_recordings.get_display_position.assert_called_once_with(5)
        self.assertEqual(result, 1)

    def test_update_take_status_with_recording(self):
        """Test updating take status when recording exists."""
        self.mock_app.state.recording.get_current_take.return_value = 2

        self.controller.update_take_status()

        # Verify status update
        self.mock_app.window.update_label_with_filename.assert_called_once()
        self.mock_app.display_controller.set_status.assert_called_once_with(
            "test_label - Take 2/3"
        )

    def test_update_take_status_no_recordings(self):
        """Test updating take status when no recordings exist."""
        self.mock_app.state.recording.get_current_take.return_value = 0
        self.mock_app.active_recordings.get_existing_takes.return_value = []

        self.controller.update_take_status()

        self.mock_app.display_controller.set_status.assert_called_once_with(
            "test_label"
        )

    def test_update_take_status_no_label(self):
        """Test updating take status when no current label."""
        self.mock_app.state.recording.current_label = None

        self.controller.update_take_status()

        # Should return early
        self.mock_app.display_controller.set_status.assert_not_called()

    def test_after_recording_saved(self):
        """Test after_recording_saved updates state correctly."""
        self.controller.after_recording_saved("test_label")

        # Verify updates
        self.mock_app.active_recordings.on_recording_completed.assert_called_once_with(
            "test_label"
        )
        self.mock_app.active_recordings.get_all_takes.assert_called_once()
        self.mock_app.active_recordings.get_highest_take.assert_called_once_with(
            "test_label"
        )
        self.mock_app.state.recording.set_displayed_take.assert_called_once()
        self.mock_app.current_session.save.assert_called_once()
        self.mock_app.display_controller.show_saved_recording.assert_called_once()

    def test_after_recording_saved_different_label(self):
        """Test after_recording_saved with different label than current."""
        self.controller.after_recording_saved("other_label")

        # Should update takes but not set displayed take for current label
        self.mock_app.active_recordings.on_recording_completed.assert_called_once()
        self.mock_app.active_recordings.get_all_takes.assert_called_once()
        self.mock_app.active_recordings.get_highest_take.assert_not_called()
        self.mock_app.current_session.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
