"""Tests for the DisplayController."""

import unittest
from unittest.mock import Mock, MagicMock, patch

from revoxx.controllers.display_controller import DisplayController


class TestDisplayController(unittest.TestCase):
    """Test cases for DisplayController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()

        # Mock state
        self.mock_app.state = Mock()
        self.mock_app.state.recording = Mock()
        self.mock_app.state.recording.utterances = [
            {"id": "001", "text": "Test utterance 1"},
            {"id": "002", "text": "Test utterance 2"},
        ]
        self.mock_app.state.recording.current_index = 0
        self.mock_app.state.recording.current_utterance = {
            "id": "001",
            "text": "Test utterance 1",
        }
        self.mock_app.state.recording.current_label = "test_label"
        self.mock_app.state.recording.get_current_take = Mock(return_value=1)
        self.mock_app.state.recording.record_button_text = "Record"
        self.mock_app.state.recording.is_recording = False

        self.mock_app.state.ui = Mock()
        self.mock_app.state.ui.spectrogram_visible = False

        # Mock window
        self.mock_app.window = Mock()
        self.mock_app.window.mel_spectrogram = Mock()
        self.mock_app.window.info_overlay = Mock()
        self.mock_app.window.info_overlay.visible = False
        self.mock_app.window.recording_timer = Mock()
        self.mock_app.window.embedded_level_meter = Mock()

        # Mock navigation controller
        self.mock_app.navigation_controller = Mock()
        self.mock_app.navigation_controller.get_display_position = Mock(return_value=1)

        # Mock audio controller
        self.mock_app.audio_controller = Mock()
        self.mock_app.audio_controller.is_monitoring = False

        # Mock file manager
        self.mock_app.file_manager = Mock()
        mock_path = Mock()
        mock_path.exists = Mock(return_value=True)
        mock_path.stat = Mock()
        mock_path.stat.return_value.st_size = 1024
        self.mock_app.file_manager.get_recording_path = Mock(return_value=mock_path)
        self.mock_app.file_manager.load_audio = Mock(
            return_value=([0.1, 0.2, 0.3], 48000)
        )

        # Mock settings manager
        self.mock_app.settings_manager = Mock()

        # Mock config
        self.mock_app.config = Mock()
        self.mock_app.config.audio = Mock()
        self.mock_app.config.audio.sample_rate = 48000
        self.mock_app.config.audio.bit_depth = 24
        self.mock_app.config.audio.channels = 1

        # Mock root
        self.mock_app.root = Mock()

        # Mock current session
        self.mock_app.current_session = Mock()
        self.mock_app.current_session.name = "Test Session"

        self.controller = DisplayController(self.mock_app)

    def test_update_display_with_utterances(self):
        """Test updating display with utterances available."""
        self.controller.update_display()

        self.mock_app.navigation_controller.get_display_position.assert_called_once_with(
            0
        )
        # New architecture uses 3 parameters: index, is_recording, display_position
        self.mock_app.window.update_display.assert_called_once_with(
            0,  # current_index
            False,  # is_recording
            1,  # display_position
        )

    def test_update_display_no_utterances(self):
        """Test updating display with no utterances."""
        self.mock_app.state.recording.utterances = []
        self.mock_app.state.recording.current_index = 0
        self.mock_app.state.recording.is_recording = False
        # When no utterances, get_display_position should return 0
        self.mock_app.navigation_controller.get_display_position = Mock(return_value=0)

        self.controller.update_display()

        # Even with no utterances, update_display is called with current state
        self.mock_app.window.update_display.assert_called_once_with(
            0,  # current_index
            False,  # is_recording
            0,  # display_position (0 when no utterances)
        )

    def test_update_display_no_current_utterance(self):
        """Test updating display with navigation controller."""
        # New architecture uses navigation controller for display position
        self.mock_app.navigation_controller = Mock()
        self.mock_app.navigation_controller.get_display_position = Mock(return_value=1)
        self.mock_app.state.recording.current_index = 0
        self.mock_app.state.recording.is_recording = False

        self.controller.update_display()

        # update_display is called with 3 parameters in new architecture
        self.mock_app.window.update_display.assert_called_once_with(0, False, 1)

    def test_update_display_no_recording(self):
        """Test updating display when no recording exists."""
        self.mock_app.state.recording.get_current_take.return_value = 0
        self.mock_app.state.recording.current_index = 0
        self.mock_app.state.recording.is_recording = False

        self.controller.update_display()

        # New architecture doesn't distinguish recording existence in update_display
        self.mock_app.window.update_display.assert_called_once_with(
            0,  # current_index
            False,  # is_recording
            1,  # display_position
        )

    def test_show_saved_recording_exists(self):
        """Test showing a saved recording that exists."""
        self.controller.show_saved_recording()

        self.mock_app.file_manager.get_recording_path.assert_called_once_with(
            "test_label", 1
        )
        self.mock_app.file_manager.load_audio.assert_called_once()
        self.mock_app.window.mel_spectrogram.show_recording.assert_called_once_with(
            [0.1, 0.2, 0.3], 48000
        )

    def test_show_saved_recording_no_label(self):
        """Test showing saved recording with no current label."""
        self.mock_app.state.recording.current_label = None

        self.controller.show_saved_recording()

        self.mock_app.file_manager.get_recording_path.assert_not_called()

    def test_show_saved_recording_no_take(self):
        """Test showing saved recording when no take exists."""
        self.mock_app.state.recording.get_current_take.return_value = 0

        self.controller.show_saved_recording()

        self.mock_app.window.mel_spectrogram.clear.assert_called_once()
        self.mock_app.file_manager.get_recording_path.assert_not_called()

    def test_show_saved_recording_file_not_exists(self):
        """Test showing saved recording when file doesn't exist."""
        mock_path = Mock()
        mock_path.exists = Mock(return_value=False)
        self.mock_app.file_manager.get_recording_path.return_value = mock_path

        self.controller.show_saved_recording()

        self.mock_app.file_manager.load_audio.assert_not_called()
        self.mock_app.window.mel_spectrogram.show_recording.assert_not_called()

    def test_show_saved_recording_load_error(self):
        """Test showing saved recording with load error."""
        self.mock_app.file_manager.load_audio.side_effect = OSError("File error")

        self.controller.show_saved_recording()

        self.mock_app.window.set_status.assert_called_once_with(
            "Error loading recording: File error"
        )

    def test_toggle_mel_spectrogram_show(self):
        """Test toggling mel spectrogram to show."""
        self.mock_app.state.ui.spectrogram_visible = False
        self.mock_app.window.toggle_spectrogram = Mock()
        self.mock_app.audio_controller = Mock()
        self.mock_app.root = Mock()

        # Mock so that toggle_spectrogram sets visible to True
        def toggle_spec():
            self.mock_app.state.ui.spectrogram_visible = True

        self.mock_app.window.toggle_spectrogram.side_effect = toggle_spec

        self.controller.toggle_mel_spectrogram()

        self.mock_app.window.toggle_spectrogram.assert_called_once()
        self.assertTrue(self.mock_app.state.ui.spectrogram_visible)
        self.mock_app.settings_manager.update_setting.assert_called_once_with(
            "show_spectrogram", True
        )

    def test_toggle_mel_spectrogram_hide(self):
        """Test toggling mel spectrogram to hide."""
        self.mock_app.state.ui.spectrogram_visible = True
        self.mock_app.window.toggle_spectrogram = Mock()
        self.mock_app.audio_controller = Mock()
        self.mock_app.root = Mock()

        # Mock so that toggle_spectrogram sets visible to False
        def toggle_spec():
            self.mock_app.state.ui.spectrogram_visible = False

        self.mock_app.window.toggle_spectrogram.side_effect = toggle_spec

        self.controller.toggle_mel_spectrogram()

        self.mock_app.window.toggle_spectrogram.assert_called_once()
        self.assertFalse(self.mock_app.state.ui.spectrogram_visible)
        self.mock_app.settings_manager.update_setting.assert_called_once_with(
            "show_spectrogram", False
        )

    def test_toggle_info_overlay_show(self):
        """Test toggling info overlay to show."""
        self.mock_app.window.info_overlay.visible = False

        with patch.object(self.controller, "update_info_overlay") as mock_update:
            self.controller.toggle_info_overlay()
            mock_update.assert_called_once()

    def test_toggle_info_overlay_hide(self):
        """Test toggling info overlay to hide."""
        self.mock_app.window.info_overlay.visible = True

        self.controller.toggle_info_overlay()

        self.mock_app.window.info_overlay.hide.assert_called_once()

    @patch("soundfile.SoundFile")
    def test_update_info_overlay_with_recording(self, mock_soundfile):
        """Test updating info overlay with a recording."""
        # Setup soundfile mock
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.samplerate = 48000
        mock_file.channels = 1
        mock_file.__len__.return_value = 48000  # 1 second of audio
        mock_soundfile.return_value = mock_file

        self.controller.update_info_overlay()

        # Verify file info was retrieved
        self.mock_app.file_manager.get_recording_path.assert_called_once_with(
            "test_label", 1
        )

        # Verify info overlay was shown with parameters
        self.mock_app.window.info_overlay.show.assert_called_once()
        call_args = self.mock_app.window.info_overlay.show.call_args
        params = call_args[0][0]
        self.assertEqual(params["sample_rate"], 48000)
        self.assertEqual(params["bit_depth"], 24)
        self.assertEqual(params["channels"], 1)
        self.assertIn("duration", params)
        self.assertIn("file_size", params)

    def test_update_info_overlay_no_label(self):
        """Test updating info overlay with no current label."""
        self.mock_app.state.recording.current_label = None

        self.controller.update_info_overlay()

        self.mock_app.window.info_overlay.show.assert_called_once()
        call_args = self.mock_app.window.info_overlay.show.call_args
        params = call_args[0][0]
        self.assertEqual(params["sample_rate"], 48000)
        self.assertEqual(params["bit_depth"], 24)
        self.assertEqual(params["channels"], 1)

    def test_update_info_overlay_no_recording(self):
        """Test updating info overlay when no recording exists."""
        self.mock_app.state.recording.get_current_take.return_value = 0

        self.controller.update_info_overlay()

        self.mock_app.file_manager.get_recording_path.assert_not_called()
        self.mock_app.window.info_overlay.show.assert_called_once()

    def test_update_recording_timer(self):
        """Test updating the recording timer."""
        self.controller.update_recording_timer(5.5)

        self.mock_app.window.recording_timer.update.assert_called_once_with(5.5)

    def test_reset_recording_timer(self):
        """Test resetting the recording timer."""
        self.controller.reset_recording_timer()

        self.mock_app.window.recording_timer.reset.assert_called_once()

    def test_update_level_meter(self):
        """Test updating the level meter."""
        self.controller.update_level_meter(0.75)

        self.mock_app.window.embedded_level_meter.update_level.assert_called_once_with(
            0.75
        )

    def test_reset_level_meter(self):
        """Test resetting the level meter."""
        self.controller.reset_level_meter()

        self.mock_app.window.embedded_level_meter.reset.assert_called_once()

    def test_show_message(self):
        """Test showing a message."""
        self.controller.show_message("Test message", 3000)

        self.mock_app.window.show_message.assert_called_once_with("Test message", 3000)

    def test_show_message_with_custom_duration(self):
        """Test showing a message with custom duration."""
        self.controller.show_message("Test message", 5000)

        self.mock_app.window.show_message.assert_called_once_with("Test message", 5000)

    def test_set_status(self):
        """Test setting the status."""
        self.controller.set_status("Ready")

        self.mock_app.window.set_status.assert_called_once_with("Ready")

    def test_update_window_title_custom(self):
        """Test updating window title with custom text."""
        self.controller.update_window_title("Custom Title")

        self.mock_app.root.title.assert_called_once_with("Custom Title")

    def test_update_window_title_default_with_session(self):
        """Test updating window title to default with session."""
        self.controller.update_window_title()

        self.mock_app.root.title.assert_called_once_with("Revoxx - Test Session")

    def test_update_window_title_default_no_session(self):
        """Test updating window title to default without session."""
        self.mock_app.current_session = None

        self.controller.update_window_title()

        self.mock_app.root.title.assert_called_once_with("Revoxx")


if __name__ == "__main__":
    unittest.main()
