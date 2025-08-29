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

        # Mock window manager
        mock_window_manager = Mock()
        mock_window_manager.get_active_windows.return_value = [self.mock_app.window]
        mock_window_manager.get_window.return_value = None
        mock_window_manager.broadcast.return_value = []
        mock_window_manager.execute_on_windows.return_value = []

        self.controller = DisplayController(self.mock_app, mock_window_manager)

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

    @patch("soundfile.SoundFile")
    def test_show_saved_recording_exists(self, mock_soundfile):
        """Test showing a saved recording that exists."""
        # Setup soundfile mock for info panel update
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.samplerate = 48000
        mock_file.channels = 1
        mock_file.__len__.return_value = 48000
        mock_soundfile.return_value = mock_file

        self.controller.show_saved_recording()

        # get_recording_path is called twice: once for loading, once for info panel
        self.assertEqual(self.mock_app.file_manager.get_recording_path.call_count, 2)
        self.mock_app.file_manager.get_recording_path.assert_any_call("test_label", 1)
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
        from revoxx.constants import MsgType

        self.mock_app.file_manager.load_audio.side_effect = OSError("File error")

        self.controller.show_saved_recording()

        self.mock_app.window.set_status.assert_called_once_with(
            "Error loading recording: File error", MsgType.ERROR
        )

    def test_toggle_meters_show(self):
        """Test toggling meters to show."""
        # Setup window with meters_visible = False
        self.mock_app.window.meters_visible = False
        self.mock_app.window.set_meters_visibility = Mock()
        self.mock_app.audio_controller = Mock()
        self.mock_app.root = Mock()

        self.controller.toggle_meters()

        # Check window's meters_visible was toggled
        self.mock_app.window.set_meters_visibility.assert_called_once_with(True)
        self.mock_app.audio_controller.update_audio_queue_state.assert_called_once()
        # No longer updating global state, only window state

    def test_toggle_meters_hide(self):
        """Test toggling meters to hide."""
        # Setup window with meters_visible = True
        self.mock_app.window.meters_visible = True
        self.mock_app.window.set_meters_visibility = Mock()
        self.mock_app.audio_controller = Mock()
        self.mock_app.root = Mock()

        self.controller.toggle_meters()

        # Check window's meters_visible was toggled
        self.mock_app.window.set_meters_visibility.assert_called_once_with(False)
        self.mock_app.audio_controller.update_audio_queue_state.assert_called_once()
        # No longer updating global state, only window state

    def test_toggle_info_panel_show(self):
        """Test toggling info panel (method removed from DisplayController)."""
        # Method no longer exists - info panel toggling moved to window
        pass

    def test_toggle_info_panel_hide(self):
        """Test toggling info panel (method removed from DisplayController)."""
        # Method no longer exists - info panel toggling moved to window
        pass

    @patch("soundfile.SoundFile")
    def test_update_info_panel_with_recording(self, mock_soundfile):
        """Test updating info panel with a recording."""
        # Setup soundfile mock
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.samplerate = 48000
        mock_file.channels = 1
        mock_file.__len__.return_value = 48000  # 1 second of audio
        mock_soundfile.return_value = mock_file

        self.controller.update_info_panel()

        # Verify file info was retrieved
        self.mock_app.file_manager.get_recording_path.assert_called_once_with(
            "test_label", 1
        )

        # Verify info panel was updated with parameters
        self.mock_app.window.update_info_panel.assert_called_once()
        call_args = self.mock_app.window.update_info_panel.call_args
        params = call_args[0][0]
        self.assertEqual(params["sample_rate"], 48000)
        self.assertEqual(params["bit_depth"], 24)
        self.assertEqual(params["channels"], 1)
        self.assertIn("duration", params)
        self.assertIn("size", params)

    def test_update_info_panel_no_label(self):
        """Test updating info panel with no current label."""
        self.mock_app.state.recording.current_label = None

        self.controller.update_info_panel()

        self.mock_app.window.update_info_panel.assert_called_once()
        call_args = self.mock_app.window.update_info_panel.call_args
        params = call_args[0][0]
        self.assertEqual(params["sample_rate"], 48000)
        self.assertEqual(params["bit_depth"], 24)
        self.assertEqual(params["channels"], 1)

    def test_update_info_panel_no_recording(self):
        """Test updating info panel when no recording exists."""
        self.mock_app.state.recording.get_current_take.return_value = 0

        self.controller.update_info_panel()

        self.mock_app.file_manager.get_recording_path.assert_not_called()
        self.mock_app.window.update_info_panel.assert_called_once()

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

    def test_set_status(self):
        """Test setting the status."""
        from revoxx.constants import MsgType

        self.controller.set_status("Ready")

        self.mock_app.window.set_status.assert_called_once_with(
            "Ready", MsgType.TEMPORARY
        )

    def test_update_window_title_custom(self):
        """Test updating window title with custom text."""
        self.mock_app.window.window = Mock()
        self.controller.update_window_title("Custom Title")

        self.mock_app.window.window.title.assert_called_once_with("Custom Title")

    def test_update_window_title_default_with_session(self):
        """Test updating window title to default with session."""
        self.mock_app.window.window = Mock()
        self.controller.update_window_title()

        self.mock_app.window.window.title.assert_called_once_with(
            "Revoxx - Test Session"
        )

    def test_update_window_title_default_no_session(self):
        """Test updating window title to default without session."""
        self.mock_app.window.window = Mock()
        self.mock_app.current_session = None

        self.controller.update_window_title()

        self.mock_app.window.window.title.assert_called_once_with("Revoxx")


if __name__ == "__main__":
    unittest.main()
