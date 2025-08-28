"""Tests for the SessionController."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from revoxx.controllers.session_controller import SessionController


class TestSessionController(unittest.TestCase):
    """Test cases for SessionController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()

        # Mock state
        self.mock_app.state = Mock()
        self.mock_app.state.recording = Mock()
        self.mock_app.state.recording.labels = []
        self.mock_app.state.recording.utterances = []
        self.mock_app.state.recording.takes = {}
        self.mock_app.state.recording.current_index = 0

        # Mock window
        self.mock_app.window = Mock()
        self.mock_app.window.info_overlay = Mock()
        self.mock_app.window.info_overlay.visible = False

        # Mock config
        self.mock_app.config = Mock()
        self.mock_app.config.audio = Mock()
        self.mock_app.config.audio.sample_rate = 48000
        self.mock_app.config.audio.bit_depth = 24
        self.mock_app.config.audio.channels = 1
        self.mock_app.config.audio.input_device = None
        self.mock_app.config.audio.__post_init__ = Mock()

        # Mock current session
        self.mock_app.current_session = None
        self.mock_app.script_file = None
        self.mock_app.recording_dir = None

        # Mock file manager
        self.mock_app.file_manager = Mock()
        self.mock_app.active_recordings = Mock()
        self.mock_app.active_recordings.set_data = Mock()
        self.mock_app.active_recordings.set_sort = Mock()
        self.mock_app.active_recordings.get_all_takes = Mock(return_value={})

        # Mock script manager
        self.mock_app.script_manager = Mock()
        self.mock_app.script_manager.load_script = Mock(
            return_value=(["label1"], ["utterance1"])
        )

        # Mock navigation controller
        self.mock_app.navigation_controller = Mock()

        # Mock shared state
        self.mock_app.shared_state = Mock()

        # Mock settings manager
        self.mock_app.settings_manager = Mock()

        # Mock root window
        self.mock_app.root = Mock()

        # Mock update methods
        self.mock_app._update_info_overlay = Mock()

        self.controller = SessionController(self.mock_app)

    @patch("revoxx.controllers.session_controller.NewSessionDialog")
    @patch("revoxx.controllers.session_controller.Path")
    def test_new_session_success(self, mock_path, mock_dialog_class):
        """Test successful new session creation."""
        # Setup
        mock_dialog = Mock()
        mock_dialog_class.return_value = mock_dialog

        mock_result = Mock()
        mock_result.base_dir = Path("/test/base")
        mock_result.speaker_name = "Test Speaker"
        mock_result.gender = "M"
        mock_result.emotion = "neutral"
        mock_result.sample_rate = 48000
        mock_result.bit_depth = 24
        mock_result.recording_format = "wav"
        mock_result.input_device = None
        mock_result.script_path = Path("/test/script.txt")
        mock_result.custom_dir_name = None

        mock_dialog.show.return_value = mock_result

        mock_session = Mock()
        mock_session.session_dir = Mock()
        mock_session.session_dir.name = "test_session"

        self.controller.session_manager.create_session = Mock(return_value=mock_session)

        with patch.object(self.controller, "load_session") as mock_load:
            # Execute
            self.controller.new_session()

            # Verify
            mock_dialog_class.assert_called_once()
            self.controller.session_manager.create_session.assert_called_once()
            mock_load.assert_called_once_with(mock_session)
            self.mock_app.window.update_session_title.assert_called_once_with(
                "test_session"
            )
            self.mock_app.window.set_status.assert_called_with(
                "Created new session: test_session"
            )

    @patch("revoxx.controllers.session_controller.NewSessionDialog")
    def test_new_session_cancelled(self, mock_dialog_class):
        """Test new session creation when dialog is cancelled."""
        # Setup
        mock_dialog = Mock()
        mock_dialog_class.return_value = mock_dialog
        mock_dialog.show.return_value = None  # Cancelled

        # Mock create_session as Mock object
        self.controller.session_manager.create_session = Mock()

        # Execute
        self.controller.new_session()

        # Verify
        self.controller.session_manager.create_session.assert_not_called()
        self.mock_app.window.update_session_title.assert_not_called()

    @patch("revoxx.controllers.session_controller.NewSessionDialog")
    def test_new_session_error(self, mock_dialog_class):
        """Test new session creation with error."""
        # Setup
        mock_dialog = Mock()
        mock_dialog_class.return_value = mock_dialog
        mock_result = Mock()
        mock_dialog.show.return_value = mock_result

        # Use OSError which is one of the caught exceptions
        self.controller.session_manager.create_session = Mock(
            side_effect=OSError("Test error")
        )

        # Execute
        self.controller.new_session()

        # Verify
        self.mock_app.window.set_status.assert_called_with(
            "Error creating session: Test error"
        )

    def test_open_session_success(self):
        """Test successful opening of a session from path."""
        # Setup
        session_path = Path("/test/session.revoxx")
        mock_session = Mock()
        mock_session.session_dir = Mock()
        mock_session.session_dir.name = "test_session"

        self.controller.session_manager.load_session = Mock(return_value=mock_session)

        with patch.object(self.controller, "load_session") as mock_load:
            # Execute
            self.controller.open_session(session_path)

            # Verify
            self.controller.session_manager.load_session.assert_called_once_with(
                session_path
            )
            mock_load.assert_called_once_with(mock_session)
            self.mock_app.window.update_session_title.assert_called_once_with(
                "test_session"
            )
            self.mock_app.window.set_status.assert_called_with(
                "Loaded session: test_session"
            )

    def test_open_session_error(self):
        """Test opening session with error."""
        # Setup
        session_path = Path("/test/session.revoxx")
        # Use json.JSONDecodeError which is one of the caught exceptions
        import json

        self.controller.session_manager.load_session = Mock(
            side_effect=json.JSONDecodeError("Load error", "test", 0)
        )

        # Execute
        self.controller.open_session(session_path)

        # Verify
        # JSONDecodeError message includes more details, so we check if it contains our error
        status_call = self.mock_app.window.set_status.call_args[0][0]
        self.assertIn("Error loading session:", status_call)
        self.assertIn("Load error", status_call)

    def test_get_recent_sessions(self):
        """Test getting recent sessions."""
        # Setup
        mock_sessions = [Path("/test/session1.revoxx"), Path("/test/session2.revoxx")]
        self.controller.session_manager.get_recent_sessions = Mock(
            return_value=mock_sessions
        )

        # Execute
        result = self.controller.get_recent_sessions()

        # Verify
        self.assertEqual(result, mock_sessions)

    def test_get_current_session(self):
        """Test getting current session."""
        # Setup
        mock_session = Mock()
        self.mock_app.current_session = mock_session

        # Execute
        result = self.controller.get_current_session()

        # Verify
        self.assertEqual(result, mock_session)

    @patch("revoxx.controllers.session_controller.RecordingFileManager")
    @patch("revoxx.controllers.session_controller.ActiveRecordings")
    def test_load_session(self, mock_ar_class, mock_fm_class):
        """Test loading a session."""
        # Setup
        mock_session = Mock()
        mock_session.session_dir = Path("/test/session.revoxx")
        mock_session.sort_column = "label"
        mock_session.sort_reverse = False
        mock_session.audio_config = Mock()
        mock_session.audio_config.sample_rate = 44100
        mock_session.audio_config.bit_depth = 16

        mock_fm = Mock()
        mock_fm_class.return_value = mock_fm

        mock_ar = Mock()
        mock_ar_class.return_value = mock_ar

        with patch.object(
            self.controller, "reload_script_and_recordings"
        ) as mock_reload:
            # Execute
            self.controller.load_session(mock_session)

            # Verify
            self.assertEqual(self.mock_app.current_session, mock_session)
            # Use Path for platform-independent path comparison
            expected_path = Path("/test/session.revoxx/script.txt")
            actual_path = Path(str(self.mock_app.script_file))
            self.assertEqual(actual_path, expected_path)
            # Use Path for platform-independent path comparison
            expected_rec_path = Path("/test/session.revoxx/recordings")
            actual_rec_path = Path(str(self.mock_app.recording_dir))
            self.assertEqual(actual_rec_path, expected_rec_path)
            mock_fm_class.assert_called_once_with(
                Path("/test/session.revoxx/recordings")
            )
            mock_ar_class.assert_called_once_with(mock_fm)
            mock_ar.set_sort.assert_called_once_with("label", False)
            mock_reload.assert_called_once()
            # resume_at_last_recording is called in app.py after load_session, not in the controller itself
            self.assertEqual(self.mock_app.config.audio.sample_rate, 44100)
            self.assertEqual(self.mock_app.config.audio.bit_depth, 16)

    def test_load_script_no_session(self):
        """Test loading script with no session."""
        # Execute
        self.controller.load_script()

        # Verify
        self.assertEqual(self.mock_app.state.recording.labels, [])
        self.assertEqual(self.mock_app.state.recording.utterances, [])
        self.assertEqual(self.mock_app.state.recording.takes, {})

    def test_load_script_with_session(self):
        """Test loading script with valid session."""
        # Setup
        self.mock_app.current_session = Mock()
        mock_script_file = Mock()
        mock_script_file.exists = Mock(return_value=True)
        self.mock_app.script_file = mock_script_file

        with patch.object(
            self.controller, "reload_script_and_recordings"
        ) as mock_reload:
            # Execute
            self.controller.load_script()

            # Verify
            mock_reload.assert_called_once()
            self.assertEqual(self.mock_app.state.recording.current_index, 0)

    def test_load_script_missing_file(self):
        """Test loading script with missing file."""
        # Setup
        self.mock_app.current_session = Mock()
        mock_script_file = Mock()
        mock_script_file.exists = Mock(return_value=False)
        self.mock_app.script_file = mock_script_file

        # Execute & Verify
        with self.assertRaises(FileNotFoundError):
            self.controller.load_script()

    def test_reload_script_and_recordings_success(self):
        """Test successful reload of script and recordings."""
        # Setup
        mock_script_file = Mock()
        mock_script_file.exists = Mock(return_value=True)
        self.mock_app.script_file = mock_script_file

        # Execute
        self.controller.reload_script_and_recordings()

        # Verify
        self.mock_app.script_manager.load_script.assert_called_once_with(
            mock_script_file
        )
        self.assertEqual(self.mock_app.state.recording.labels, ["label1"])
        self.assertEqual(self.mock_app.state.recording.utterances, ["utterance1"])
        self.mock_app.active_recordings.set_data.assert_called_once_with(
            ["label1"], ["utterance1"]
        )
        self.mock_app.active_recordings.get_all_takes.assert_called_once()

    def test_reload_script_and_recordings_missing_file(self):
        """Test reload with missing script file."""
        # Setup
        self.mock_app.script_file = None

        # Execute
        self.controller.reload_script_and_recordings()

        # Verify
        self.assertEqual(self.mock_app.state.recording.labels, [])
        self.assertEqual(self.mock_app.state.recording.utterances, [])
        self.assertEqual(self.mock_app.state.recording.takes, {})
        self.mock_app.script_manager.load_script.assert_not_called()

    def test_reload_script_and_recordings_error(self):
        """Test reload with error during script loading."""
        # Setup
        mock_script_file = Mock()
        mock_script_file.exists = Mock(return_value=True)
        self.mock_app.script_file = mock_script_file
        # Use ValueError which is one of the caught exceptions
        self.mock_app.script_manager.load_script = Mock(
            side_effect=ValueError("Parse error")
        )

        # Execute
        self.controller.reload_script_and_recordings()

        # Verify
        self.assertEqual(self.mock_app.state.recording.labels, [])
        self.assertEqual(self.mock_app.state.recording.utterances, [])
        self.assertEqual(self.mock_app.state.recording.takes, {})


if __name__ == "__main__":
    unittest.main()
