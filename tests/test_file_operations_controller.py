"""Tests for the FileOperationsController."""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import numpy as np

from revoxx.controllers.file_operations_controller import FileOperationsController
from revoxx.constants import MsgType


class TestFileOperationsController(unittest.TestCase):
    """Test cases for FileOperationsController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()

        # Mock state
        self.mock_app.state = Mock()
        self.mock_app.state.recording = Mock()
        self.mock_app.state.recording.current_label = "test_label"
        self.mock_app.state.recording.get_current_take = Mock(return_value=2)
        self.mock_app.state.recording.takes = {"test_label": 2, "other_label": 1}
        self.mock_app.state.recording.utterances = [
            {"id": "001", "text": "Test utterance 1"},
            {"id": "002", "text": "Test utterance 2"},
        ]

        # Mock window
        self.mock_app.window = Mock()
        self.mock_app.window.mel_spectrogram = Mock()
        self.mock_app.window.info_overlay = Mock()
        self.mock_app.window.info_overlay.visible = False

        # Mock file manager
        self.mock_app.file_manager = Mock()
        self.mock_app.file_manager.move_to_trash = Mock(return_value=True)
        self.mock_app.file_manager.restore_from_trash = Mock(return_value=True)
        self.mock_app.file_manager.get_deleted_takes = Mock(return_value=[1, 3])
        mock_path = Mock()
        mock_path.exists = Mock(return_value=True)
        mock_path.stat = Mock()
        mock_path.stat.return_value.st_size = 1024
        self.mock_app.file_manager.get_recording_path = Mock(return_value=mock_path)
        self.mock_app.file_manager.load_audio = Mock(
            return_value=(np.array([0.1, 0.2, 0.3]), 48000)
        )

        # Mock active recordings
        self.mock_app.active_recordings = Mock()
        self.mock_app.active_recordings.get_existing_takes = Mock(return_value=[1, 2])
        self.mock_app.active_recordings.get_highest_take = Mock(return_value=2)
        self.mock_app.active_recordings.get_all_takes = Mock(
            return_value={"test_label": 2, "other_label": 1}
        )

        # Mock navigation controller
        self.mock_app.navigation_controller = Mock()

        # Mock display controller
        self.mock_app.display_controller = Mock()

        # Mock current session
        self.mock_app.current_session = Mock()
        self.mock_app.current_session.recordings_dir = Path("/test/recordings")

        # Mock root for tkinter dialogs
        self.mock_app.root = Mock()

        self.controller = FileOperationsController(self.mock_app)

    @patch("revoxx.controllers.file_operations_controller.messagebox.askyesno")
    def test_delete_current_recording_success(self, mock_askyesno):
        """Test successful deletion of current recording."""
        mock_askyesno.return_value = True
        self.controller.delete_current_recording()

        # Verify file was moved to trash
        self.mock_app.file_manager.move_to_trash.assert_called_once_with(
            "test_label", 2
        )

        # Verify cache invalidation (implementation passes both label and take)
        self.mock_app.active_recordings.on_recording_deleted.assert_called_once_with(
            "test_label", 2
        )

        # Verify UI updates
        self.mock_app.window.mel_spectrogram.clear.assert_called_once()
        self.mock_app.display_controller.show_saved_recording.assert_called_once()
        self.mock_app.navigation_controller.update_take_status.assert_called_once()

        # Verify success message
        self.mock_app.window.set_status.assert_called_with(
            "Recording moved to trash: take_002", MsgType.TEMPORARY
        )

    def test_delete_current_recording_no_label(self):
        """Test deleting when no current label."""
        self.mock_app.state.recording.current_label = None

        self.controller.delete_current_recording()

        # Should return early
        self.mock_app.file_manager.move_to_trash.assert_not_called()

    def test_delete_current_recording_no_take(self):
        """Test deleting when no recording exists."""
        self.mock_app.state.recording.get_current_take.return_value = 0

        self.controller.delete_current_recording()

        self.mock_app.window.set_status.assert_called_once_with(
            "No recording to delete", MsgType.TEMPORARY
        )
        self.mock_app.file_manager.move_to_trash.assert_not_called()

    @patch("revoxx.controllers.file_operations_controller.messagebox.askyesno")
    def test_delete_current_recording_last_take(self, mock_askyesno):
        """Test deleting the last take."""
        mock_askyesno.return_value = True
        self.mock_app.active_recordings.get_existing_takes.return_value = []

        self.controller.delete_current_recording()

        # Should set displayed take to 0
        self.mock_app.state.recording.set_displayed_take.assert_called_with(
            "test_label", 0
        )

    @patch("revoxx.controllers.file_operations_controller.messagebox.askyesno")
    def test_delete_current_recording_failure(self, mock_askyesno):
        """Test failed deletion."""
        mock_askyesno.return_value = True
        self.mock_app.file_manager.move_to_trash.return_value = False

        self.controller.delete_current_recording()

        self.mock_app.window.set_status.assert_called_with(
            "Failed to delete recording", MsgType.ERROR
        )

    def test_restore_deleted_recording_success(self):
        """Test successful restoration of deleted recording."""
        self.controller.restore_deleted_recording()

        # Verify restoration of highest deleted take
        self.mock_app.file_manager.restore_from_trash.assert_called_once_with(
            "test_label", 3
        )

        # Verify cache invalidation
        self.mock_app.active_recordings.on_recording_restored.assert_called_once_with(
            "test_label"
        )

        # Verify UI updates
        self.mock_app.state.recording.set_displayed_take.assert_called_with(
            "test_label", 3
        )
        self.mock_app.display_controller.show_saved_recording.assert_called_once()
        self.mock_app.navigation_controller.update_take_status.assert_called_once()

        # Verify success message
        self.mock_app.window.set_status.assert_called_with(
            "Recording restored: take_003", MsgType.TEMPORARY
        )

    def test_restore_deleted_recording_no_label(self):
        """Test restoring when no current label."""
        self.mock_app.state.recording.current_label = None

        self.controller.restore_deleted_recording()

        # Should return early
        self.mock_app.file_manager.get_deleted_takes.assert_not_called()

    def test_restore_deleted_recording_none_deleted(self):
        """Test restoring when no deleted recordings."""
        self.mock_app.file_manager.get_deleted_takes.return_value = []

        self.controller.restore_deleted_recording()

        self.mock_app.window.set_status.assert_called_once_with(
            "No deleted recordings to restore", MsgType.TEMPORARY
        )
        self.mock_app.file_manager.restore_from_trash.assert_not_called()

    def test_restore_deleted_recording_failure(self):
        """Test failed restoration."""
        self.mock_app.file_manager.restore_from_trash.return_value = False

        self.controller.restore_deleted_recording()

        self.mock_app.window.set_status.assert_called_with(
            "Failed to restore recording", MsgType.ERROR
        )

    @patch("shutil.copy2")
    def test_export_session_success(self, mock_copy2):
        """Test successful session export."""
        export_path = Mock(spec=Path)
        export_path.mkdir = Mock()

        # Create mocks for path operations
        script_dest = Mock(spec=Path)
        recordings_dir = Mock(spec=Path)
        recordings_dir.mkdir = Mock()

        # Setup path division to return appropriate mocks
        export_path.__truediv__ = Mock(
            side_effect=lambda x: script_dest if x == "script.txt" else recordings_dir
        )
        recordings_dir.__truediv__ = Mock(return_value=Mock(spec=Path))

        with patch("builtins.open", create=True) as mock_open:
            self.controller.export_session(export_path)

        # Verify directory creation
        export_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify script export
        mock_open.assert_called()

        # Verify success message
        self.mock_app.window.set_status.assert_called()
        message = self.mock_app.window.set_status.call_args[0][0]
        self.assertIn("Session exported", message)

    def test_export_session_error(self):
        """Test session export with error."""
        export_path = Mock(spec=Path)
        export_path.mkdir.side_effect = OSError("Permission denied")

        self.controller.export_session(export_path)

        self.mock_app.window.set_status.assert_called()
        message = self.mock_app.window.set_status.call_args[0][0]
        self.assertIn("Export failed", message)

    @patch("shutil.copy2")
    def test_export_current_recording_success(self, mock_copy2):
        """Test successful export of current recording."""
        export_path = Mock(spec=Path)
        export_path.name = "exported.wav"

        self.controller.export_current_recording(export_path)

        # Verify file copy
        mock_copy2.assert_called_once()

        # Verify success message
        self.mock_app.window.set_status.assert_called_with(
            "Recording exported to exported.wav", MsgType.TEMPORARY
        )

    def test_export_current_recording_no_label(self):
        """Test exporting when no current label."""
        self.mock_app.state.recording.current_label = None
        export_path = Mock(spec=Path)

        self.controller.export_current_recording(export_path)

        self.mock_app.window.set_status.assert_called_once_with(
            "No recording to export", MsgType.TEMPORARY
        )

    def test_export_current_recording_no_take(self):
        """Test exporting when no recording exists."""
        self.mock_app.state.recording.get_current_take.return_value = 0
        export_path = Mock(spec=Path)

        self.controller.export_current_recording(export_path)

        self.mock_app.window.set_status.assert_called_once_with(
            "No recording to export", MsgType.TEMPORARY
        )

    def test_check_recording_exists_true(self):
        """Test checking if recording exists - exists."""
        result = self.controller.check_recording_exists("test_label", 1)

        self.assertTrue(result)
        self.mock_app.file_manager.get_recording_path.assert_called_once_with(
            "test_label", 1
        )

    def test_check_recording_exists_false_zero_take(self):
        """Test checking if recording exists - zero take."""
        result = self.controller.check_recording_exists("test_label", 0)

        self.assertFalse(result)
        self.mock_app.file_manager.get_recording_path.assert_not_called()

    def test_check_recording_exists_false_not_found(self):
        """Test checking if recording exists - file not found."""
        mock_path = Mock()
        mock_path.exists = Mock(return_value=False)
        self.mock_app.file_manager.get_recording_path.return_value = mock_path

        result = self.controller.check_recording_exists("test_label", 1)

        self.assertFalse(result)

    @patch("soundfile.SoundFile")
    def test_get_recording_info_success(self, mock_soundfile):
        """Test getting recording info successfully."""
        # Setup soundfile mock
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.samplerate = 48000
        mock_file.channels = 1
        mock_file.__len__.return_value = 48000
        mock_soundfile.return_value = mock_file

        result = self.controller.get_recording_info("test_label", 1)

        self.assertIsNotNone(result)
        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["file_size"], 1024)

    def test_get_recording_info_not_exists(self):
        """Test getting info for non-existent recording."""
        with patch.object(
            self.controller, "check_recording_exists", return_value=False
        ):
            result = self.controller.get_recording_info("test_label", 1)

        self.assertIsNone(result)

    def test_load_audio_data_success(self):
        """Test loading audio data successfully."""
        result = self.controller.load_audio_data("test_label", 1)

        self.assertIsNotNone(result)
        audio_data, sample_rate = result
        self.assertEqual(sample_rate, 48000)
        np.testing.assert_array_equal(audio_data, np.array([0.1, 0.2, 0.3]))

    def test_load_audio_data_not_exists(self):
        """Test loading audio data for non-existent recording."""
        with patch.object(
            self.controller, "check_recording_exists", return_value=False
        ):
            result = self.controller.load_audio_data("test_label", 1)

        self.assertIsNone(result)

    def test_get_total_recording_count(self):
        """Test getting total recording count."""
        count = self.controller.get_total_recording_count()

        self.assertEqual(count, 2)  # test_label and other_label

    def test_get_session_size(self):
        """Test getting total session size."""
        size = self.controller.get_session_size()

        self.assertEqual(size, 2048)  # 2 files * 1024 bytes each

    def test_cleanup_empty_directories(self):
        """Test cleaning up empty directories."""
        self.controller.cleanup_empty_directories()

        self.mock_app.file_manager.cleanup_empty_directories.assert_called_once_with(
            Path("/test/recordings")
        )


if __name__ == "__main__":
    unittest.main()
