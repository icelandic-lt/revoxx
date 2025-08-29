"""Tests for the DialogController."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from revoxx.controllers.dialog_controller import DialogController
from revoxx.ui.dialogs.utterance_list_base import SortDirection


class TestDialogController(unittest.TestCase):
    """Test cases for DialogController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()

        # Mock window (WindowBase architecture)
        self.mock_app.window = Mock()
        self.mock_app.window.window = Mock()  # WindowBase uses window.window now

        # Mock config
        self.mock_app.config = Mock()
        self.mock_app.config.paths = Mock()
        self.mock_app.config.paths.sessions_base_dir = Path("/test/sessions")
        self.mock_app.config.audio = Mock()
        self.mock_app.config.audio.sample_rate = 48000
        self.mock_app.config.audio.bit_depth = 24
        self.mock_app.config.audio.channels = 1
        self.mock_app.config.audio.input_device = None
        self.mock_app.config.audio.output_device = None

        # Mock state
        self.mock_app.state = Mock()
        self.mock_app.state.recording = Mock()
        self.mock_app.state.recording.is_recording = False
        self.mock_app.state.recording.utterances = [
            {"id": "001", "text": "Test 1"},
            {"id": "002", "text": "Test 2"},
        ]
        self.mock_app.state.recording.current_index = 0

        # Mock controllers
        self.mock_app.navigation_controller = Mock()
        self.mock_app.navigation_controller.get_display_position = Mock(return_value=1)
        self.mock_app.display_controller = Mock()
        self.mock_app.device_controller = Mock()

        # Mock active recordings
        self.mock_app.active_recordings = Mock()
        self.mock_app.active_recordings.get_sorted_indices = Mock(return_value=[0, 1])
        self.mock_app.active_recordings.sort_column = "default"
        self.mock_app.active_recordings.sort_reverse = False

        # Mock additional required attributes
        self.mock_app.state.recording.labels = ["label1", "label2"]
        self.mock_app.file_manager = Mock()
        self.mock_app.session_manager = Mock()
        self.mock_app.session_manager.get_default_base_dir = Mock(
            return_value=Path("/test/sessions")
        )

        self.controller = DialogController(self.mock_app)

    @patch("revoxx.controllers.dialog_controller.OpenSessionDialog")
    def test_show_open_session_dialog_selected(self, mock_dialog_class):
        """Test showing open session dialog with selection."""
        mock_dialog = Mock()
        mock_dialog.show.return_value = Path("/test/selected/session")
        mock_dialog_class.return_value = mock_dialog

        result = self.controller.show_open_session_dialog()

        self.assertEqual(result, Path("/test/selected/session"))
        mock_dialog_class.assert_called_once_with(
            self.mock_app.window.window,
            Path("/test/sessions"),
        )
        mock_dialog.show.assert_called_once()

    @patch("revoxx.controllers.dialog_controller.OpenSessionDialog")
    def test_show_open_session_dialog_cancelled(self, mock_dialog_class):
        """Test showing open session dialog when cancelled."""
        mock_dialog = Mock()
        mock_dialog.show.return_value = None
        mock_dialog_class.return_value = mock_dialog

        result = self.controller.show_open_session_dialog()

        self.assertIsNone(result)

    @patch("revoxx.controllers.dialog_controller.filedialog.askdirectory")
    def test_show_export_dialog_selected(self, mock_askdir):
        """Test showing export dialog with selection."""
        mock_askdir.return_value = "/test/export/dir"

        result = self.controller.show_export_dialog()

        self.assertEqual(result, Path("/test/export/dir"))
        mock_askdir.assert_called_once_with(
            title="Select Export Directory", parent=self.mock_app.window.window
        )

    @patch("revoxx.controllers.dialog_controller.filedialog.askdirectory")
    def test_show_export_dialog_cancelled(self, mock_askdir):
        """Test showing export dialog when cancelled."""
        mock_askdir.return_value = ""

        result = self.controller.show_export_dialog()

        self.assertIsNone(result)

    @patch("revoxx.controllers.dialog_controller.filedialog.asksaveasfilename")
    def test_show_export_file_dialog_selected(self, mock_asksave):
        """Test showing export file dialog with selection."""
        mock_asksave.return_value = "/test/export/file.wav"

        result = self.controller.show_export_file_dialog()

        self.assertEqual(result, Path("/test/export/file.wav"))
        mock_asksave.assert_called_once_with(
            title="Export Recording",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            parent=self.mock_app.window.window,
        )

    @patch("revoxx.controllers.dialog_controller.filedialog.asksaveasfilename")
    def test_show_export_file_dialog_cancelled(self, mock_asksave):
        """Test showing export file dialog when cancelled."""
        mock_asksave.return_value = ""

        result = self.controller.show_export_file_dialog()

        self.assertIsNone(result)

    @patch("revoxx.controllers.dialog_controller.FindDialog")
    def test_show_find_dialog_new(self, mock_find_dialog_class):
        """Test showing find dialog for the first time."""
        mock_dialog = Mock()
        mock_find_dialog_class.return_value = mock_dialog

        self.controller.show_find_dialog()

        mock_find_dialog_class.assert_called_once_with(
            self.mock_app.window.window,
            self.mock_app.state.recording.utterances,
            self.mock_app.state.recording.labels,
            self.mock_app.file_manager,
            self.mock_app.state.recording.current_index,
            self.controller._on_find_select,
            self.mock_app.active_recordings.get_sorted_indices(),
            self.mock_app.active_recordings.sort_column,
            SortDirection.UP,  # Changed from sort_reverse (False) to SortDirection.UP
        )
        self.assertEqual(self.controller.find_dialog, mock_dialog)

    @patch("revoxx.controllers.dialog_controller.FindDialog")
    def test_show_find_dialog_existing(self, mock_find_dialog_class):
        """Test showing find dialog when already exists."""
        mock_dialog = Mock()
        mock_dialog.dialog = Mock()
        mock_dialog.dialog.winfo_exists = Mock(return_value=True)
        mock_dialog.dialog.lift = Mock()
        self.controller.find_dialog = mock_dialog

        self.controller.show_find_dialog()

        mock_dialog.dialog.lift.assert_called_once()
        mock_find_dialog_class.assert_not_called()

    def test_on_find_select(self):
        """Test handling find dialog selection."""
        mock_dialog = Mock()
        mock_dialog.dialog = Mock()
        mock_dialog.dialog.destroy = Mock()
        self.controller.find_dialog = mock_dialog

        self.controller._on_find_select(5)

        self.mock_app.navigation_controller.find_utterance.assert_called_once_with(5)
        mock_dialog.dialog.destroy.assert_called_once()
        self.assertIsNone(self.controller.find_dialog)

    @patch("revoxx.controllers.dialog_controller.UtteranceOrderDialog")
    def test_show_utterance_order_dialog_new(self, mock_order_dialog_class):
        """Test showing utterance order dialog for the first time."""
        mock_dialog = Mock()
        # Mock show() to return a tuple for sort settings
        mock_dialog.show = Mock(return_value=("default", False))
        mock_order_dialog_class.return_value = mock_dialog
        self.mock_app.current_session = Mock()
        self.mock_app.active_recordings.set_sort = Mock()
        self.mock_app.display_controller.update_display = Mock()

        self.controller.show_utterance_order_dialog()

        mock_order_dialog_class.assert_called_once_with(
            self.mock_app.window.window,
            self.mock_app.state.recording.utterances,
            self.mock_app.state.recording.labels,
            self.mock_app.file_manager,
            self.mock_app.active_recordings.get_sorted_indices(),
            self.mock_app.active_recordings.sort_column,
            SortDirection.UP,  # Changed from sort_reverse (False) to SortDirection.UP
            self.mock_app.state.recording.current_index,
        )
        # Verify that set_sort and update were called with the dialog result
        self.mock_app.active_recordings.set_sort.assert_called_once_with(
            "default", False
        )
        self.mock_app.display_controller.update_display.assert_called_once()

    @patch("revoxx.controllers.dialog_controller.get_device_manager")
    @patch("revoxx.controllers.dialog_controller.messagebox.showerror")
    def test_show_device_selection_dialog_no_devices(self, mock_showerror, mock_get_dm):
        """Test showing device selection dialog when no devices found."""
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(return_value=[])
        mock_get_dm.return_value = mock_dm

        self.controller.show_device_selection_dialog("input")

        mock_showerror.assert_called_once_with(
            "Error", "No input devices found", parent=self.mock_app.window.window
        )

    @patch("revoxx.controllers.dialog_controller.messagebox.askyesno")
    def test_confirm_quit_while_recording(self, mock_askyesno):
        """Test confirming quit while recording."""
        self.mock_app.state.recording.is_recording = True
        mock_askyesno.return_value = True

        result = self.controller.confirm_quit()

        self.assertTrue(result)
        mock_askyesno.assert_called_once_with(
            "Recording in Progress",
            "Recording is in progress. Do you want to stop and quit?",
            parent=self.mock_app.window.window,
        )

    def test_confirm_quit_not_recording(self):
        """Test confirming quit when not recording."""
        self.mock_app.state.recording.is_recording = False

        result = self.controller.confirm_quit()

        self.assertTrue(result)

    @patch("revoxx.controllers.dialog_controller.messagebox.showerror")
    def test_show_error(self, mock_showerror):
        """Test showing error dialog."""
        self.controller.show_error("Test Error", "Error message")

        mock_showerror.assert_called_once_with(
            "Test Error", "Error message", parent=self.mock_app.window.window
        )

    @patch("revoxx.controllers.dialog_controller.messagebox.showinfo")
    def test_show_info(self, mock_showinfo):
        """Test showing info dialog."""
        self.controller.show_info("Test Info", "Info message")

        mock_showinfo.assert_called_once_with(
            "Test Info", "Info message", parent=self.mock_app.window.window
        )

    @patch("revoxx.controllers.dialog_controller.messagebox.showwarning")
    def test_show_warning(self, mock_showwarning):
        """Test showing warning dialog."""
        self.controller.show_warning("Test Warning", "Warning message")

        mock_showwarning.assert_called_once_with(
            "Test Warning", "Warning message", parent=self.mock_app.window.window
        )

    def test_cleanup_with_dialogs(self):
        """Test cleanup with open dialogs."""
        # Create mock dialogs - find and order use .dialog, settings and device don't
        mock_find = Mock()
        mock_find.dialog = Mock()
        mock_find.dialog.winfo_exists = Mock(return_value=True)
        mock_find.dialog.destroy = Mock()
        self.controller.find_dialog = mock_find

        mock_order = Mock()
        mock_order.dialog = Mock()
        mock_order.dialog.winfo_exists = Mock(return_value=True)
        mock_order.dialog.destroy = Mock()
        self.controller.order_dialog = mock_order

        # settings_dialog and device_dialog don't use .dialog structure
        mock_settings = Mock()
        mock_settings.winfo_exists = Mock(return_value=True)
        mock_settings.destroy = Mock()
        self.controller.settings_dialog = mock_settings

        mock_device = Mock()
        mock_device.winfo_exists = Mock(return_value=True)
        mock_device.destroy = Mock()
        self.controller.device_dialog = mock_device

        self.controller.cleanup()

        mock_find.dialog.destroy.assert_called_once()
        mock_order.dialog.destroy.assert_called_once()
        mock_settings.destroy.assert_called_once()
        mock_device.destroy.assert_called_once()

    def test_cleanup_without_dialogs(self):
        """Test cleanup without open dialogs."""
        # Should not raise any exceptions
        self.controller.cleanup()


if __name__ == "__main__":
    unittest.main()
