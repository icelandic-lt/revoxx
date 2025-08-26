"""Tests for the ProcessManager."""

import unittest
from unittest.mock import Mock, patch
import queue

from revoxx.controllers.process_manager import ProcessManager


class TestProcessManager(unittest.TestCase):
    """Test cases for ProcessManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with minimal required attributes
        self.mock_app = Mock()
        self.mock_app.config.audio = Mock()
        self.mock_app.shared_state.name = "test_shared_state"
        self.mock_app.window.ui_state.spectrogram_visible = False

        # Create controller with mocked initialization
        with patch.object(ProcessManager, "_initialize_resources"):
            self.controller = ProcessManager(self.mock_app)

        self.controller.manager = Mock()
        self.controller.shutdown_event = Mock()
        self.controller.manager_dict = {"save_path": None}
        self.controller.audio_queue = Mock()
        self.controller.record_queue = Mock()
        self.controller.playback_queue = Mock()
        self.controller.queue_manager = Mock()
        self.controller.queue_manager.get_audio_data = Mock(side_effect=queue.Empty)

    @patch("revoxx.controllers.process_manager.AudioQueueManager")
    @patch("revoxx.controllers.process_manager.mp.Manager")
    @patch("revoxx.controllers.process_manager.mp.Event")
    def test_initialize_resources(
        self, mock_event_class, mock_manager_class, mock_queue_manager_class
    ):
        """Test initializing multiprocessing resources."""
        # Setup mocks
        mock_manager = Mock()
        mock_dict = {}  # Use real dict instead of Mock
        mock_manager.dict = Mock(return_value=mock_dict)
        mock_manager_class.return_value = mock_manager

        mock_event = Mock()
        mock_event_class.return_value = mock_event

        # Mock AudioQueueManager
        mock_queue_manager = Mock()
        mock_audio_queue = Mock()
        mock_record_queue = Mock()
        mock_playback_queue = Mock()
        mock_queue_manager.audio_queue = mock_audio_queue
        mock_queue_manager.record_queue = mock_record_queue
        mock_queue_manager.playback_queue = mock_playback_queue
        mock_queue_manager_class.return_value = mock_queue_manager

        # Create new controller to test initialization
        controller = ProcessManager(self.mock_app)

        # Verify manager creation
        mock_manager_class.assert_called_once()
        mock_event_class.assert_called_once()
        mock_queue_manager_class.assert_called_once()

        # Verify controller has correct references
        self.assertEqual(controller.manager, mock_manager)
        self.assertEqual(controller.shutdown_event, mock_event)
        self.assertEqual(controller.manager_dict, mock_dict)
        self.assertEqual(controller.queue_manager, mock_queue_manager)
        self.assertEqual(controller.audio_queue, mock_audio_queue)
        self.assertEqual(controller.record_queue, mock_record_queue)
        self.assertEqual(controller.playback_queue, mock_playback_queue)

        # Verify app references set
        self.assertEqual(self.mock_app.shutdown_event, mock_event)
        self.assertEqual(self.mock_app.manager_dict, mock_dict)
        self.assertEqual(self.mock_app.queue_manager, mock_queue_manager)
        # Direct queue references are no longer set in app
        self.assertIsNotNone(self.mock_app.queue_manager)

    @patch("revoxx.controllers.process_manager.mp.Process")
    def test_start_processes(self, mock_process_class):
        """Test starting background processes."""
        # Setup
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        # Execute
        self.controller.start_processes()

        # Verify two processes created
        self.assertEqual(mock_process_class.call_count, 2)

        # Verify processes started
        self.assertEqual(mock_process.start.call_count, 2)

        # Verify process references stored
        self.assertIsNotNone(self.controller.record_process)
        self.assertIsNotNone(self.controller.playback_process)

    @patch("revoxx.controllers.process_manager.threading.Thread")
    def test_start_audio_queue_processing(self, mock_thread_class):
        """Test starting audio queue processing."""
        # Setup
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        # Execute
        self.controller.start_audio_queue_processing()

        # Verify state updated
        self.assertTrue(self.controller.is_audio_queue_active())

        # Verify thread created and started
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()
        self.assertTrue(mock_thread.daemon)

        # Verify thread reference stored
        self.assertEqual(self.controller.transfer_thread, mock_thread)

    def test_stop_audio_queue_processing(self):
        """Test stopping audio queue processing."""
        # Setup
        mock_thread = Mock()
        mock_thread.is_alive = Mock(return_value=True)
        self.controller.transfer_thread = mock_thread

        # Execute
        self.controller.stop_audio_queue_processing()

        # Verify state updated
        self.assertFalse(self.controller.is_audio_queue_active())

        # Verify thread join called
        mock_thread.join.assert_called_once_with(timeout=0.2)

    def test_stop_audio_queue_processing_no_thread(self):
        """Test stopping audio queue processing when no thread."""
        # Execute
        self.controller.stop_audio_queue_processing()

        # Verify state updated
        self.assertFalse(self.controller.is_audio_queue_active())

    def test_set_audio_queue_active(self):
        """Test setting audio queue active state."""
        # Test setting to True
        self.controller.set_audio_queue_active(True)
        self.assertTrue(self.controller.is_audio_queue_active())

        # Test setting to False
        self.controller.set_audio_queue_active(False)
        self.assertFalse(self.controller.is_audio_queue_active())

    def test_set_save_path(self):
        """Test setting save path."""
        # Test with path
        self.controller.set_save_path("/test/path.wav")
        self.assertEqual(self.controller.get_save_path(), "/test/path.wav")

        # Test with None
        self.controller.set_save_path(None)
        self.assertIsNone(self.controller.get_save_path())

    def test_shutdown_process_not_responding(self):
        """Test shutdown when process doesn't respond to terminate."""
        # Ensure shutdown_event is set up
        self.controller.shutdown_event = Mock()

        # Setup process that stays alive after terminate
        mock_process = Mock()
        mock_process.is_alive = Mock(side_effect=[True, True, False])
        self.controller.record_process = mock_process

        # Execute
        self.controller.shutdown()

        # Verify kill was called
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_shutdown_with_broken_pipe(self):
        """Test shutdown with broken pipe errors."""
        # Ensure shutdown_event is set up
        self.controller.shutdown_event = Mock()

        # Setup manager that raises BrokenPipeError
        self.controller.manager.shutdown.side_effect = BrokenPipeError

        # Should not raise exception
        self.controller.shutdown()

    def test_is_audio_queue_active_true(self):
        """Test checking if audio queue is active - true."""
        self.controller.set_audio_queue_active(True)

        result = self.controller.is_audio_queue_active()

        self.assertTrue(result)

    def test_is_audio_queue_active_false(self):
        """Test checking if audio queue is active - false."""
        self.controller.set_audio_queue_active(False)

        result = self.controller.is_audio_queue_active()

        self.assertFalse(result)

    def test_is_audio_queue_active_no_dict(self):
        """Test checking audio queue when no dict."""
        self.controller.manager_dict = None

        result = self.controller.is_audio_queue_active()

        self.assertFalse(result)

    def test_are_processes_running_true(self):
        """Test checking if processes are running - true."""
        mock_record = Mock()
        mock_record.is_alive = Mock(return_value=True)
        self.controller.record_process = mock_record

        mock_playback = Mock()
        mock_playback.is_alive = Mock(return_value=True)
        self.controller.playback_process = mock_playback

        result = self.controller.are_processes_running()

        self.assertTrue(result)

    def test_are_processes_running_false(self):
        """Test checking if processes are running - false."""
        mock_record = Mock()
        mock_record.is_alive = Mock(return_value=False)
        self.controller.record_process = mock_record

        mock_playback = Mock()
        mock_playback.is_alive = Mock(return_value=True)
        self.controller.playback_process = mock_playback

        result = self.controller.are_processes_running()

        self.assertFalse(result)

    def test_are_processes_running_none(self):
        """Test checking processes when None."""
        self.controller.record_process = None
        self.controller.playback_process = None

        result = self.controller.are_processes_running()

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
