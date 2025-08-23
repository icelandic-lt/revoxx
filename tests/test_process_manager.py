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
        self.controller.manager_dict = {"audio_queue_active": False, "save_path": None}
        self.controller.audio_queue = Mock()
        self.controller.record_queue = Mock()
        self.controller.playback_queue = Mock()

    @patch("revoxx.controllers.process_manager.mp.Manager")
    @patch("revoxx.controllers.process_manager.mp.Event")
    @patch("revoxx.controllers.process_manager.mp.Queue")
    def test_initialize_resources(
        self, mock_queue_class, mock_event_class, mock_manager_class
    ):
        """Test initializing multiprocessing resources."""
        # Setup mocks
        mock_manager = Mock()
        mock_dict = {}  # Use real dict instead of Mock
        mock_manager.dict = Mock(return_value=mock_dict)
        mock_manager_class.return_value = mock_manager

        mock_event = Mock()
        mock_event_class.return_value = mock_event

        mock_queues = [Mock(), Mock(), Mock()]
        mock_queue_class.side_effect = mock_queues

        # Create new controller to test initialization
        controller = ProcessManager(self.mock_app)

        # Verify manager creation
        mock_manager_class.assert_called_once()
        mock_event_class.assert_called_once()

        # Verify queue creation (3 queues)
        self.assertEqual(mock_queue_class.call_count, 3)

        # Verify controller has correct references
        self.assertEqual(controller.manager, mock_manager)
        self.assertEqual(controller.shutdown_event, mock_event)
        self.assertEqual(controller.manager_dict, mock_dict)
        self.assertEqual(controller.audio_queue, mock_queues[0])
        self.assertEqual(controller.record_queue, mock_queues[1])
        self.assertEqual(controller.playback_queue, mock_queues[2])

        # Verify app references set
        self.assertEqual(self.mock_app.shutdown_event, mock_event)
        self.assertEqual(self.mock_app.manager_dict, mock_dict)
        self.assertEqual(self.mock_app.audio_queue, mock_queues[0])
        self.assertEqual(self.mock_app.record_queue, mock_queues[1])
        self.assertEqual(self.mock_app.playback_queue, mock_queues[2])

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
        self.assertTrue(self.controller.manager_dict["audio_queue_active"])

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
        self.assertFalse(self.controller.manager_dict["audio_queue_active"])

        # Verify thread join called
        mock_thread.join.assert_called_once_with(timeout=1.0)

    def test_stop_audio_queue_processing_no_thread(self):
        """Test stopping audio queue processing when no thread."""
        # Execute
        self.controller.stop_audio_queue_processing()

        # Verify state updated
        self.assertFalse(self.controller.manager_dict["audio_queue_active"])

    def test_update_audio_queue_state(self):
        """Test updating audio queue state."""
        # Test setting to True
        self.controller.update_audio_queue_state(True)
        self.assertTrue(self.controller.manager_dict["audio_queue_active"])

        # Test setting to False
        self.controller.update_audio_queue_state(False)
        self.assertFalse(self.controller.manager_dict["audio_queue_active"])

    def test_set_save_path(self):
        """Test setting save path."""
        # Test with path
        self.controller.set_save_path("/test/path.wav")
        self.assertEqual(self.controller.manager_dict["save_path"], "/test/path.wav")

        # Test with None
        self.controller.set_save_path(None)
        self.assertIsNone(self.controller.manager_dict["save_path"])

    def test_send_record_command(self):
        """Test sending command to record process."""
        command = {"action": "start", "path": "/test.wav"}

        self.controller.send_record_command(command)

        self.controller.record_queue.put.assert_called_once_with(command, block=False)

    def test_send_record_command_queue_full(self):
        """Test sending command when queue is full."""
        self.controller.record_queue.put.side_effect = queue.Full
        command = {"action": "start"}

        # Should not raise exception
        self.controller.send_record_command(command)

    def test_send_playback_command(self):
        """Test sending command to playback process."""
        command = {"action": "play", "path": "/test.wav"}

        self.controller.send_playback_command(command)

        self.controller.playback_queue.put.assert_called_once_with(command, block=False)

    def test_send_playback_command_queue_full(self):
        """Test sending playback command when queue is full."""
        self.controller.playback_queue.put.side_effect = queue.Full
        command = {"action": "play"}

        # Should not raise exception
        self.controller.send_playback_command(command)

    def test_clear_audio_queue(self):
        """Test clearing audio queue."""
        # Setup queue with items
        self.controller.audio_queue.get_nowait.side_effect = [
            "item1",
            "item2",
            "item3",
            queue.Empty,
        ]

        self.controller.clear_audio_queue()

        # Verify all items were removed
        self.assertEqual(self.controller.audio_queue.get_nowait.call_count, 4)

    def test_clear_audio_queue_already_empty(self):
        """Test clearing already empty queue."""
        self.controller.audio_queue.get_nowait.side_effect = queue.Empty

        # Should not raise exception
        self.controller.clear_audio_queue()

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
        self.controller.manager_dict["audio_queue_active"] = True

        result = self.controller.is_audio_queue_active()

        self.assertTrue(result)

    def test_is_audio_queue_active_false(self):
        """Test checking if audio queue is active - false."""
        self.controller.manager_dict["audio_queue_active"] = False

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
