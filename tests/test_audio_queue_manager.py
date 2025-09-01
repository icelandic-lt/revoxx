"""Tests for the AudioQueueManager."""

import queue
import unittest
from unittest.mock import Mock

from revoxx.audio.queue_manager import AudioQueueManager


class TestAudioQueueManager(unittest.TestCase):
    """Test cases for AudioQueueManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock queues
        self.mock_record_queue = Mock()
        self.mock_playback_queue = Mock()
        self.mock_audio_queue = Mock()

        # Create queue manager
        self.queue_manager = AudioQueueManager(
            self.mock_record_queue, self.mock_playback_queue, self.mock_audio_queue
        )

    def test_initialization(self):
        """Test queue manager initialization."""
        self.assertEqual(self.queue_manager._record_queue, self.mock_record_queue)
        self.assertEqual(self.queue_manager._playback_queue, self.mock_playback_queue)
        self.assertEqual(self.queue_manager._audio_queue, self.mock_audio_queue)

    # ========== Playback Control Tests ==========

    def test_start_playback(self):
        """Test starting playback."""
        buffer_metadata = {"name": "test", "shape": (100,), "dtype": "float32"}
        sample_rate = 48000

        self.queue_manager.start_playback(buffer_metadata, sample_rate)

        self.mock_playback_queue.put.assert_called_once_with(
            {
                "action": "play",
                "buffer_metadata": buffer_metadata,
                "sample_rate": sample_rate,
            },
            block=False,
        )

    def test_stop_playback(self):
        """Test stopping playback."""
        self.queue_manager.stop_playback()

        self.mock_playback_queue.put.assert_called_once_with({"action": "stop"})

    def test_set_output_device_success(self):
        """Test setting output device successfully."""
        self.mock_playback_queue.put = Mock()

        result = self.queue_manager.set_output_device(1)

        self.assertTrue(result)
        self.mock_playback_queue.put.assert_called_once_with(
            {"action": "set_output_device", "device_name": 1}, block=False
        )

    def test_set_output_device_queue_full(self):
        """Test setting output device when queue is full."""
        self.mock_playback_queue.put = Mock(side_effect=queue.Full)

        result = self.queue_manager.set_output_device(1)

        self.assertFalse(result)

    def test_set_output_device_none(self):
        """Test setting output device to None (system default)."""
        result = self.queue_manager.set_output_device(None)

        self.assertTrue(result)
        self.mock_playback_queue.put.assert_called_once_with(
            {"action": "set_output_device", "device_name": None}, block=False
        )

    def test_set_output_channel_mapping_success(self):
        """Test setting output channel mapping."""
        mapping = [0, 1]

        result = self.queue_manager.set_output_channel_mapping(mapping)

        self.assertTrue(result)
        self.mock_playback_queue.put.assert_called_once_with(
            {"action": "set_output_channel_mapping", "mapping": mapping}, block=False
        )

    def test_set_output_channel_mapping_none(self):
        """Test setting output channel mapping to None."""
        result = self.queue_manager.set_output_channel_mapping(None)

        self.assertTrue(result)
        self.mock_playback_queue.put.assert_called_once_with(
            {"action": "set_output_channel_mapping", "mapping": None}, block=False
        )

    def test_quit_playback_process(self):
        """Test sending quit command to playback process."""
        result = self.queue_manager.quit_playback_process()

        self.assertTrue(result)
        self.mock_playback_queue.put.assert_called_once_with(
            {"action": "quit"}, block=False
        )

    # ========== Recording Control Tests ==========

    def test_start_recording(self):
        """Test starting recording."""
        self.queue_manager.start_recording()

        self.mock_record_queue.put.assert_called_once_with(
            {"action": "start"}, block=False
        )

    def test_stop_recording(self):
        """Test stopping recording."""
        self.queue_manager.stop_recording()

        self.mock_record_queue.put.assert_called_once_with({"action": "stop"})

    def test_set_input_device_success(self):
        """Test setting input device successfully."""
        result = self.queue_manager.set_input_device(2)

        self.assertTrue(result)
        self.mock_record_queue.put.assert_called_once_with(
            {"action": "set_input_device", "device_name": 2}, block=False
        )

    def test_set_input_device_queue_full(self):
        """Test setting input device when queue is full."""
        self.mock_record_queue.put = Mock(side_effect=queue.Full)

        result = self.queue_manager.set_input_device(2)

        self.assertFalse(result)

    def test_set_input_channel_mapping_success(self):
        """Test setting input channel mapping."""
        mapping = [1, 0]

        result = self.queue_manager.set_input_channel_mapping(mapping)

        self.assertTrue(result)
        self.mock_record_queue.put.assert_called_once_with(
            {"action": "set_input_channel_mapping", "mapping": mapping}, block=False
        )

    def test_quit_record_process(self):
        """Test sending quit command to record process."""
        result = self.queue_manager.quit_record_process()

        self.assertTrue(result)
        self.mock_record_queue.put.assert_called_once_with(
            {"action": "quit"}, block=False
        )

    # ========== Audio Visualization Queue Tests ==========

    def test_get_audio_data_success(self):
        """Test getting audio data from queue."""
        test_data = {"type": "audio", "data": [0.1, 0.2, 0.3]}
        self.mock_audio_queue.get = Mock(return_value=test_data)

        result = self.queue_manager.get_audio_data(timeout=0.5)

        self.assertEqual(result, test_data)
        self.mock_audio_queue.get.assert_called_once_with(timeout=0.5)

    def test_get_audio_data_empty(self):
        """Test getting audio data when queue is empty."""
        self.mock_audio_queue.get = Mock(side_effect=queue.Empty)

        with self.assertRaises(queue.Empty):
            self.queue_manager.get_audio_data(timeout=0.1)

    def test_audio_queue_property(self):
        """Test audio_queue property returns the queue."""
        result = self.queue_manager.audio_queue

        self.assertEqual(result, self.mock_audio_queue)


if __name__ == "__main__":
    unittest.main()
