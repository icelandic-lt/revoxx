"""Integration tests for IPC communication between processes.

These tests verify that the multiprocessing queue communication
works correctly without mocking.
"""

import unittest
import multiprocessing as mp
import numpy as np

from revoxx.audio.queue_manager import AudioQueueManager


def child_process_func(record_queue, playback_queue, result_queue):
    """Helper function for cross-process communication test."""
    # Create manager in child process mode
    child_manager = AudioQueueManager(
        record_queue=record_queue, playback_queue=playback_queue, audio_queue=None
    )

    # Receive command
    command = child_manager.get_record_command(timeout=1.0)
    if command:
        result_queue.put(command["action"])

    # Send response
    child_manager._playback_queue.put({"action": "response"})


class TestIPCCommunication(unittest.TestCase):
    """Test inter-process communication using real multiprocessing queues."""

    def test_queue_manager_main_process_creates_queues(self):
        """Test that AudioQueueManager creates queues in main process mode."""
        manager = AudioQueueManager()

        # Verify queues are created
        self.assertIsNotNone(manager.record_queue)
        self.assertIsNotNone(manager.playback_queue)
        self.assertIsNotNone(manager.audio_queue)

        # Verify they are actual Queue objects
        self.assertIsInstance(manager.record_queue, mp.queues.Queue)
        self.assertIsInstance(manager.playback_queue, mp.queues.Queue)
        self.assertIsInstance(manager.audio_queue, mp.queues.Queue)

    def test_queue_manager_child_process_uses_existing_queues(self):
        """Test that AudioQueueManager uses existing queues in child process mode."""
        # Create queues externally
        record_queue = mp.Queue()
        playback_queue = mp.Queue()
        audio_queue = mp.Queue()

        # Create manager in child mode
        manager = AudioQueueManager(
            record_queue=record_queue,
            playback_queue=playback_queue,
            audio_queue=audio_queue,
        )

        # Verify it uses the provided queues
        self.assertIs(manager.record_queue, record_queue)
        self.assertIs(manager.playback_queue, playback_queue)
        self.assertIs(manager.audio_queue, audio_queue)

    def test_record_command_communication(self):
        """Test sending and receiving record commands through real queues."""
        manager = AudioQueueManager()

        # Send command
        manager.start_recording()

        # Receive command
        command = manager.get_record_command(timeout=0.1)

        # Verify
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "start")

        # Send stop command
        manager.stop_recording()

        # Receive stop command
        command = manager.get_record_command(timeout=0.1)
        self.assertEqual(command["action"], "stop")

    def test_playback_command_communication(self):
        """Test sending and receiving playback commands through real queues."""
        manager = AudioQueueManager()

        # Create buffer metadata
        buffer_metadata = {
            "name": "test_buffer",
            "shape": (1000, 1),
            "dtype": "float32",
        }

        # Send play command
        manager.start_playback(buffer_metadata, 44100)

        # Receive command
        command = manager.get_playback_command(timeout=0.1)

        # Verify
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "play")
        self.assertEqual(command["buffer_metadata"], buffer_metadata)
        self.assertEqual(command["sample_rate"], 44100)

        # Send stop command
        manager.stop_playback()

        # Receive stop command
        command = manager.get_playback_command(timeout=0.1)
        self.assertEqual(command["action"], "stop")

    def test_audio_data_queue_communication(self):
        """Test sending and receiving audio data through visualization queue."""
        manager = AudioQueueManager()

        # Create test audio data
        audio_data = np.random.random((1024, 1)).astype(np.float32)

        # Send data
        success = manager.put_audio_data(audio_data)
        self.assertTrue(success)

        # Receive data
        received_data = manager.get_audio_data(timeout=0.1)

        # Verify data integrity
        np.testing.assert_array_equal(received_data, audio_data)

    def test_queue_full_handling(self):
        """Test behavior when queue is full."""
        # Create manager with small queue
        manager = AudioQueueManager()

        # Fill the record queue (maxsize=10)
        for i in range(10):
            manager._record_queue.put({"action": f"test_{i}"}, block=False)

        # Try to add one more - should return False
        success = manager.set_input_device(99)
        self.assertFalse(success)

    def test_queue_empty_handling(self):
        """Test behavior when queue is empty."""
        manager = AudioQueueManager()

        # Try to get from empty queue
        command = manager.get_record_command(timeout=0.01)
        self.assertIsNone(command)

        command = manager.get_playback_command(timeout=0.01)
        self.assertIsNone(command)

    def test_invalid_command_type_raises_exception(self):
        """Test that non-dictionary commands raise TypeError."""
        manager = AudioQueueManager()

        # Put invalid command directly into queue
        manager._record_queue.put("not a dict")

        # Should raise TypeError
        with self.assertRaises(TypeError) as context:
            manager.get_record_command(timeout=0.1)

        self.assertIn("Expected dict command", str(context.exception))

    def test_cross_process_communication(self):
        """Test that queues work across process boundaries."""
        # Create manager in main process
        main_manager = AudioQueueManager()

        # Create result queue for test verification
        result_queue = mp.Queue()

        # Start child process using the module-level function
        process = mp.Process(
            target=child_process_func,
            args=(main_manager.record_queue, main_manager.playback_queue, result_queue),
        )
        process.start()

        # Send command from main process
        main_manager.start_recording()

        # Get result from child process
        result = result_queue.get(timeout=2.0)
        self.assertEqual(result, "start")

        # Receive response in main process
        response = main_manager.get_playback_command(timeout=2.0)
        self.assertEqual(response["action"], "response")

        # Clean up
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join()

    def test_device_configuration_commands(self):
        """Test device configuration command communication."""
        manager = AudioQueueManager()

        # Test input device setting
        success = manager.set_input_device(5)
        self.assertTrue(success)

        command = manager.get_record_command(timeout=0.1)
        self.assertEqual(command["action"], "set_input_device")
        self.assertEqual(command["device_name"], 5)  # Now passes device_name

        # Test output device setting
        success = manager.set_output_device(3)
        self.assertTrue(success)

        command = manager.get_playback_command(timeout=0.1)
        self.assertEqual(command["action"], "set_output_device")
        self.assertEqual(command["device_name"], 3)  # Now passes device_name

        # Test channel mapping
        mapping = [0, 2, 4]
        success = manager.set_input_channel_mapping(mapping)
        self.assertTrue(success)

        command = manager.get_record_command(timeout=0.1)
        self.assertEqual(command["action"], "set_input_channel_mapping")
        self.assertEqual(command["mapping"], mapping)

    def test_quit_commands(self):
        """Test quit command communication."""
        manager = AudioQueueManager()

        # Test record quit
        success = manager.quit_record_process()
        self.assertTrue(success)

        command = manager.get_record_command(timeout=0.1)
        self.assertEqual(command["action"], "quit")

        # Test playback quit
        success = manager.quit_playback_process()
        self.assertTrue(success)

        command = manager.get_playback_command(timeout=0.1)
        self.assertEqual(command["action"], "quit")


if __name__ == "__main__":
    unittest.main()
