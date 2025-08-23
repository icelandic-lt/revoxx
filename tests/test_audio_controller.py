"""Tests for the AudioController."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from revoxx.controllers.audio_controller import AudioController


class TestAudioController(unittest.TestCase):
    """Test cases for AudioController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()
        self.mock_app.state = Mock()
        self.mock_app.state.recording = Mock()
        self.mock_app.state.recording.is_recording = False
        self.mock_app.state.recording.current_label = "test_label"
        self.mock_app.state.is_ready_to_play = Mock(return_value=True)

        self.mock_app.window = Mock()
        self.mock_app.window.mel_spectrogram = Mock()
        self.mock_app.window.info_overlay = Mock()
        self.mock_app.window.info_overlay.visible = False
        self.mock_app.window.level_meter_var = Mock()
        self.mock_app.window.level_meter_var.get = Mock(return_value=False)
        self.mock_app.window.embedded_level_meter = Mock()
        self.mock_app.window.monitoring_var = Mock()

        self.mock_app.shared_state = Mock()
        self.mock_app.config = Mock()
        self.mock_app.config.audio = Mock()
        self.mock_app.config.audio.sample_rate = 48000
        self.mock_app.config.audio.bit_depth = 24
        self.mock_app.config.audio.channels = 1
        self.mock_app.config.audio.input_device = None
        self.mock_app.config.audio.output_device = None

        self.mock_app.file_manager = Mock()
        self.mock_app.file_manager.get_next_take_number = Mock(return_value=1)
        self.mock_app.file_manager.get_recording_path = Mock(
            return_value=Path("/test/path.wav")
        )
        self.mock_app.file_manager.load_audio = Mock(
            return_value=([0.1, 0.2, 0.3], 48000)
        )

        self.mock_app.buffer_manager = Mock()
        self.mock_app.buffer_manager.create_buffer = Mock()

        self.mock_app.manager_dict = {}
        self.mock_app.record_queue = Mock()
        self.mock_app.playback_queue = Mock()
        self.mock_app.root = Mock()
        self.mock_app.settings_manager = Mock()

        self.mock_app._default_input_in_effect = False
        self.mock_app._default_output_in_effect = False
        self.mock_app._notified_default_input = False
        self.mock_app._notified_default_output = False
        self.mock_app.last_output_error = False

        # Add display_controller mock for new architecture
        self.mock_app.display_controller = Mock()
        self.mock_app.display_controller.update_display = Mock()
        self.mock_app.display_controller.show_saved_recording = Mock()
        self.mock_app.display_controller.update_info_overlay = Mock()
        self.mock_app.display_controller.toggle_mel_spectrogram = Mock()

        self.controller = AudioController(self.mock_app)

    def test_toggle_recording_starts_when_not_recording(self):
        """Test that toggle_recording starts recording when not currently recording."""
        self.mock_app.state.recording.is_recording = False

        with patch.object(self.controller, "start_recording") as mock_start:
            self.controller.toggle_recording()
            mock_start.assert_called_once()

    def test_toggle_recording_stops_when_recording(self):
        """Test that toggle_recording stops recording when currently recording."""
        self.mock_app.state.recording.is_recording = True

        with patch.object(self.controller, "stop_recording") as mock_stop:
            self.controller.toggle_recording()
            mock_stop.assert_called_once()

    def test_start_recording_calls_start_audio_capture(self):
        """Test that start_recording calls _start_audio_capture with 'recording' mode."""
        with patch.object(self.controller, "_start_audio_capture") as mock_capture:
            self.controller.start_recording()
            mock_capture.assert_called_once_with("recording")

    def test_stop_recording_calls_stop_audio_capture(self):
        """Test that stop_recording calls _stop_audio_capture with 'recording' mode."""
        with patch.object(self.controller, "_stop_audio_capture") as mock_capture:
            self.controller.stop_recording()
            mock_capture.assert_called_once_with("recording")

    @patch("revoxx.controllers.audio_controller.sd")
    @patch("revoxx.controllers.audio_controller.time.sleep")
    def test_play_current_with_no_recordings(self, mock_sleep, mock_sd):
        """Test play_current when no recordings are available."""
        self.mock_app.state.is_ready_to_play = Mock(return_value=False)

        self.controller.play_current()

        self.mock_app.window.show_message.assert_called_once_with(
            "No recording available"
        )
        mock_sd.stop.assert_not_called()

    @patch("revoxx.controllers.audio_controller.sd")
    @patch("revoxx.controllers.audio_controller.time.sleep")
    @patch("revoxx.controllers.audio_controller.get_device_manager")
    def test_play_current_with_recording(self, mock_get_dm, mock_sleep, mock_sd):
        """Test play_current when a recording is available."""
        # Setup
        self.mock_app.state.recording.get_current_take = Mock(return_value=1)
        mock_path = Mock()
        mock_path.exists = Mock(return_value=True)
        self.mock_app.file_manager.get_recording_path = Mock(return_value=mock_path)

        mock_buffer = Mock()
        mock_buffer.get_metadata = Mock(return_value={"test": "metadata"})
        mock_buffer.close = Mock()
        self.mock_app.buffer_manager.create_buffer = Mock(return_value=mock_buffer)

        # Execute
        self.controller.play_current()

        # Verify
        mock_sd.stop.assert_called_once()
        self.mock_app.playback_queue.put.assert_any_call({"action": "stop"})
        self.mock_app.window.mel_spectrogram.stop_playback.assert_called_once()
        self.mock_app.shared_state.reset_level_meter.assert_called()
        self.mock_app.file_manager.load_audio.assert_called_once_with(mock_path)
        mock_buffer.close.assert_called_once()

    def test_toggle_monitoring_starts_when_not_monitoring(self):
        """Test that toggle_monitoring starts monitoring when not currently monitoring."""
        self.controller.is_monitoring = False

        with patch.object(self.controller, "start_monitoring_mode") as mock_start:
            self.controller.toggle_monitoring()
            mock_start.assert_called_once()

    def test_toggle_monitoring_stops_when_monitoring(self):
        """Test that toggle_monitoring stops monitoring when currently monitoring."""
        self.controller.is_monitoring = True

        with patch.object(self.controller, "stop_monitoring_mode") as mock_stop:
            self.controller.toggle_monitoring()
            mock_stop.assert_called_once()

    def test_start_monitoring_mode_calls_start_audio_capture(self):
        """Test that start_monitoring_mode calls _start_audio_capture with 'monitoring' mode."""
        with patch.object(self.controller, "_start_audio_capture") as mock_capture:
            self.controller.start_monitoring_mode()
            mock_capture.assert_called_once_with("monitoring")

    def test_stop_monitoring_mode_calls_stop_audio_capture(self):
        """Test that stop_monitoring_mode calls _stop_audio_capture with 'monitoring' mode."""
        with patch.object(self.controller, "_stop_audio_capture") as mock_capture:
            self.controller.stop_monitoring_mode()
            mock_capture.assert_called_once_with("monitoring")

    @patch("revoxx.controllers.audio_controller.get_device_manager")
    def test_start_audio_capture_recording_mode(self, mock_get_dm):
        """Test _start_audio_capture in recording mode."""
        # Setup
        mock_dm = Mock()
        mock_get_dm.return_value = mock_dm

        # Execute
        self.controller._start_audio_capture("recording")

        # Verify
        self.assertTrue(self.mock_app.state.recording.is_recording)
        self.mock_app.file_manager.get_next_take_number.assert_called_once_with(
            "test_label"
        )
        self.mock_app.file_manager.get_recording_path.assert_called_once_with(
            "test_label", 1
        )
        self.assertIn("save_path", self.mock_app.manager_dict)
        self.mock_app.record_queue.put.assert_called_with({"action": "start"})
        self.mock_app.window.mel_spectrogram.clear.assert_called_once()
        self.mock_app.window.mel_spectrogram.start_recording.assert_called_once_with(
            48000
        )

    @patch("revoxx.controllers.audio_controller.get_device_manager")
    def test_start_audio_capture_monitoring_mode(self, mock_get_dm):
        """Test _start_audio_capture in monitoring mode."""
        # Setup
        mock_dm = Mock()
        mock_get_dm.return_value = mock_dm
        self.mock_app.state.ui = Mock()
        self.mock_app.state.ui.spectrogram_visible = True

        # Execute
        self.controller._start_audio_capture("monitoring")

        # Verify
        self.assertTrue(self.controller.is_monitoring)
        self.assertEqual(self.controller.saved_spectrogram_state, True)
        self.assertFalse(self.mock_app.state.recording.is_recording)
        self.mock_app.record_queue.put.assert_called_with({"action": "start"})
        self.mock_app.window.set_status.assert_called_with("Monitoring input levels...")

    def test_start_audio_capture_invalid_mode(self):
        """Test _start_audio_capture with invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.controller._start_audio_capture("invalid")

        self.assertIn("Invalid mode: invalid", str(context.exception))

    def test_stop_audio_capture_recording_mode(self):
        """Test _stop_audio_capture in recording mode."""
        # Execute
        self.controller._stop_audio_capture("recording")

        # Verify
        self.assertFalse(self.mock_app.state.recording.is_recording)
        self.mock_app.record_queue.put.assert_called_with({"action": "stop"})
        self.mock_app.window.mel_spectrogram.stop_recording.assert_called_once()
        self.mock_app.display_controller.update_display.assert_called_once()

    def test_stop_audio_capture_monitoring_mode(self):
        """Test _stop_audio_capture in monitoring mode."""
        # Setup
        self.controller.is_monitoring = True
        self.controller.saved_spectrogram_state = False
        self.mock_app.state.ui = Mock()
        self.mock_app.state.ui.spectrogram_visible = True

        # Execute
        self.controller._stop_audio_capture("monitoring")

        # Verify
        self.assertFalse(self.controller.is_monitoring)
        self.mock_app.record_queue.put.assert_called_with({"action": "stop"})
        self.mock_app.window.mel_spectrogram.stop_recording.assert_called_once()
        self.mock_app.window.set_status.assert_called_with("Ready")
        self.mock_app.display_controller.show_saved_recording.assert_called_once()

    def test_stop_audio_capture_invalid_mode(self):
        """Test _stop_audio_capture with invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.controller._stop_audio_capture("invalid")

        self.assertIn("Invalid mode: invalid", str(context.exception))

    def test_stop_synchronized_playback(self):
        """Test stop_synchronized_playback sends stop command and resets meter."""
        self.controller.stop_synchronized_playback()

        self.mock_app.playback_queue.put.assert_called_once_with({"action": "stop"})
        self.mock_app.window.embedded_level_meter.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
