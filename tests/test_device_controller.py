"""Tests for the DeviceController."""

import unittest
from unittest.mock import Mock, patch

from revoxx.controllers.device_controller import DeviceController


class TestDeviceController(unittest.TestCase):
    """Test cases for DeviceController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock app with all required attributes
        self.mock_app = Mock()

        # Mock config
        self.mock_app.config = Mock()
        self.mock_app.config.audio = Mock()
        self.mock_app.config.audio.input_device = None
        self.mock_app.config.audio.output_device = None
        self.mock_app.config.audio.channels = 1
        self.mock_app.config.audio.sample_rate = 48000
        self.mock_app.config.audio.bit_depth = 24
        self.mock_app.config.audio.input_channel_mapping = None
        self.mock_app.config.audio.output_channel_mapping = None

        # Mock window
        self.mock_app.window = Mock()

        # Mock settings manager
        self.mock_app.settings_manager = Mock()
        self.mock_app.settings_manager.get_setting = Mock(return_value=None)

        # Mock queue_manager instead of direct queues
        self.mock_app.queue_manager = Mock()
        self.mock_app.queue_manager.set_input_device = Mock(return_value=True)
        self.mock_app.queue_manager.set_output_device = Mock(return_value=True)
        self.mock_app.queue_manager.set_input_channel_mapping = Mock(return_value=True)
        self.mock_app.queue_manager.set_output_channel_mapping = Mock(return_value=True)

        # Mock shared state
        self.mock_app.shared_state = Mock()

        # Mock current session
        self.mock_app.current_session = None

        self.controller = DeviceController(self.mock_app)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_device_valid(self, mock_get_dm):
        """Test setting a valid input device."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[
                {"index": 0, "name": "Device 0"},
                {"index": 1, "name": "Device 1"},
            ]
        )
        mock_get_dm.return_value = mock_dm

        mock_dm.get_device_name_by_index = Mock(return_value="Device 1")

        # Execute
        self.controller.set_input_device(1)

        # Verify
        self.assertEqual(self.mock_app.config.audio.input_device, "Device 1")
        self.mock_app.settings_manager.update_setting.assert_called_with(
            "input_device", "Device 1"
        )
        self.mock_app.queue_manager.set_input_device.assert_called_with("Device 1")
        self.assertFalse(self.controller.is_default_input_active)
        self.mock_app.window.set_status.assert_called_with("Input device: Device 1")

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_device_none(self, mock_get_dm):
        """Test setting input device to None (system default)."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(return_value=[])
        mock_get_dm.return_value = mock_dm

        # Execute
        self.controller.set_input_device(None)

        # Verify
        self.assertIsNone(self.mock_app.config.audio.input_device)
        self.mock_app.settings_manager.update_setting.assert_called_with(
            "input_device", None
        )
        self.mock_app.queue_manager.set_input_device.assert_called_with(None)
        self.assertTrue(self.controller.is_default_input_active)
        self.mock_app.window.set_status.assert_called_with(
            "Input device: System Default"
        )

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_device_no_save(self, mock_get_dm):
        """Test setting input device without saving."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[{"index": 1, "name": "Device 1"}]
        )
        mock_get_dm.return_value = mock_dm

        mock_dm.get_device_name_by_index = Mock(return_value="Device 1")

        # Execute
        self.controller.set_input_device(1, save=False)

        # Verify
        self.assertEqual(self.mock_app.config.audio.input_device, "Device 1")
        self.mock_app.settings_manager.update_setting.assert_not_called()

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_device_queue_full(self, mock_get_dm):
        """Test setting input device when queue is full."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[{"index": 1, "name": "Device 1"}]
        )
        mock_get_dm.return_value = mock_dm
        self.mock_app.queue_manager.set_input_device = Mock(return_value=False)

        mock_dm.get_device_name_by_index = Mock(return_value="Device 1")

        # Execute - should not raise exception
        self.controller.set_input_device(1)

        # Verify - should still update config
        self.assertEqual(self.mock_app.config.audio.input_device, "Device 1")

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_output_device_valid(self, mock_get_dm):
        """Test setting a valid output device."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_output_devices = Mock(
            return_value=[
                {"index": 0, "name": "Device 0"},
                {"index": 2, "name": "Device 2"},
            ]
        )
        mock_get_dm.return_value = mock_dm

        mock_dm.get_device_name_by_index = Mock(return_value="Device 2")

        # Execute
        self.controller.set_output_device(2)

        # Verify
        self.assertEqual(self.mock_app.config.audio.output_device, "Device 2")
        self.mock_app.settings_manager.update_setting.assert_called_with(
            "output_device", "Device 2"
        )
        self.mock_app.queue_manager.set_output_device.assert_called_with("Device 2")
        self.assertFalse(self.controller.is_default_output_active)
        self.mock_app.window.set_status.assert_called_with("Output device: Device 2")

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_output_device_none(self, mock_get_dm):
        """Test setting output device to None (system default)."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_output_devices = Mock(return_value=[])
        mock_get_dm.return_value = mock_dm

        # Execute
        self.controller.set_output_device(None)

        # Verify
        self.assertIsNone(self.mock_app.config.audio.output_device)
        self.assertTrue(self.controller.is_default_output_active)
        self.mock_app.window.set_status.assert_called_with(
            "Output device: System Default"
        )

    def test_set_input_channel_mapping(self):
        """Test setting input channel mapping."""
        # Execute
        self.controller.set_input_channel_mapping([0, 1])

        # Verify
        self.assertEqual(self.mock_app.config.audio.input_channel_mapping, [0, 1])
        self.mock_app.settings_manager.update_setting.assert_called_with(
            "input_channel_mapping", [0, 1]
        )
        self.mock_app.queue_manager.set_input_channel_mapping.assert_called_with([0, 1])
        self.mock_app.window.set_status.assert_called_with(
            "Input channel mapping: [0, 1]"
        )

    def test_set_input_channel_mapping_none(self):
        """Test setting input channel mapping to None."""
        # Execute
        self.controller.set_input_channel_mapping(None)

        # Verify
        self.assertIsNone(self.mock_app.config.audio.input_channel_mapping)
        self.mock_app.window.set_status.assert_called_with(
            "Input channel mapping: Default"
        )

    def test_set_output_channel_mapping(self):
        """Test setting output channel mapping."""
        # Execute
        self.controller.set_output_channel_mapping([1, 0])

        # Verify
        self.assertEqual(self.mock_app.config.audio.output_channel_mapping, [1, 0])
        self.mock_app.settings_manager.update_setting.assert_called_with(
            "output_channel_mapping", [1, 0]
        )
        self.mock_app.queue_manager.set_output_channel_mapping.assert_called_with(
            [1, 0]
        )
        self.mock_app.window.set_status.assert_called_with(
            "Output channel mapping: [1, 0]"
        )

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_get_available_input_devices(self, mock_get_dm):
        """Test getting available input devices."""
        # Setup
        mock_dm = Mock()
        expected_devices = [
            {"index": 0, "name": "Device 0"},
            {"index": 1, "name": "Device 1"},
        ]
        mock_dm.get_input_devices = Mock(return_value=expected_devices)
        mock_get_dm.return_value = mock_dm

        # Execute
        result = self.controller.get_available_input_devices()

        # Verify
        self.assertEqual(result, expected_devices)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_get_available_output_devices(self, mock_get_dm):
        """Test getting available output devices."""
        # Setup
        mock_dm = Mock()
        expected_devices = [
            {"index": 0, "name": "Device 0"},
            {"index": 2, "name": "Device 2"},
        ]
        mock_dm.get_output_devices = Mock(return_value=expected_devices)
        mock_get_dm.return_value = mock_dm

        # Execute
        result = self.controller.get_available_output_devices()

        # Verify
        self.assertEqual(result, expected_devices)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_validate_device_compatibility(self, mock_get_dm):
        """Test validating device compatibility."""
        # Setup
        mock_dm = Mock()
        mock_dm.check_device_compatibility = Mock(return_value=True)
        mock_get_dm.return_value = mock_dm

        # Execute
        result = self.controller.validate_device_compatibility(
            "Test Device", is_input=True
        )

        # Verify
        self.assertTrue(result)
        mock_dm.check_device_compatibility.assert_called_once_with(
            device_name="Test Device",
            sample_rate=48000,
            bit_depth=24,
            channels=1,
            is_input=True,
        )

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_find_compatible_device_found(self, mock_get_dm):
        """Test finding a compatible device when one exists."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[
                {"index": 0, "name": "Device 0"},
                {"index": 1, "name": "Device 1"},
            ]
        )
        mock_get_dm.return_value = mock_dm

        with patch.object(
            self.controller, "validate_device_compatibility"
        ) as mock_validate:
            mock_validate.side_effect = [
                False,
                True,
            ]  # First device incompatible, second compatible

            # Execute
            result = self.controller.find_compatible_device(is_input=True)

            # Verify
            self.assertEqual(result, 1)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_find_compatible_device_not_found(self, mock_get_dm):
        """Test finding a compatible device when none exist."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[{"index": 0, "name": "Device 0"}]
        )
        mock_get_dm.return_value = mock_dm

        with patch.object(
            self.controller, "validate_device_compatibility"
        ) as mock_validate:
            mock_validate.return_value = False

            # Execute
            result = self.controller.find_compatible_device(is_input=True)

            # Verify
            self.assertIsNone(result)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_refresh_devices(self, mock_get_dm):
        """Test refreshing device list."""
        # Setup
        mock_dm = Mock()
        mock_get_dm.return_value = mock_dm

        # Execute
        self.controller.refresh_devices()

        # Verify
        mock_dm.refresh.assert_called_once()

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_refresh_devices_error(self, mock_get_dm):
        """Test refreshing devices with error."""
        # Setup
        mock_get_dm.side_effect = ImportError()

        # Execute - should not raise exception
        self.controller.refresh_devices()

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_apply_saved_settings(self, mock_get_dm):
        """Test applying saved device settings."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_device_index_by_name = Mock(
            side_effect=[1, 2]
        )  # Return indices for the device names
        mock_get_dm.return_value = mock_dm

        self.mock_app.settings_manager.get_setting.side_effect = [
            "Device 1",  # input_device (now a name)
            "Device 2",  # output_device (now a name)
            [0],  # input_channel_mapping
            [1, 0],  # output_channel_mapping
        ]

        with patch.object(self.controller, "set_input_device") as mock_set_input:
            with patch.object(self.controller, "set_output_device") as mock_set_output:
                with patch.object(
                    self.controller, "set_input_channel_mapping"
                ) as mock_set_input_map:
                    with patch.object(
                        self.controller, "set_output_channel_mapping"
                    ) as mock_set_output_map:
                        # Execute
                        self.controller.apply_saved_settings()

                        # Verify
                        mock_set_input.assert_called_once_with(1, save=False)
                        mock_set_output.assert_called_once_with(2, save=False)
                        mock_set_input_map.assert_called_once_with([0], save=False)
                        mock_set_output_map.assert_called_once_with([1, 0], save=False)

    def test_mark_input_notified(self):
        """Test marking input device notification."""
        self.assertFalse(self.controller.has_notified_default_input)
        self.controller.mark_input_notified()
        self.assertTrue(self.controller.has_notified_default_input)

    def test_mark_output_notified(self):
        """Test marking output device notification."""
        self.assertFalse(self.controller.has_notified_default_output)
        self.controller.mark_output_notified()
        self.assertTrue(self.controller.has_notified_default_output)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_device_updates_session(self, mock_get_dm):
        """Test that setting input device updates session configuration."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[{"index": 1, "name": "Test Device"}]
        )
        mock_dm.get_device_name_by_index = Mock(return_value="Test Device")
        mock_get_dm.return_value = mock_dm

        # Create mock session
        mock_session = Mock()
        mock_session.audio_config = Mock()
        mock_session.audio_config.input_device = None
        mock_session.audio_config.input_channel_mapping = None
        self.mock_app.current_session = mock_session

        # Execute
        self.controller.set_input_device(1)

        # Verify session was updated
        self.assertEqual(mock_session.audio_config.input_device, "Test Device")
        mock_session.save.assert_called_once()

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_device_resets_channel_mapping_on_change(self, mock_get_dm):
        """Test that changing device resets channel mapping."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_input_devices = Mock(
            return_value=[{"index": 2, "name": "New Device"}]
        )
        mock_dm.get_device_name_by_index = Mock(return_value="New Device")
        mock_get_dm.return_value = mock_dm

        # Create mock session with existing device
        mock_session = Mock()
        mock_session.audio_config = Mock()
        mock_session.audio_config.input_device = "Old Device"
        mock_session.audio_config.input_channel_mapping = [0, 1]
        self.mock_app.current_session = mock_session

        # Set initial device to simulate a change
        self.mock_app.config.audio.input_device = 1

        # Execute (changing from device 1 to device 2)
        self.controller.set_input_device(2)

        # Verify channel mapping was reset
        self.assertIsNone(mock_session.audio_config.input_channel_mapping)
        mock_session.save.assert_called_once()

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_set_input_channel_mapping_updates_session(self, mock_get_dm):
        """Test that setting channel mapping updates session."""
        # Setup
        mock_session = Mock()
        mock_session.audio_config = Mock()
        mock_session.audio_config.input_channel_mapping = None
        self.mock_app.current_session = mock_session

        # Execute
        self.controller.set_input_channel_mapping([1, 0])

        # Verify
        self.assertEqual(mock_session.audio_config.input_channel_mapping, [1, 0])
        mock_session.save.assert_called_once()

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_apply_session_audio_settings_with_valid_device(self, mock_get_dm):
        """Test applying session settings when device is available."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_device_index_by_name = Mock(return_value=3)
        mock_get_dm.return_value = mock_dm

        # Create mock session config
        mock_config = Mock()
        mock_config.input_device = "Test Device"
        mock_config.input_channel_mapping = [0]
        mock_config.output_device = "Output Device"
        mock_config.output_channel_mapping = [1, 0]

        with patch.object(self.controller, "set_input_device") as mock_set_input:
            with patch.object(self.controller, "set_output_device") as mock_set_output:
                with patch.object(
                    self.controller, "set_input_channel_mapping"
                ) as mock_set_input_map:
                    with patch.object(
                        self.controller, "set_output_channel_mapping"
                    ) as mock_set_output_map:
                        # Execute
                        self.controller.apply_session_audio_settings(mock_config)

                        # Verify devices were set
                        mock_set_input.assert_called_once_with(3, save=False)
                        mock_set_output.assert_called_once_with(3, save=False)
                        mock_set_input_map.assert_called_once_with([0], save=False)
                        mock_set_output_map.assert_called_once_with([1, 0], save=False)

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_apply_session_audio_settings_with_missing_device(self, mock_get_dm):
        """Test applying session settings when device is not available."""
        # Setup
        mock_dm = Mock()
        mock_dm.get_device_index_by_name = Mock(return_value=None)  # Device not found
        mock_get_dm.return_value = mock_dm

        # Create mock session config without output_device attribute
        mock_config = Mock(spec=["input_device", "input_channel_mapping"])
        mock_config.input_device = "Missing Device"
        mock_config.input_channel_mapping = [0]

        with patch.object(self.controller, "set_input_device") as mock_set_input:
            with patch.object(
                self.controller, "set_input_channel_mapping"
            ) as mock_set_input_map:
                # Execute
                self.controller.apply_session_audio_settings(mock_config)

                # Verify fallback to default
                mock_set_input.assert_called_once_with(None, save=False)
                mock_set_input_map.assert_called_once_with(None, save=False)
                self.mock_app.window.set_status.assert_called_with(
                    "Input device not found, using default"
                )

    @patch("revoxx.controllers.device_controller.get_device_manager")
    def test_apply_session_audio_settings_with_default_device(self, mock_get_dm):
        """Test applying session settings with default device."""
        # Setup
        mock_config = Mock()
        mock_config.input_device = "default"
        mock_config.input_channel_mapping = None

        with patch.object(self.controller, "set_input_device") as mock_set_input:
            with patch.object(
                self.controller, "set_input_channel_mapping"
            ) as mock_set_input_map:
                # Execute
                self.controller.apply_session_audio_settings(mock_config)

                # Verify default was set
                mock_set_input.assert_called_once_with(None, save=False)
                mock_set_input_map.assert_called_once_with(None, save=False)


if __name__ == "__main__":
    unittest.main()
