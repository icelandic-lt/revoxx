"""Tests for DeviceManager default device resolution."""

import unittest
from unittest.mock import patch, PropertyMock

from revoxx.utils.device_manager import DeviceManager


class PortAudioError(Exception):
    """Stand-in for sd.PortAudioError in tests."""


SAMPLE_DEVICES = [
    {
        "name": "Built-in Mic",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
        "default_low_input_latency": 0.01,
        "default_low_output_latency": 0.0,
        "hostapi": 0,
    },
    {
        "name": "Built-in Speaker",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
        "default_low_input_latency": 0.0,
        "default_low_output_latency": 0.01,
        "hostapi": 0,
    },
    {
        "name": "USB Mic",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
        "default_low_input_latency": 0.005,
        "default_low_output_latency": 0.0,
        "hostapi": 0,
    },
    {
        "name": "Scarlett 2i2",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
        "default_low_input_latency": 0.01,
        "default_low_output_latency": 0.0,
        "hostapi": 0,
    },
    {
        "name": "HDMI Output",
        "max_input_channels": 0,
        "max_output_channels": 8,
        "default_samplerate": 48000.0,
        "default_low_input_latency": 0.0,
        "default_low_output_latency": 0.02,
        "hostapi": 0,
    },
    {
        "name": "DAC Output",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
        "default_low_input_latency": 0.0,
        "default_low_output_latency": 0.01,
        "hostapi": 0,
    },
]


def _make_query_devices(devices, kind_map=None):
    """Create a side_effect for sd.query_devices.

    Without arguments: returns the full devices list.
    With kind='input'/'output': returns kind_map[kind] or raises PortAudioError.
    """

    def query_devices(*args, **kwargs):
        if "kind" in kwargs:
            kind = kwargs["kind"]
            if kind_map and kind in kind_map:
                result = kind_map[kind]
                if isinstance(result, Exception):
                    raise result
                return result
            raise PortAudioError("No default device")
        return devices

    return query_devices


@patch("revoxx.utils.device_manager.sd")
class TestGetDefaultDeviceIndices(unittest.TestCase):
    """Tests for DeviceManager.get_default_device_indices()."""

    def _create_manager(self, mock_sd, devices=None, kind_map=None):
        if devices is None:
            devices = SAMPLE_DEVICES
        mock_sd.PortAudioError = PortAudioError
        mock_sd.query_devices.side_effect = _make_query_devices(devices, kind_map)
        return DeviceManager()

    def test_both_defaults_valid(self, mock_sd):
        """Valid indices in sd.default.device are returned directly."""
        mock_sd.default.device = [3, 5]
        dm = self._create_manager(mock_sd)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (3, 5))

    def test_both_minus_one(self, mock_sd):
        """[-1, -1] triggers PortAudio fallback for both devices."""
        mock_sd.default.device = [-1, -1]
        kind_map = {
            "input": {"name": "Built-in Mic", "hostapi": 0},
            "output": {"name": "Built-in Speaker", "hostapi": 0},
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (0, 1))

    def test_input_valid_output_minus_one(self, mock_sd):
        """Input index valid, output -1 resolved via PortAudio fallback."""
        mock_sd.default.device = [3, -1]
        kind_map = {
            "output": {"name": "DAC Output", "hostapi": 0},
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (3, 5))

    def test_input_minus_one_output_valid(self, mock_sd):
        """Input -1 resolved via fallback, output index returned directly."""
        mock_sd.default.device = [-1, 5]
        kind_map = {
            "input": {"name": "USB Mic", "hostapi": 0},
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (2, 5))

    def test_sd_default_raises(self, mock_sd):
        """When sd.default.device raises, both resolved via PortAudio fallback."""
        type(mock_sd.default).device = PropertyMock(
            side_effect=RuntimeError("PortAudio not initialized")
        )
        kind_map = {
            "input": {"name": "Built-in Mic", "hostapi": 0},
            "output": {"name": "Built-in Speaker", "hostapi": 0},
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (0, 1))

    def test_no_input_device(self, mock_sd):
        """When no input default exists, input index is None."""
        mock_sd.default.device = [-1, -1]
        kind_map = {
            "input": PortAudioError("No input device"),
            "output": {"name": "Built-in Speaker", "hostapi": 0},
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (None, 1))

    def test_no_output_device(self, mock_sd):
        """When no output default exists, output index is None."""
        mock_sd.default.device = [-1, -1]
        kind_map = {
            "input": {"name": "Built-in Mic", "hostapi": 0},
            "output": PortAudioError("No output device"),
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (0, None))

    def test_no_devices_at_all(self, mock_sd):
        """When both PortAudio queries fail, both indices are None."""
        mock_sd.default.device = [-1, -1]
        kind_map = {
            "input": PortAudioError("No input device"),
            "output": PortAudioError("No output device"),
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (None, None))

    def test_name_not_in_cache(self, mock_sd):
        """When PortAudio default name doesn't match any cached device, return None."""
        mock_sd.default.device = [-1, -1]
        kind_map = {
            "input": {"name": "Phantom Device", "hostapi": 99},
            "output": {"name": "Ghost Speaker", "hostapi": 99},
        }
        dm = self._create_manager(mock_sd, kind_map=kind_map)

        result = dm.get_default_device_indices()

        self.assertEqual(result, (None, None))


if __name__ == "__main__":
    unittest.main()
