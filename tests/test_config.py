"""Central test configuration for audio settings.

This module provides default audio configurations that work reliably
in CI/CD environments across different platforms.

Note: When running tests in PyCharm with the working directory set to 'tests/',
      imports from this module may fail with relative imports. To fix this:
      1. Change the working directory in your run configuration to the project root
      2. Or run tests using: python -m pytest tests/
      3. Or use pytest from the project root directory
"""

from revoxx.session.models import SessionConfig


class TestAudioConfig:
    """Default audio configuration for tests.

    Uses conservative settings that should work on most systems,
    including CI/CD environments without audio hardware.
    """

    # Standard CD-quality audio settings that are widely supported
    SAMPLE_RATE = 44100  # CD standard, universally supported
    BIT_DEPTH = 16  # CD standard, universally supported
    CHANNELS = 1  # Mono, simpler for testing
    FORMAT = "wav"  # Most compatible format

    @classmethod
    def get_default_config(cls) -> SessionConfig:
        """Get default test audio configuration.

        Returns:
            SessionConfig with conservative, widely-supported settings
        """
        return SessionConfig(
            sample_rate=cls.SAMPLE_RATE,
            bit_depth=cls.BIT_DEPTH,
            channels=cls.CHANNELS,
            format=cls.FORMAT,
        )

    @classmethod
    def get_config_with_device(cls, device_name: str) -> SessionConfig:
        """Get test config with specific device.

        Args:
            device_name: Name of the audio device

        Returns:
            SessionConfig with device specified
        """
        return SessionConfig(
            sample_rate=cls.SAMPLE_RATE,
            bit_depth=cls.BIT_DEPTH,
            channels=cls.CHANNELS,
            format=cls.FORMAT,
            input_device=device_name,
        )

    @classmethod
    def get_stereo_config(cls) -> SessionConfig:
        """Get test config for stereo audio.

        Returns:
            SessionConfig with 2 channels
        """
        return SessionConfig(
            sample_rate=cls.SAMPLE_RATE,
            bit_depth=cls.BIT_DEPTH,
            channels=2,
            format=cls.FORMAT,
        )

    @classmethod
    def get_high_quality_config(cls) -> SessionConfig:
        """Get higher quality test config.

        Note: This may not work in all CI environments.
        Use only when specifically testing high-quality audio.

        Returns:
            SessionConfig with higher sample rate and bit depth
        """
        return SessionConfig(
            sample_rate=48000, bit_depth=24, channels=cls.CHANNELS, format=cls.FORMAT
        )
