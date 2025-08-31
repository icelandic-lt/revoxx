"""Settings manager for persisting user preferences."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class UserSettings:
    """User settings that persist between sessions.

    These settings override default configuration values.
    """

    # Audio settings
    sample_rate: int = 48000
    bit_depth: int = 24
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    input_channel_mapping: Optional[list] = (
        None  # list of ints (0-based) or None for default
    )
    output_channel_mapping: Optional[list] = (
        None  # list of ints (0-based) or None for default
    )
    audio_sync_response_time_ms: float = 10.0

    # Display settings
    show_meters: bool = True  # Combined spectrogram & level meter visibility
    show_info_overlay: bool = False
    show_info_panel: bool = True  # Combined info panel visibility
    fullscreen: bool = False
    theme: str = "cyan"  # Theme preset

    # Window settings - N-screen support
    windows: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.windows is None:
            self.windows = {}

    # Level meter settings
    level_meter_preset: str = "broadcast_ebu"
    level_meter_custom_target_min: float = -18.0
    level_meter_custom_target_max: float = -12.0
    level_meter_custom_warning: float = -6.0
    level_meter_custom_danger: float = -3.0

    # Dataset export settings
    base_sessions_dir: Optional[str] = None
    last_export_dir: Optional[str] = None
    export_format: str = "flac"
    export_include_intensity: bool = True

    # Text import settings
    import_input_dir: Optional[str] = None
    import_output_dir: Optional[str] = None

    # Session settings
    last_session_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSettings":
        """Create settings from dictionary."""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class SettingsManager:
    """Manages loading and saving user settings.

    Settings are stored in ~/.emospeech_settings as JSON.
    """

    def __init__(self):
        """Initialize settings manager."""
        self.settings_file = Path.home() / ".emospeech_settings"
        self.settings = self.load_settings()

    def load_settings(self) -> UserSettings:
        """Load settings from file.

        Returns:
            UserSettings object with loaded or default values
        """
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                return UserSettings.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading settings: {e}")
                print("Using default settings")

        return UserSettings()

    def save_settings(self) -> None:
        """Save current settings to file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def update_setting(self, key: str, value: Any) -> None:
        """Update a single setting and save.

        Args:
            key: Setting name
            value: New value
        """
        if hasattr(self.settings, key):
            setattr(self.settings, key, value)
            self.save_settings()
        # Silently ignore unknown settings (already filtered in from_dict)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            key: Setting name
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        return getattr(self.settings, key, default)

    def save_window_settings(
        self, window_id: str, window_settings: Dict[str, Any]
    ) -> None:
        """Save settings for a specific window.

        Args:
            window_id: Window identifier (e.g., 'main', 'monitor1')
            window_settings: Dictionary of window settings to save
        """
        if self.settings.windows is None:
            self.settings.windows = {}

        if window_id not in self.settings.windows:
            self.settings.windows[window_id] = {}

        self.settings.windows[window_id].update(window_settings)
        self.save_settings()
