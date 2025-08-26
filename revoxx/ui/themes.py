"""Theme configuration system for UI styling."""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum
from matplotlib.colors import LinearSegmentedColormap


class ThemePreset(Enum):
    """Available theme presets."""

    OLIVE = "olive"
    CYAN = "cyan"


@dataclass
class ThemeColors:
    """Color configuration for a theme."""

    # Background colors
    COLOR_BACKGROUND: str
    COLOR_BACKGROUND_SECONDARY: str
    COLOR_BACKGROUND_TERTIARY: str

    # Text colors
    COLOR_TEXT_NORMAL: str
    COLOR_TEXT_SECONDARY: str
    COLOR_TEXT_RECORDING: str
    COLOR_TEXT_INACTIVE: str

    # Accent colors
    COLOR_ACCENT: str
    COLOR_WARNING: str
    COLOR_SUCCESS: str
    COLOR_BORDER: str

    # Specific UI elements
    COLOR_CLIPPING: str
    COLOR_PLAYBACK_LINE: str
    COLOR_EDGE_INDICATOR: str

    # Level meter colors
    LEVEL_COLOR_BACKGROUND: str
    LEVEL_COLOR_LED_OFF: str
    LEVEL_COLOR_OPTIMAL: str
    LEVEL_COLOR_WARNING: str
    LEVEL_COLOR_DANGER: str
    LEVEL_COLOR_LOW: str
    LEVEL_COLOR_RMS: str
    LEVEL_COLOR_PEAK: str
    LEVEL_COLOR_TEXT: str
    LEVEL_COLOR_GRID: str
    LEVEL_DIM_FACTOR: float


class Theme:
    """Theme definition with colors and colormap."""

    def __init__(self, name: str, colors: ThemeColors, colormap_name: str = None):
        """Initialize theme.

        Args:
            name: Theme name
            colors: Color configuration
            colormap_name: Name of matplotlib colormap or custom colormap
        """
        self.name = name
        self.colors = colors
        self.colormap_name = colormap_name
        self._colormap = None

    @property
    def colormap(self):
        """Get the colormap for this theme."""
        if self._colormap is None and self.colormap_name:
            if self.colormap_name == "cyan_spectrum":
                self._colormap = self._create_cyan_colormap()
            else:
                # Use built-in matplotlib colormap
                self._colormap = self.colormap_name
        return self._colormap

    @staticmethod
    def _create_cyan_colormap():
        """Create the cyan-orange colormap."""
        cyan_colors = {
            "red": [
                (0.0, 0.0, 0.0),  # Pure black background
                (0.15, 0.0, 0.0),  # Black
                (0.35, 0.0, 0.0),  # Still dark
                (0.55, 0.0, 0.0),  # Dark cyan-green
                (0.70, 0.0, 0.0),  # Medium cyan-green
                (0.80, 0.8, 0.8),  # Starting warm orange
                (0.90, 1.0, 1.0),  # Bright orange
                (0.95, 1.0, 1.0),  # Yellow-orange
                (1.0, 1.0, 1.0),
            ],  # Bright yellow for max
            "green": [
                (0.0, 0.0, 0.0),  # Pure black background
                (0.15, 0.12, 0.12),  # Subtle dark green-cyan
                (0.35, 0.35, 0.35),  # Dark green-cyan
                (0.55, 0.65, 0.65),  # Medium green-cyan (more green)
                (0.70, 0.9, 0.9),  # Bright green-cyan
                (0.80, 0.5, 0.5),  # Orange (medium green)
                (0.90, 0.65, 0.65),  # Orange-yellow
                (0.95, 0.85, 0.85),  # Yellow-orange (high green)
                (1.0, 1.0, 1.0),
            ],  # Bright yellow
            "blue": [
                (0.0, 0.0, 0.0),  # Pure black background
                (0.15, 0.1, 0.1),  # Dark blue
                (0.35, 0.3, 0.3),  # Medium dark blue
                (0.55, 0.55, 0.55),  # Cyan-green blue (less blue)
                (0.70, 0.75, 0.75),  # Bright cyan (less blue)
                (0.80, 0.0, 0.0),  # Orange (no blue)
                (0.90, 0.0, 0.0),  # Orange-yellow (no blue)
                (0.95, 0.0, 0.0),  # Yellow-orange (no blue)
                (1.0, 0.0, 0.0),
            ],  # Yellow (no blue)
        }
        return LinearSegmentedColormap("cyan_spectrum", cyan_colors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert theme to dictionary for easy access."""
        return {
            "name": self.name,
            "colors": self.colors.__dict__,
            "colormap": self.colormap,
        }


# Define theme presets
THEMES = {
    ThemePreset.OLIVE: Theme(
        name="Olive",
        colors=ThemeColors(
            # Background colors
            COLOR_BACKGROUND="#000000",  # Pure black
            COLOR_BACKGROUND_SECONDARY="#0A0F14",  # Slightly elevated for bottom panel
            COLOR_BACKGROUND_TERTIARY="#000000",  # Pure black
            # Text colors
            COLOR_TEXT_NORMAL="#40C040",  # Pleasant green
            COLOR_TEXT_SECONDARY="#40C040",  # Pleasant green
            COLOR_TEXT_RECORDING="#FF0000",  # Red
            COLOR_TEXT_INACTIVE="#808080",  # Gray
            # Accent colors
            COLOR_ACCENT="#40C040",  # Pleasant green
            COLOR_WARNING="#FFFF00",  # Yellow
            COLOR_SUCCESS="#40C040",  # Pleasant green
            COLOR_BORDER="#808080",  # Gray
            # Specific UI elements
            COLOR_CLIPPING="#FF0000",  # Red
            COLOR_PLAYBACK_LINE="#FF0000",  # Red
            COLOR_EDGE_INDICATOR="#00FF00",  # Lime green
            # Level meter colors
            LEVEL_COLOR_BACKGROUND="#000000",  # Pure black
            LEVEL_COLOR_LED_OFF="#2d2d2d",
            LEVEL_COLOR_OPTIMAL="#4CAF50",
            LEVEL_COLOR_WARNING="#FFC107",
            LEVEL_COLOR_DANGER="#F44336",
            LEVEL_COLOR_LOW="#2196F3",
            LEVEL_COLOR_RMS="#FFFFFF",
            LEVEL_COLOR_PEAK="#FF9800",
            LEVEL_COLOR_TEXT="#FFFFFF",
            LEVEL_COLOR_GRID="#444444",
            LEVEL_DIM_FACTOR=0.25,
        ),
        colormap_name="viridis",
    ),
    ThemePreset.CYAN: Theme(
        name="Cyan",
        colors=ThemeColors(
            # Background colors
            COLOR_BACKGROUND="#000000",  # Pure black background
            COLOR_BACKGROUND_SECONDARY="#0A0F14",  # Slightly lighter for panels
            COLOR_BACKGROUND_TERTIARY="#141B22",  # For frames and borders
            # Text colors
            COLOR_TEXT_NORMAL="#00E5FF",
            COLOR_TEXT_SECONDARY="#00ACC1",
            COLOR_TEXT_RECORDING="#FF0080",
            COLOR_TEXT_INACTIVE="#4A5568",
            # Accent colors
            COLOR_ACCENT="#00FFFF",
            COLOR_WARNING="#FF00FF",
            COLOR_SUCCESS="#00FF88",
            COLOR_BORDER="#1F2937",
            # Specific UI elements
            COLOR_CLIPPING="#FF0080",
            COLOR_PLAYBACK_LINE="#FF0080",
            COLOR_EDGE_INDICATOR="#00FF88",
            # Level meter colors
            LEVEL_COLOR_BACKGROUND="#000000",  # Pure black
            LEVEL_COLOR_LED_OFF="#0A0F14",  # Dark gray for off LEDs
            LEVEL_COLOR_OPTIMAL="#00E5FF",
            LEVEL_COLOR_WARNING="#FF9800",  # Orange
            LEVEL_COLOR_DANGER="#FF0080",
            LEVEL_COLOR_LOW="#006080",
            LEVEL_COLOR_RMS="#00FFFF",
            LEVEL_COLOR_PEAK="#FF0080",  # Magenta
            LEVEL_COLOR_TEXT="#00E5FF",
            LEVEL_COLOR_GRID="#1F2937",
            LEVEL_DIM_FACTOR=0.15,
        ),
        colormap_name="cyan_spectrum",
    ),
}


class ThemeManager:
    """Manages the current theme and provides access to theme properties."""

    def __init__(self, initial_theme: ThemePreset = ThemePreset.CYAN):
        """Initialize theme manager.

        Args:
            initial_theme: Initial theme to use
        """
        self._current_preset = initial_theme
        self._current_theme = THEMES[initial_theme]
        self._callbacks = []

    @property
    def current_theme(self) -> Theme:
        """Get the current theme."""
        return self._current_theme

    @property
    def current_preset(self) -> ThemePreset:
        """Get the current theme preset."""
        return self._current_preset

    @property
    def colors(self) -> ThemeColors:
        """Get current theme colors."""
        return self._current_theme.colors

    @property
    def colormap(self):
        """Get current theme colormap."""
        return self._current_theme.colormap

    def set_theme(self, preset: ThemePreset) -> None:
        """Change the current theme.

        Args:
            preset: Theme preset to switch to
        """
        if preset != self._current_preset:
            self._current_preset = preset
            self._current_theme = THEMES[preset]
            self._notify_callbacks()

    def register_callback(self, callback) -> None:
        """Register a callback to be notified when theme changes.

        Args:
            callback: Function to call when theme changes
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks about theme change."""
        for callback in self._callbacks:
            try:
                callback(self._current_theme)
            except Exception:
                pass

    def get_available_themes(self) -> Dict[str, str]:
        """Get available theme names.

        Returns:
            Dictionary of preset values to display names
        """
        return {preset.value: theme.name for preset, theme in THEMES.items()}


# Global theme manager instance
theme_manager = ThemeManager()
