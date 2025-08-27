"""Font management for UI components."""

from typing import Tuple, Optional, List
import tkinter.font as tkfont

from ..constants import UIConstants
from ..utils.state import UIState
from ..utils.config import RecorderConfig


class FontManager:
    """Manages font sizing and text layout for the application.

    This class handles:
    - Dynamic font size calculation based on window dimensions
    - Text fitting and wrapping for available space
    - Font family selection with fallbacks
    - Adaptive text sizing for long content
    """

    def __init__(self, ui_state: UIState, config: RecorderConfig):
        """Initialize the font manager.

        Args:
            ui_state: UI state containing window dimensions and font sizes
            config: Application configuration with base font settings
        """
        self.ui_state = ui_state
        self.config = config
        self._font_cache = {}  # Cache for tkfont.Font objects

    def calculate_base_sizes(self, window_width: int, window_height: int) -> None:
        """Calculate base font sizes based on window dimensions.

        Updates ui_state with calculated font sizes that scale
        proportionally with window size.

        Args:
            window_width: Current window width in pixels
            window_height: Current window height in pixels
        """
        # Scale factor based on window size
        scale_factor = min(window_width / 1200, window_height / 900)

        # Update UI state with calculated sizes
        self.ui_state.calculate_font_sizes(self.config.ui.base_font_size, scale_factor)

    @staticmethod
    def get_font_with_fallback(
        font_family: Tuple[str, ...], preferred_type: str = "mono"
    ) -> str:
        """Get an available font from family list with fallback.

        Args:
            font_family: Tuple of font names in preference order
            preferred_type: Type of font ("mono" or "sans") for fallback

        Returns:
            Name of available font
        """
        # Try each font in the family
        for font in font_family:
            try:
                if font in tkfont.families():
                    return font
            except Exception:
                pass

        # Return system default based on type
        return "Courier" if preferred_type == "mono" else "Helvetica"

    def get_mono_font(self) -> str:
        """Get available monospace font.

        Returns:
            Name of available monospace font
        """
        return self.get_font_with_fallback(UIConstants.FONT_FAMILY_MONO, "mono")

    def get_sans_font(self) -> str:
        """Get available sans-serif font.

        Returns:
            Name of available sans-serif font
        """
        return self.get_font_with_fallback(UIConstants.FONT_FAMILY_SANS, "sans")

    def calculate_adaptive_font_size(
        self,
        text: str,
        available_width: int,
        available_height: int,
        max_font_size: Optional[int] = None,
        min_font_size: int = 14,
        use_mono_font: bool = False,
    ) -> Tuple[int, int]:
        """Calculate optimal font size for text to fit in available space.

        Uses binary search to find the largest font size that allows
        the text to fit within the given dimensions.

        Args:
            text: Text to display
            available_width: Available width in pixels
            available_height: Available height in pixels
            max_font_size: Maximum font size to try (default: ui_state.font_size_large)
            min_font_size: Minimum readable font size (default: 14)
            use_mono_font: Whether to use mono font instead of sans font

        Returns:
            Tuple of (optimal_font_size, wraplength)
        """
        if not text or available_width <= 0 or available_height <= 0:
            return max_font_size or self.ui_state.font_size_large, available_width

        # Use default max size if not specified
        if max_font_size is None:
            max_font_size = self.ui_state.font_size_large

        # Get font family based on parameter
        font_family = self.get_mono_font() if use_mono_font else self.get_sans_font()

        # Calculate wrap width (90% of available for safety margin)
        wrap_width = int(available_width * 0.9)

        # First check if text fits with max size
        if self._text_fits(
            text,
            font_family,
            max_font_size,
            wrap_width,
            available_width,
            available_height,
        ):
            return max_font_size, int(available_width * UIConstants.TEXT_WRAP_RATIO)

        # Binary search for optimal size
        current_min = min_font_size
        current_max = max_font_size

        while current_max - current_min > 1:
            test_size = (current_max + current_min) // 2

            if self._text_fits(
                text,
                font_family,
                test_size,
                wrap_width,
                available_width,
                available_height,
            ):
                current_min = test_size  # Can go bigger
            else:
                current_max = test_size  # Need to go smaller

        # Return the largest size that fits
        return current_min, int(available_width * UIConstants.TEXT_WRAP_RATIO)

    def _text_fits(
        self,
        text: str,
        font_family: str,
        font_size: int,
        wrap_width: int,
        available_width: int,
        available_height: int,
    ) -> bool:
        """Check if text fits in available space with given font size.

        Args:
            text: Text to check
            font_family: Font family name
            font_size: Font size to test
            wrap_width: Width for text wrapping
            available_width: Total available width
            available_height: Total available height

        Returns:
            True if text fits, False otherwise
        """
        # Create or get cached font
        font_key = (font_family, font_size)
        if font_key not in self._font_cache:
            self._font_cache[font_key] = tkfont.Font(family=font_family, size=font_size)
        test_font = self._font_cache[font_key]

        # Calculate wrapped lines
        lines = self.wrap_text(text, test_font, wrap_width)

        # Calculate dimensions
        total_height = len(lines) * test_font.metrics("linespace")
        max_line_width = max(test_font.measure(line) for line in lines) if lines else 0

        return total_height <= available_height and max_line_width <= available_width

    @staticmethod
    def wrap_text(text: str, font: tkfont.Font, wrap_width: int) -> List[str]:
        """Wrap text to fit within given width.

        Breaks text into lines that fit within the specified width,
        respecting word boundaries where possible.

        Args:
            text: Text to wrap
            font: Font to use for measuring
            wrap_width: Maximum width in pixels

        Returns:
            List of wrapped lines
        """
        if not text:
            return []

        words = text.split()
        lines = []
        current_line = []

        for word in words:
            # Test adding this word to current line
            test_line = " ".join(current_line + [word])
            if font.measure(test_line) <= wrap_width:
                current_line.append(word)
            else:
                # Current line is full
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, add it as its own line
                    lines.append(word)
                    current_line = []

        # Add any remaining words
        if current_line:
            lines.append(" ".join(current_line))

        return lines if lines else [text]

    def apply_fonts_to_widgets(self, widgets_config: dict) -> None:
        """Apply calculated fonts to widgets.

        Args:
            widgets_config: Dictionary mapping widgets to their font configurations
                          Format: {widget: {"type": "mono"|"sans", "size": "large"|"medium"|"small", "style": "normal"|"bold"}}
        """
        for widget, config in widgets_config.items():
            font_type = config.get("type", "sans")
            size_key = config.get("size", "medium")
            style = config.get("style", "normal")

            # Get font family
            font_family = (
                self.get_mono_font() if font_type == "mono" else self.get_sans_font()
            )

            # Get size from ui_state
            size = getattr(
                self.ui_state, f"font_size_{size_key}", self.ui_state.font_size_medium
            )

            # Apply to widget
            if hasattr(widget, "config"):
                widget.config(font=(font_family, size, style))

    def clear_font_cache(self) -> None:
        """Clear the font cache.

        Should be called when theme changes or fonts need to be refreshed.
        """
        self._font_cache.clear()
