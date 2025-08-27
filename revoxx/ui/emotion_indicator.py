"""Emotion level indicator widget for displaying emotional intensity."""

import tkinter as tk
from typing import Optional
from ..constants import UIConstants
from .font_manager import FontManager


class EmotionLevelIndicator(tk.Frame):
    """Visual indicator for emotion intensity levels.

    Displays emotion level as a series of square blocks where the active level
    is highlighted and shows the number. Active block is larger and more prominent.
    """

    def __init__(
        self,
        parent: tk.Widget,
        font_manager: FontManager,
        max_level: int = 5,
        bg_color: Optional[str] = None,
    ):
        """Initialize the emotion level indicator.

        Args:
            parent: Parent tkinter widget
            font_manager: Font manager for font handling
            max_level: Maximum number of emotion levels (default: 5)
            bg_color: Background color (default: uses theme)
        """
        super().__init__(
            parent,
            bg=bg_color or UIConstants.COLOR_BACKGROUND_SECONDARY,
            highlightthickness=0,
        )

        self.max_level = max_level
        self.current_level = 0
        self.blocks = []
        self.block_labels = []
        self.bg_color = bg_color or UIConstants.COLOR_BACKGROUND_SECONDARY
        self.font_manager = font_manager

        # Dimensions
        self.inactive_size = 24  # Square size for inactive blocks
        self.active_size = 32  # Larger size for active block
        self.active_width = 40  # Wider width for active block to accommodate number

        # Get mono font from FontManager
        self.mono_font = self.font_manager.get_mono_font()

        # Create label with same font as other elements in info bar
        self.label = tk.Label(
            self,
            text="Emotion:",
            fg=UIConstants.COLOR_TEXT_SECONDARY,
            bg=self.bg_color,
            font=(self.mono_font, 24),  # Same as status bar
        )
        self.label.pack(side=tk.LEFT, padx=(0, 10))

        # Create container for blocks
        self.container = tk.Frame(self, bg=self.bg_color)
        self.container.pack(side=tk.LEFT, padx=5)

        # Create blocks
        self._create_blocks()

    def _create_blocks(self) -> None:
        """Create the visual blocks for each level."""
        # Clear existing blocks and labels
        for block in self.blocks:
            block.destroy()
        self.blocks.clear()
        self.block_labels.clear()

        # Create new blocks (1-indexed for display)
        for i in range(self.max_level):
            level_num = i + 1  # Display numbers start at 1

            # Get gradient color for this level
            gradient_color = self._get_block_color(level_index=i)

            # Create frame for block
            block_frame = tk.Frame(
                self.container,
                width=self.inactive_size,
                height=self.inactive_size,
                bg=gradient_color,
                highlightthickness=0,
            )
            block_frame.pack(side=tk.LEFT, padx=3)
            block_frame.pack_propagate(False)

            # Create label for number (initially hidden)
            label = tk.Label(
                block_frame,
                text=str(level_num),
                fg="white",
                bg=gradient_color,
                font=(self.mono_font, 12, "bold"),
            )
            # Don't pack the label yet - will be shown only for active block

            self.blocks.append(block_frame)
            self.block_labels.append(label)

    @staticmethod
    def _parse_hex_color(hex_color: str, fallback: tuple = (0, 188, 212)) -> tuple:
        """Parse hex color string to RGB tuple.

        Args:
            hex_color: Hex color string (#RGB or #RRGGBB)
            fallback: Fallback RGB values if parsing fails

        Returns:
            Tuple of (r, g, b) values
        """
        if not hex_color.startswith("#"):
            return fallback

        hex_str = hex_color[1:]
        if len(hex_str) == 3:
            # Convert #RGB to #RRGGBB
            hex_str = "".join([c * 2 for c in hex_str])

        try:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return r, g, b
        except (ValueError, IndexError):
            return fallback

    @staticmethod
    def _blend_with_opacity(fg_color: tuple, bg_color: tuple, opacity: float) -> str:
        """Blend foreground and background colors with given opacity.

        Args:
            fg_color: Foreground RGB tuple
            bg_color: Background RGB tuple
            opacity: Opacity value (0.0 to 1.0)

        Returns:
            Blended color as hex string
        """
        r = int(bg_color[0] + (fg_color[0] - bg_color[0]) * opacity)
        g = int(bg_color[1] + (fg_color[1] - bg_color[1]) * opacity)
        b = int(bg_color[2] + (fg_color[2] - bg_color[2]) * opacity)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _get_block_color(self, level_index: int = None, active: bool = False) -> str:
        """Get color for a block with appropriate opacity.

        Args:
            level_index: 0-based index for gradient calculation (None for active)
            active: True for active block (90% opacity), False for gradient

        Returns:
            Hex color string with appropriate opacity
        """
        # Parse colors - get current theme colors each time
        accent_rgb = self._parse_hex_color(UIConstants.COLOR_ACCENT)
        bg_rgb = self._parse_hex_color(
            UIConstants.COLOR_BACKGROUND_SECONDARY, (32, 32, 32)
        )

        if active:
            # Active block: 90% opacity
            opacity = 0.90
        else:
            # Gradient for inactive blocks: from 15% to 75% opacity
            min_opacity = 0.15
            max_opacity = 0.75
            opacity = min_opacity + (max_opacity - min_opacity) * (
                level_index / max(self.max_level - 1, 1)
            )

        return self._blend_with_opacity(accent_rgb, bg_rgb, opacity)

    def set_level(self, level: int) -> None:
        """Set the current emotion level.

        Args:
            level: Emotion level to display (1 to max_level, 0 for none)
        """
        # Clamp level to valid range
        level = max(0, min(level, self.max_level))
        self.current_level = level

        # Update all blocks
        for i in range(self.max_level):
            block = self.blocks[i]
            label = self.block_labels[i]

            if (
                i == level - 1 and level > 0
            ):  # This is the active block (level is 1-indexed)
                # Make block larger and wider
                active_color = self._get_block_color(active=True)
                block.configure(
                    width=self.active_width,
                    height=self.active_size,
                    bg=active_color,
                )
                # Show and center the number
                label.configure(
                    fg="white",
                    bg=active_color,
                    font=(self.mono_font, 18, "bold"),
                )
                label.place(relx=0.5, rely=0.5, anchor="center")
            else:
                # Inactive block - smaller, square, with gradient color
                gradient_color = self._get_block_color(level_index=i)
                block.configure(
                    width=self.inactive_size,
                    height=self.inactive_size,
                    bg=gradient_color,
                )
                # Update label background but keep hidden
                label.configure(bg=gradient_color)
                # Hide the number
                label.place_forget()

    def set_max_level(self, max_level: int) -> None:
        """Update the maximum number of levels.

        Args:
            max_level: New maximum level
        """
        if max_level != self.max_level:
            self.max_level = max_level
            self._create_blocks()
            self.set_level(self.current_level)

    def refresh_colors(self) -> None:
        """Refresh colors after theme change."""
        # Update background - get fresh colors from UIConstants
        bg_color = UIConstants.COLOR_BACKGROUND_SECONDARY
        self.bg_color = bg_color
        self.configure(bg=bg_color)
        self.container.configure(bg=bg_color)
        self.label.configure(fg=UIConstants.COLOR_TEXT_SECONDARY, bg=bg_color)

        # Recreate blocks with new theme colors
        # This ensures gradient colors are recalculated with new theme
        self._create_blocks()

        # Restore current level with new colors
        self.set_level(self.current_level)

    def hide(self) -> None:
        """Hide the emotion indicator."""
        self.pack_forget()

    def show(self) -> None:
        """Show the emotion indicator."""
        self.pack(side=tk.LEFT, expand=True)
