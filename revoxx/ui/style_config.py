"""Style configuration for UI elements."""

import tkinter as tk
from typing import Dict, Any
from ..constants import UIConstants


class CyanStyle:
    """Style configuration for tkinter widgets."""

    @staticmethod
    def apply_window_style(window: tk.Toplevel) -> None:
        """Apply dark theme to a window.

        Args:
            window: The window to style
        """
        window.configure(bg=UIConstants.COLOR_BACKGROUND)
        try:
            # Try to set window transparency for visual enhancement
            window.attributes("-alpha", 0.98)
        except Exception:
            pass

    @staticmethod
    def get_button_style() -> Dict[str, Any]:
        """Get button style configuration.

        Returns:
            Dictionary of button style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND_SECONDARY,
            "fg": UIConstants.COLOR_ACCENT,
            "activebackground": UIConstants.COLOR_ACCENT,
            "activeforeground": UIConstants.COLOR_BACKGROUND,
            "bd": 1,
            "highlightbackground": UIConstants.COLOR_BORDER,
            "highlightthickness": 1,
            "font": (UIConstants.FONT_FAMILY_MONO[0], 10),
            "cursor": "hand2",
            "padx": 15,
            "pady": 8,
        }

    @staticmethod
    def get_label_style() -> Dict[str, Any]:
        """Get label style configuration.

        Returns:
            Dictionary of label style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND,
            "fg": UIConstants.COLOR_TEXT_NORMAL,
            "font": (UIConstants.FONT_FAMILY_SANS[0], 11),
        }

    @staticmethod
    def get_entry_style() -> Dict[str, Any]:
        """Get entry field style configuration.

        Returns:
            Dictionary of entry style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND_TERTIARY,
            "fg": UIConstants.COLOR_TEXT_NORMAL,
            "insertbackground": UIConstants.COLOR_ACCENT,
            "selectbackground": UIConstants.COLOR_ACCENT,
            "selectforeground": UIConstants.COLOR_BACKGROUND,
            "highlightbackground": UIConstants.COLOR_BORDER,
            "highlightcolor": UIConstants.COLOR_ACCENT,
            "highlightthickness": 1,
            "bd": 1,
            "font": (UIConstants.FONT_FAMILY_MONO[0], 11),
        }

    @staticmethod
    def get_listbox_style() -> Dict[str, Any]:
        """Get listbox style configuration.

        Returns:
            Dictionary of listbox style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND_TERTIARY,
            "fg": UIConstants.COLOR_TEXT_NORMAL,
            "selectbackground": UIConstants.COLOR_ACCENT,
            "selectforeground": UIConstants.COLOR_BACKGROUND,
            "highlightbackground": UIConstants.COLOR_BORDER,
            "highlightcolor": UIConstants.COLOR_ACCENT,
            "highlightthickness": 1,
            "bd": 1,
            "font": (UIConstants.FONT_FAMILY_MONO[0], 10),
            "activestyle": "none",
        }

    @staticmethod
    def get_frame_style() -> Dict[str, Any]:
        """Get frame style configuration.

        Returns:
            Dictionary of frame style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND,
            "highlightbackground": UIConstants.COLOR_BORDER,
            "highlightthickness": 0,
        }

    @staticmethod
    def get_checkbutton_style() -> Dict[str, Any]:
        """Get checkbutton style configuration.

        Returns:
            Dictionary of checkbutton style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND,
            "fg": UIConstants.COLOR_TEXT_NORMAL,
            "activebackground": UIConstants.COLOR_BACKGROUND,
            "activeforeground": UIConstants.COLOR_ACCENT,
            "selectcolor": UIConstants.COLOR_BACKGROUND_SECONDARY,
            "highlightbackground": UIConstants.COLOR_BACKGROUND,
            "highlightcolor": UIConstants.COLOR_ACCENT,
            "font": (UIConstants.FONT_FAMILY_SANS[0], 11),
        }

    @staticmethod
    def get_radiobutton_style() -> Dict[str, Any]:
        """Get radiobutton style configuration.

        Returns:
            Dictionary of radiobutton style properties
        """
        return {
            "bg": UIConstants.COLOR_BACKGROUND,
            "fg": UIConstants.COLOR_TEXT_NORMAL,
            "activebackground": UIConstants.COLOR_BACKGROUND,
            "activeforeground": UIConstants.COLOR_ACCENT,
            "selectcolor": UIConstants.COLOR_BACKGROUND_SECONDARY,
            "highlightbackground": UIConstants.COLOR_BACKGROUND,
            "highlightcolor": UIConstants.COLOR_ACCENT,
            "font": (UIConstants.FONT_FAMILY_SANS[0], 11),
        }

    @staticmethod
    def create_styled_button(
        parent: tk.Widget, text: str, command=None, **kwargs
    ) -> tk.Button:
        """Create a styled button.

        Args:
            parent: Parent widget
            text: Button text
            command: Button command
            **kwargs: Additional button options

        Returns:
            Styled button widget
        """
        style = CyanStyle.get_button_style()
        style.update(kwargs)

        button = tk.Button(parent, text=text.upper(), command=command, **style)

        # Add hover effects
        def on_enter(_):
            button.config(bg=UIConstants.COLOR_ACCENT, fg=UIConstants.COLOR_BACKGROUND)

        def on_leave(_):
            button.config(
                bg=UIConstants.COLOR_BACKGROUND_SECONDARY, fg=UIConstants.COLOR_ACCENT
            )

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

        return button

    @staticmethod
    def create_styled_label(parent: tk.Widget, text: str = "", **kwargs) -> tk.Label:
        """Create a styled label.

        Args:
            parent: Parent widget
            text: Label text
            **kwargs: Additional label options

        Returns:
            Styled label widget
        """
        style = CyanStyle.get_label_style()
        style.update(kwargs)
        return tk.Label(parent, text=text, **style)

    @staticmethod
    def create_styled_entry(parent: tk.Widget, **kwargs) -> tk.Entry:
        """Create a styled entry field.

        Args:
            parent: Parent widget
            **kwargs: Additional entry options

        Returns:
            Styled entry widget
        """
        style = CyanStyle.get_entry_style()
        style.update(kwargs)
        return tk.Entry(parent, **style)

    @staticmethod
    def create_styled_frame(parent: tk.Widget, **kwargs) -> tk.Frame:
        """Create a styled frame.

        Args:
            parent: Parent widget
            **kwargs: Additional frame options

        Returns:
            Styled frame widget
        """
        style = CyanStyle.get_frame_style()
        style.update(kwargs)
        return tk.Frame(parent, **style)
