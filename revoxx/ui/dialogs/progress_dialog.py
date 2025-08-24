"""Reusable progress dialog for long operations."""

import tkinter as tk
from tkinter import ttk

from .dialog_utils import setup_dialog_window


class ProgressDialog:
    """Simple progress dialog for long operations.

    Can be used for any operation that needs to show progress to the user.
    Supports both indeterminate (spinning) and determinate (percentage) modes.
    """

    def __init__(self, parent, title="Processing", width=400, height=150):
        """Create progress dialog.

        Args:
            parent: Parent window
            title: Dialog title
            width: Dialog width in pixels
            height: Dialog height in pixels
        """
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.resizable(False, False)

        # Create widgets first
        self._create_widgets()
        # Note: center_on_parent=False to center on screen
        setup_dialog_window(
            self.dialog,
            self.parent,
            title=title,
            width=width,
            height=height,
            center_on_parent=False,
        )

    def _create_widgets(self):
        """Create dialog widgets."""
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        self.label = ttk.Label(frame, text="Processing...")
        self.label.pack(pady=(0, 10))

        self.progress = ttk.Progressbar(frame, mode="indeterminate", length=350)
        self.progress.pack(pady=(0, 10))
        self.progress.start(10)

        self.status = ttk.Label(frame, text="")
        self.status.pack()

        self.dialog.update()

    def update(self, value: int = None, text: str = "", maximum: int = None):
        """Update progress dialog.

        Args:
            value: Current progress value (for determinate mode)
            text: Status text to display
            maximum: Maximum value (switches to determinate mode if provided)
        """
        if maximum is not None and self.progress["mode"] == "indeterminate":
            # Switch to determinate mode
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=maximum)

        if value is not None and self.progress["mode"] == "determinate":
            self.progress["value"] = value

        if text:
            self.status.config(text=text)

        self.dialog.update()

    def set_label(self, text: str):
        """Set the main label text.

        Args:
            text: Label text
        """
        self.label.config(text=text)
        self.dialog.update()

    def close(self):
        """Close progress dialog."""
        try:
            if self.progress["mode"] == "indeterminate":
                self.progress.stop()
            self.dialog.destroy()
        except tk.TclError:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
