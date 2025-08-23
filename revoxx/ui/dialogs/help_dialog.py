"""Help dialog for displaying keyboard shortcuts."""

from pathlib import Path
import platform
import tkinter as tk

from ...constants import UIConstants
from .dialog_utils import setup_dialog_window


class HelpDialog:
    """Dialog for displaying keyboard shortcuts from file."""

    def __init__(self, parent: tk.Tk):
        """Initialize help dialog.

        Args:
            parent: Parent window
        """
        self.parent = parent
        self.dialog = None

    def show(self) -> None:
        """Display the help dialog."""
        # Create dialog window
        self.dialog = tk.Toplevel(self.parent)
        setup_dialog_window(self.dialog, self.parent, "Keyboard Shortcuts", 900, 800)
        self.dialog.resizable(False, False)

        # Create main container
        main_frame = tk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create frame for text widget with scrollbar
        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create text widget with monospaced font for alignment
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            padx=30,
            pady=30,
            bg=UIConstants.COLOR_BACKGROUND,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            font=("Courier New", 18),
            yscrollcommand=scrollbar.set,
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # Load help text from resources file
        try:
            resources_path = (
                Path(__file__).parent.parent.parent
                / "resources"
                / "keyboard_shortcuts.txt"
            )
            with open(resources_path, "r", encoding="utf-8") as f:
                help_text = f.read()
                # Replace Ctrl with Cmd on macOS
                if platform.system() == "Darwin":
                    help_text = help_text.replace("Ctrl", "Cmd")
        except (OSError, ValueError):
            help_text = "Help file not found. Please check the installation."

        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

        # Add OK button at the bottom
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ok_button = tk.Button(
            button_frame, text="OK", width=10, command=self.dialog.destroy
        )
        ok_button.pack()

        # Bind ESC key to close dialog
        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())

        # Focus the dialog
        self.dialog.focus_set()
