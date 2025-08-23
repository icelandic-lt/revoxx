"""Open Session Dialog for selecting existing sessions."""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import json

from .dialog_utils import setup_dialog_window


class OpenSessionDialog:
    """Dialog for selecting and opening existing Revoxx sessions.

    This dialog provides a more flexible alternative to the standard
    file dialog, allowing users to browse and preview session information
    before opening.
    """

    def __init__(self, parent: tk.Tk, default_base_dir: Optional[Path] = None):
        """Initialize the Open Session Dialog.

        Args:
            parent: Parent window
            default_base_dir: Initial directory to browse
        """
        self.parent = parent
        self.result = None
        self.current_dir = default_base_dir or Path.home()

        # Create dialog window
        self.dialog = tk.Toplevel(parent)

        # Create UI before setup to ensure widgets are ready
        self._create_widgets()

        # Load initial directory before setup
        self._load_directory(self.current_dir)

        # Use the utility function to setup the dialog window
        # This handles: title, hide, transient, size, center, show, grab
        setup_dialog_window(
            self.dialog,
            self.parent,
            title="Open Session",
            width=800,
            height=600,
            center_on_parent=True,
        )

        # Bind events after setup
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_widgets(self):
        """Create dialog widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Directory selection frame
        dir_frame = ttk.Frame(main_frame)
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Directory:").grid(row=0, column=0, padx=(0, 5))

        self.dir_var = tk.StringVar(value=str(self.current_dir))
        self.dir_entry = ttk.Entry(
            dir_frame, textvariable=self.dir_var, state="readonly"
        )
        self.dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        ttk.Button(dir_frame, text="Browse...", command=self._browse_directory).grid(
            row=0, column=2
        )

        ttk.Button(dir_frame, text="Parent", command=self._go_to_parent).grid(
            row=0, column=3, padx=(5, 0)
        )

        # Filter frame
        filter_frame = ttk.Frame(main_frame)
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))

        self.filter_var = tk.StringVar()
        self.filter_var.trace("w", lambda *args: self._apply_filter())
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        filter_entry.pack(side=tk.LEFT, padx=(0, 10))

        self.show_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            filter_frame,
            text="Show all directories (not just .revoxx)",
            variable=self.show_all_var,
            command=self._reload_directory,
        ).pack(side=tk.LEFT)

        # Session list with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Create treeview for session list
        columns = ("type", "speaker", "emotion", "utterances", "recordings")
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="tree headings", selectmode="browse"
        )

        # Configure columns
        self.tree.heading("#0", text="Name")
        self.tree.heading("type", text="Type")
        self.tree.heading("speaker", text="Speaker")
        self.tree.heading("emotion", text="Emotion")
        self.tree.heading("utterances", text="Utterances")
        self.tree.heading("recordings", text="Recordings")

        # Set column widths
        self.tree.column("#0", width=200)
        self.tree.column("type", width=80)
        self.tree.column("speaker", width=120)
        self.tree.column("emotion", width=80)
        self.tree.column("utterances", width=80)
        self.tree.column("recordings", width=80)

        # Add scrollbars
        v_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        h_scroll = ttk.Scrollbar(
            list_frame, orient=tk.HORIZONTAL, command=self.tree.xview
        )
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Grid layout for treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        # Bind double-click to open
        self.tree.bind("<Double-Button-1>", lambda e: self._on_item_double_click())
        self.tree.bind("<Return>", lambda e: self._on_item_double_click())

        # Info label
        self.info_label = ttk.Label(main_frame, text="", foreground="gray")
        self.info_label.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, sticky="ew")

        # Buttons on the right
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(button_frame, text="Open", command=self._on_open).pack(side=tk.RIGHT)

    def _browse_directory(self):
        """Browse for a different directory."""
        from tkinter import filedialog

        new_dir = filedialog.askdirectory(
            title="Select Directory", initialdir=self.current_dir, parent=self.dialog
        )

        if new_dir:
            self._load_directory(Path(new_dir))

    def _go_to_parent(self):
        """Navigate to parent directory."""
        parent = self.current_dir.parent
        if parent != self.current_dir:
            self._load_directory(parent)

    def _load_directory(self, directory: Path):
        """Load sessions from a directory.

        Args:
            directory: Directory to load
        """
        self.current_dir = directory
        self.dir_var.set(str(directory))
        self._reload_directory()

    def _reload_directory(self):
        """Reload the current directory."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.current_dir.exists():
            self.info_label.config(text="Directory does not exist")
            return

        # Find sessions and subdirectories
        sessions = []
        subdirs = []

        try:
            for path in sorted(self.current_dir.iterdir()):
                if path.is_dir():
                    if path.suffix == ".revoxx":
                        # This is a session directory
                        session_info = self._get_session_info(path)
                        if session_info:
                            sessions.append((path, session_info))
                    elif self.show_all_var.get() or self._contains_sessions(path):
                        # Regular directory that might contain sessions
                        subdirs.append(path)

        except PermissionError:
            self.info_label.config(text="Permission denied")
            return

        # Add subdirectories first
        for subdir in subdirs:
            self.tree.insert(
                "", "end", text=subdir.name, values=("Directory", "", "", "", "")
            )

        # Add sessions
        for session_path, info in sessions:
            self.tree.insert(
                "",
                "end",
                text=session_path.name,
                values=(
                    "Session",
                    info.get("speaker", ""),
                    info.get("emotion", ""),
                    info.get("utterances", ""),
                    info.get("recordings", ""),
                ),
                tags=("session",),
            )

        # Update info label
        session_count = len(sessions)
        dir_count = len(subdirs)
        info_text = f"Found {session_count} session(s)"
        if dir_count > 0:
            info_text += f" and {dir_count} subdirectory(ies)"
        self.info_label.config(text=info_text)

        # Apply filter if set
        if self.filter_var.get():
            self._apply_filter()

    def _contains_sessions(self, directory: Path) -> bool:
        """Check if a directory contains any .revoxx sessions.

        Args:
            directory: Directory to check

        Returns:
            True if directory contains sessions
        """
        try:
            for path in directory.iterdir():
                if path.is_dir() and path.suffix == ".revoxx":
                    return True
        except PermissionError:
            pass
        return False

    def _get_session_info(self, session_path: Path) -> Optional[dict]:
        """Get information about a session.

        Args:
            session_path: Path to session directory

        Returns:
            Dictionary with session information or None
        """
        info = {}

        # Read session.json
        session_file = session_path / "session.json"
        if session_file.exists():
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                    if "speaker" in data:
                        info["speaker"] = data["speaker"].get("name", "")
                        info["emotion"] = data["speaker"].get("emotion", "")
            except (json.JSONDecodeError, IOError):
                pass

        # Count utterances from script
        script_file = session_path / "script.txt"
        if script_file.exists():
            try:
                with open(script_file, "r") as f:
                    lines = f.readlines()
                    utterance_count = sum(
                        1
                        for line in lines
                        if line.strip() and not line.strip().startswith("#")
                    )
                    info["utterances"] = str(utterance_count)
            except IOError:
                pass

        # Count recordings
        recordings_dir = session_path / "recordings"
        if recordings_dir.exists():
            try:
                recording_count = sum(
                    1
                    for d in recordings_dir.iterdir()
                    if d.is_dir() and any(d.glob("take_*.flac"))
                )
                info["recordings"] = str(recording_count)
            except IOError:
                pass

        return info if info else None

    def _apply_filter(self):
        """Apply filter to the session list."""
        filter_text = self.filter_var.get().lower()

        # First, re-attach all items
        for item in self.tree.get_children(""):
            pass  # Items are already attached

        if not filter_text:
            return

        # Hide items that don't match filter
        to_hide = []
        for item in self.tree.get_children():
            item_text = self.tree.item(item, "text").lower()
            values = self.tree.item(item, "values")

            # Check if any field matches the filter
            matches = filter_text in item_text
            for value in values:
                if filter_text in str(value).lower():
                    matches = True
                    break

            if not matches:
                to_hide.append(item)

        # Detach non-matching items
        for item in to_hide:
            self.tree.detach(item)

    def _on_item_double_click(self):
        """Handle double-click on an item."""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        item_type = item["values"][0] if item["values"] else ""

        if item_type == "Directory":
            # Navigate into directory
            dir_name = item["text"]
            new_path = self.current_dir / dir_name
            self._load_directory(new_path)
        elif item_type == "Session":
            # Open the session
            self._on_open()

    def _on_open(self):
        """Handle Open button click."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Selection", "Please select a session to open.", parent=self.dialog
            )
            return

        item = self.tree.item(selection[0])
        item_type = item["values"][0] if item["values"] else ""

        if item_type != "Session":
            messagebox.showwarning(
                "Invalid Selection",
                "Please select a session directory (*.revoxx).",
                parent=self.dialog,
            )
            return

        # Get the selected session path
        session_name = item["text"]
        self.result = self.current_dir / session_name

        # Verify it's a valid session
        if not self.result.exists() or not self.result.suffix == ".revoxx":
            messagebox.showerror(
                "Invalid Session",
                "The selected directory is not a valid Revoxx session.",
                parent=self.dialog,
            )
            self.result = None
            return

        self.dialog.destroy()

    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()

    def show(self) -> Optional[Path]:
        """Show the dialog and return the selected path.

        Returns:
            Selected session path or None if cancelled
        """
        self.dialog.wait_window()
        return self.result
