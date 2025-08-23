"""Dialog for creating datasets from Revoxx sessions."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from ...dataset import DatasetExporter
from ...utils.settings_manager import SettingsManager
from ...session.inspector import SessionInspector
from .progress_dialog import ProgressDialog
from .dialog_utils import setup_dialog_window


class DatasetDialog:
    """Dialog for selecting sessions and creating a dataset."""

    # Dialog dimensions
    DIALOG_WIDTH = 800
    DIALOG_HEIGHT = 600

    # Summary dialog dimensions
    SUMMARY_WIDTH = 600
    SUMMARY_HEIGHT = 400

    # Column widths for the session tree
    COLUMN_WIDTHS = {
        "#0": (250, 150),  # (width, minwidth)
        "Speaker": (120, 80),
        "Emotion": (80, 60),
        "Recordings": (80, 60),
        "Created": (130, 100),
    }

    # Padding values
    PADDING_STANDARD = 10
    PADDING_SMALL = 5
    PADDING_FRAME = "10"

    # Entry field widths
    ENTRY_WIDTH_STANDARD = 40

    def __init__(self, parent, base_dir: Path, settings_manager: SettingsManager):
        """Initialize dataset creation dialog.

        Args:
            parent: Parent window
            base_dir: Base directory containing sessions
            settings_manager: Shared SettingsManager instance
        """
        self.parent = parent
        self.settings_manager = settings_manager
        self.result = None

        # Use provided base_dir
        self.base_dir = Path(base_dir)

        # Get last export directory from settings
        last_export = getattr(self.settings_manager.settings, "last_export_dir", None)
        if last_export:
            self.export_dir = Path(last_export)
        else:
            self.export_dir = Path.home() / "revoxx_datasets"

        self._create_dialog()
        self._scan_sessions()

    def _create_dialog(self):
        """Create the dialog window and widgets."""
        self.dialog = tk.Toplevel(self.parent)

        # Set resizable before setup
        self.dialog.resizable(True, True)

        # Main container
        main_frame = ttk.Frame(self.dialog, padding=self.PADDING_FRAME)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Base directory selection
        dir_frame = ttk.LabelFrame(
            main_frame, text="Session Directory", padding=str(self.PADDING_SMALL)
        )
        dir_frame.pack(fill=tk.X, pady=(0, self.PADDING_STANDARD))

        self.base_dir_var = tk.StringVar(value=str(self.base_dir))
        ttk.Entry(dir_frame, textvariable=self.base_dir_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, self.PADDING_SMALL)
        )
        ttk.Button(dir_frame, text="Browse...", command=self._browse_base_dir).pack(
            side=tk.RIGHT
        )

        # Session selection
        session_frame = ttk.LabelFrame(
            main_frame, text="Select Sessions", padding=str(self.PADDING_SMALL)
        )
        session_frame.pack(fill=tk.BOTH, expand=True, pady=(0, self.PADDING_STANDARD))

        # Create treeview for sessions
        columns = ("Speaker", "Emotion", "Recordings", "Created")
        self.session_tree = ttk.Treeview(
            session_frame, columns=columns, show="tree headings", selectmode="extended"
        )

        # Configure styles for better appearance
        self.style = ttk.Style()
        self.style.configure(
            "Treeview.Heading",
            relief="flat",
            background="#d9d9d9",
            foreground="black",
            borderwidth=1,
        )
        self.style.map("Treeview.Heading", background=[("active", "#c0c0c0")])

        # Configure columns with sorting - default to created date ascending
        self.sort_reverse = False
        self.sort_column = "created"
        self.session_tree.heading(
            "#0", text="  Session ◦", command=lambda: self._sort_by("name")
        )
        self.session_tree.heading(
            "Speaker", text="  Speaker ◦", command=lambda: self._sort_by("speaker")
        )
        self.session_tree.heading(
            "Emotion", text="  Emotion ◦", command=lambda: self._sort_by("emotion")
        )
        self.session_tree.heading(
            "Recordings",
            text="  Recordings ◦",
            command=lambda: self._sort_by("recordings"),
        )
        self.session_tree.heading(
            "Created", text="[ Created ▲ ]", command=lambda: self._sort_by("created")
        )

        self.session_tree.column(
            "#0",
            width=self.COLUMN_WIDTHS["#0"][0],
            minwidth=self.COLUMN_WIDTHS["#0"][1],
        )
        self.session_tree.column(
            "Speaker",
            width=self.COLUMN_WIDTHS["Speaker"][0],
            minwidth=self.COLUMN_WIDTHS["Speaker"][1],
        )
        self.session_tree.column(
            "Emotion",
            width=self.COLUMN_WIDTHS["Emotion"][0],
            minwidth=self.COLUMN_WIDTHS["Emotion"][1],
        )
        self.session_tree.column(
            "Recordings",
            width=self.COLUMN_WIDTHS["Recordings"][0],
            minwidth=self.COLUMN_WIDTHS["Recordings"][1],
        )
        self.session_tree.column(
            "Created",
            width=self.COLUMN_WIDTHS["Created"][0],
            minwidth=self.COLUMN_WIDTHS["Created"][1],
        )

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            session_frame, orient=tk.VERTICAL, command=self.session_tree.yview
        )
        self.session_tree.configure(yscrollcommand=scrollbar.set)

        self.session_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Output settings
        output_frame = ttk.LabelFrame(
            main_frame, text="Output Settings", padding=str(self.PADDING_SMALL)
        )
        output_frame.pack(fill=tk.X, pady=(0, self.PADDING_STANDARD))

        # Output directory
        ttk.Label(output_frame, text="Output Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, self.PADDING_SMALL), pady=2
        )
        self.output_dir_var = tk.StringVar(value=str(self.export_dir))
        ttk.Entry(
            output_frame,
            textvariable=self.output_dir_var,
            width=self.ENTRY_WIDTH_STANDARD,
        ).grid(row=0, column=1, sticky=tk.EW, padx=(0, self.PADDING_SMALL), pady=2)
        ttk.Button(
            output_frame, text="Browse...", command=self._browse_output_dir
        ).grid(row=0, column=2, pady=2)

        # Dataset name
        ttk.Label(output_frame, text="Dataset Name:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, self.PADDING_SMALL), pady=2
        )
        self.dataset_name_var = tk.StringVar()
        ttk.Entry(
            output_frame,
            textvariable=self.dataset_name_var,
            width=self.ENTRY_WIDTH_STANDARD,
        ).grid(row=1, column=1, sticky=tk.EW, padx=(0, self.PADDING_SMALL), pady=2)
        ttk.Label(output_frame, text="(leave empty for auto)").grid(
            row=1, column=2, sticky=tk.W, pady=2
        )

        # Format selection
        ttk.Label(output_frame, text="Audio Format:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, self.PADDING_SMALL), pady=2
        )
        self.format_var = tk.StringVar(
            value=getattr(self.settings_manager.settings, "export_format", "flac")
        )
        format_frame = ttk.Frame(output_frame)
        format_frame.grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(
            format_frame, text="FLAC", variable=self.format_var, value="flac"
        ).pack(side=tk.LEFT, padx=(0, self.PADDING_STANDARD))
        ttk.Radiobutton(
            format_frame, text="WAV", variable=self.format_var, value="wav"
        ).pack(side=tk.LEFT)

        # Emotion level export option
        ttk.Label(output_frame, text="Export Options:").grid(
            row=3, column=0, sticky=tk.W, padx=(0, self.PADDING_SMALL), pady=2
        )
        self.include_intensity_var = tk.BooleanVar(
            value=getattr(
                self.settings_manager.settings, "export_include_intensity", True
            )
        )
        ttk.Checkbutton(
            output_frame,
            text="Include intensity levels in index.tsv",
            variable=self.include_intensity_var,
        ).grid(row=3, column=1, sticky=tk.W, pady=2)

        output_frame.columnconfigure(1, weight=1)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(self.PADDING_STANDARD, 0))

        create_btn = ttk.Button(
            button_frame, text="Create Dataset", command=self._create_dataset
        )
        create_btn.pack(side=tk.RIGHT, padx=(self.PADDING_SMALL, 0))

        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(
            side=tk.RIGHT, padx=(0, self.PADDING_SMALL)
        )

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(button_frame, textvariable=self.status_var).pack(
            side=tk.LEFT, padx=self.PADDING_SMALL
        )

        setup_dialog_window(
            self.dialog,
            self.parent,
            title="Create Dataset",
            width=self.DIALOG_WIDTH,
            height=self.DIALOG_HEIGHT,
            center_on_parent=False,  # Center on screen for dataset dialog
        )

    def _scan_sessions(self):
        """Scan base directory for Revoxx sessions."""
        self.session_tree.delete(*self.session_tree.get_children())

        if not self.base_dir.exists():
            self.status_var.set(f"Directory not found: {self.base_dir}")
            return

        # Use SessionInspector to find and analyze sessions
        session_infos = SessionInspector.find_sessions(self.base_dir)

        self.sessions_data = []  # Store for sorting
        for info in session_infos:
            # Format creation date for display
            created_display = self._format_date(info.created_at)

            self.sessions_data.append(
                {
                    "name": info.name,
                    "speaker": info.speaker,
                    "emotion": info.emotion,
                    "recordings": info.recording_files,
                    "created": info.created_at,  # Store full ISO string for sorting
                    "created_display": created_display,  # Store formatted string for display
                    "path": str(info.path),
                    "session_info": info,  # Store full info for later use
                }
            )

        # Sort by creation date (ascending) by default
        if self.sessions_data:
            self.sessions_data.sort(key=lambda x: x["created"] or "")

        self._populate_tree()
        self._update_sort_indicators()
        self.status_var.set(f"Found {len(self.sessions_data)} sessions")

    def _populate_tree(self):
        """Populate tree with session data."""
        self.session_tree.delete(*self.session_tree.get_children())

        # Configure tags for alternating row colors
        self.session_tree.tag_configure("evenrow", background="#f5f5f5")
        self.session_tree.tag_configure("oddrow", background="white")

        for i, session in enumerate(self.sessions_data):
            # Alternate row colors for better readability
            tags = [session["path"]]
            if i % 2 == 0:
                tags.append("evenrow")
            else:
                tags.append("oddrow")

            self.session_tree.insert(
                "",
                "end",
                text=session["name"],
                values=(
                    session["speaker"],
                    session["emotion"],
                    session["recordings"],
                    session["created_display"],
                ),
                tags=tuple(tags),
            )

    def _sort_by(self, column: str):
        """Sort tree by specified column."""
        # Toggle sort direction if same column, otherwise reset
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        # Sort data
        if column == "recordings":
            # Sort numerically for recordings
            self.sessions_data.sort(key=lambda x: x[column], reverse=self.sort_reverse)
        elif column == "created":
            # Sort by ISO date string (works because ISO format is sortable)
            self.sessions_data.sort(
                key=lambda x: x[column] or "", reverse=self.sort_reverse
            )
        else:
            # Sort alphabetically for text columns
            self.sessions_data.sort(
                key=lambda x: x[column].lower(), reverse=self.sort_reverse
            )

        # Update column headers with arrows
        self._update_sort_indicators()
        self._populate_tree()

    def _update_sort_indicators(self):
        """Update column headers with sort indicators."""
        # Define arrow symbols - more prominent for active column
        arrow_up = " ▲"
        arrow_down = " ▼"
        inactive_arrow = " ◦"

        # Column display names
        columns = {
            "name": ("Session", "#0"),
            "speaker": ("Speaker", "Speaker"),
            "emotion": ("Emotion", "Emotion"),
            "recordings": ("Recordings", "Recordings"),
            "created": ("Created", "Created"),
        }

        # Update all column headers
        for col_key, (display_name, tree_col) in columns.items():
            if col_key == self.sort_column:
                # Active column with prominent arrow and brackets
                arrow = arrow_down if self.sort_reverse else arrow_up
                text = f"[ {display_name}{arrow} ]"
            else:
                # Inactive columns with subtle indicator
                text = f"  {display_name}{inactive_arrow}"

            # Update the heading
            self.session_tree.heading(tree_col, text=text)

    @staticmethod
    def _format_date(created_at: str) -> str:
        """Format ISO date string for display."""
        if not created_at:
            return "Unknown"

        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            return created_at[:16] if len(created_at) >= 16 else created_at

    def _browse_base_dir(self):
        """Browse for base directory containing sessions."""
        dir_path = filedialog.askdirectory(
            parent=self.dialog,
            title="Select Sessions Directory",
            initialdir=self.base_dir,
        )
        if dir_path:
            self.base_dir = Path(dir_path)
            self.base_dir_var.set(str(self.base_dir))
            self._scan_sessions()

            # Save to settings
            self.settings_manager.update_setting(
                "base_sessions_dir", str(self.base_dir)
            )

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = filedialog.askdirectory(
            parent=self.dialog,
            title="Select Output Directory",
            initialdir=self.output_dir_var.get(),
        )
        if dir_path:
            self.output_dir_var.set(dir_path)

    @staticmethod
    def _validate_sessions(session_paths: List[Path]) -> Dict:
        """Validate sessions for completeness using SessionInspector.

        Returns:
            Dict with validation results
        """
        validation_result = SessionInspector.validate_sessions(session_paths)

        return {
            "valid_sessions": validation_result.valid_sessions,
            "incomplete": validation_result.incomplete_sessions,
            "empty": validation_result.empty_sessions,
        }

    def _get_selected_session_paths(self) -> Optional[List[Path]]:
        """Get paths of selected sessions from tree.

        Returns:
            List of session paths or None if none selected
        """
        selected = self.session_tree.selection()
        if not selected:
            messagebox.showwarning(
                "No Selection",
                "Please select at least one session to export.",
                parent=self.dialog,
            )
            return None

        session_paths = []
        for item in selected:
            tags = self.session_tree.item(item, "tags")
            if tags:
                session_paths.append(Path(tags[0]))
        return session_paths

    def _handle_session_validation(self, validation_info: Dict) -> Optional[List[Path]]:
        """Handle validation results and show appropriate warnings.

        Args:
            validation_info: Validation results dictionary

        Returns:
            List of valid session paths or None if cancelled
        """
        # Handle empty sessions
        if validation_info["empty"]:
            empty_names = [s["name"] for s in validation_info["empty"]]
            messagebox.showwarning(
                "Empty Sessions",
                f"The following sessions have no recordings and will be ignored:\n\n"
                f"{', '.join(empty_names)}",
                parent=self.dialog,
            )

        # Handle incomplete sessions
        if validation_info["incomplete"]:
            incomplete_msg = "The following sessions have missing recordings:\n\n"
            for session in validation_info["incomplete"]:
                incomplete_msg += f"• {session['name']}: {session['recorded']}/{session['total']} recordings ({session['missing']} missing)\n"
            incomplete_msg += "\nDo you want to continue?"

            if not messagebox.askyesno(
                "Incomplete Sessions", incomplete_msg, parent=self.dialog
            ):
                return None

        # Check if we have any valid sessions left
        valid_sessions = validation_info["valid_sessions"]
        if not valid_sessions:
            messagebox.showerror(
                "No Valid Sessions",
                "No sessions with recordings were found.",
                parent=self.dialog,
            )
            return None

        return valid_sessions

    def _prepare_output_directory(self, dataset_name: Optional[str]) -> Optional[Path]:
        """Prepare and validate output directory.

        Args:
            dataset_name: Optional dataset name

        Returns:
            Output directory path or None if error/cancelled
        """
        output_dir = Path(self.output_dir_var.get())

        # Create directory if needed
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not create output directory: {e}",
                    parent=self.dialog,
                )
                return None

        # Check if dataset already exists
        if dataset_name:
            dataset_path = output_dir / dataset_name
            if dataset_path.exists():
                if not messagebox.askyesno(
                    "Dataset Exists",
                    f"Dataset '{dataset_name}' already exists. Overwrite?",
                    parent=self.dialog,
                ):
                    return None

        return output_dir

    def _create_dataset(self):
        """Create dataset from selected sessions."""
        # Get selected session paths
        session_paths = self._get_selected_session_paths()
        if not session_paths:
            return

        # Validate sessions
        validation_info = self._validate_sessions(session_paths)
        valid_sessions = self._handle_session_validation(validation_info)
        if not valid_sessions:
            return

        # Get dataset name and prepare output directory
        dataset_name = self.dataset_name_var.get().strip() or None
        output_dir = self._prepare_output_directory(dataset_name)
        if not output_dir:
            return

        # Save settings and run export
        self._save_export_settings(output_dir)
        self._run_export(valid_sessions, output_dir, dataset_name)

    def _save_export_settings(self, output_dir: Path) -> None:
        """Save export settings for next time.

        Args:
            output_dir: Output directory path
        """
        self.settings_manager.update_setting("last_export_dir", str(output_dir))
        self.settings_manager.update_setting("export_format", self.format_var.get())
        self.settings_manager.update_setting(
            "export_include_intensity", self.include_intensity_var.get()
        )

    def _run_export(
        self, session_paths: List[Path], output_dir: Path, dataset_name: Optional[str]
    ) -> None:
        """Run the actual export with progress dialog.

        Args:
            session_paths: Paths to export
            output_dir: Output directory
            dataset_name: Optional dataset name
        """
        progress_dialog = ProgressDialog(self.dialog, "Creating Dataset")

        try:
            # Create exporter
            exporter = DatasetExporter(
                output_dir=output_dir,
                audio_format=self.format_var.get(),
                include_intensity=self.include_intensity_var.get(),
            )

            # Export sessions
            def progress_callback(count):
                progress_dialog.update(count, f"Processing utterance {count}")

            dataset_paths, statistics = exporter.export_sessions(
                session_paths,
                dataset_name=dataset_name,
                progress_callback=progress_callback,
            )

            progress_dialog.close()

            # Show success message
            self._show_export_summary(dataset_paths, statistics)
            self.result = dataset_paths
            self.dialog.destroy()

        except Exception as e:
            progress_dialog.close()
            messagebox.showerror(
                "Export Error", f"Failed to create dataset: {e}", parent=self.dialog
            )

    def _show_export_summary(self, dataset_paths: List[Path], statistics: Dict):
        """Show export summary in a scrollable dialog."""
        summary_dialog = tk.Toplevel(self.dialog)

        summary_dialog.resizable(True, True)

        # Main frame
        main_frame = ttk.Frame(summary_dialog, padding=self.PADDING_FRAME)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        if statistics["datasets_created"] == 1:
            title_text = "Dataset created successfully!"
        else:
            title_text = (
                f"{statistics['datasets_created']} datasets created successfully!"
            )

        title_label = ttk.Label(main_frame, text=title_text, font=("", 12, "bold"))
        title_label.pack(pady=(0, self.PADDING_STANDARD))

        # Create scrollable text widget for summary
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Text widget with scrollbar
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            width=60,
            height=15,
            background="#f9f9f9",
            relief=tk.FLAT,
            borderwidth=1,
        )

        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=text_widget.yview
        )
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build summary text
        summary = f"Sessions processed: {statistics['sessions_processed']}\n"
        summary += f"Total utterances: {statistics['total_utterances']}\n"
        summary += "-" * 50 + "\n"

        # Show details for each speaker
        for speaker_info in statistics["speakers"]:
            summary += f"\nSpeaker: {speaker_info['name']}\n"
            summary += f"  Output: {speaker_info['output_path']}\n"
            summary += f"  Emotions: {', '.join(speaker_info['emotions'])}\n"
            if speaker_info.get("file_counts"):
                summary += "  Files per emotion:\n"
                for emotion, count in speaker_info["file_counts"].items():
                    summary += f"    • {emotion}: {count} files\n"

        if statistics.get("missing_recordings"):
            summary += "\n" + "-" * 50 + "\n"
            summary += f"⚠ Warning: {statistics['missing_recordings']} recordings were missing\n"

        # Insert text and make read-only
        text_widget.insert("1.0", summary)
        text_widget.configure(state="disabled")

        # OK button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(self.PADDING_STANDARD, 0))

        ok_button = ttk.Button(button_frame, text="OK", command=summary_dialog.destroy)
        ok_button.pack(side=tk.RIGHT)

        setup_dialog_window(
            summary_dialog,
            self.dialog,  # Parent is the dataset dialog, not main window
            title="Export Summary",
            width=self.SUMMARY_WIDTH,
            height=self.SUMMARY_HEIGHT,
            center_on_parent=True,  # Center on parent (dataset dialog)
        )

        # Focus on OK button
        ok_button.focus_set()

        # Wait for dialog to close
        summary_dialog.wait_window()

    def _cancel(self):
        """Cancel dialog."""
        self.dialog.destroy()

    def show(self) -> Optional[Path]:
        """Show dialog and return result.

        Returns:
            Path to created dataset or None if cancelled
        """
        # Dialog is already shown by _create_dialog after proper positioning
        self.dialog.wait_window()
        return self.result
