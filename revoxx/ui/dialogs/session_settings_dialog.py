"""Session Settings Dialog for viewing current session configuration."""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict

from ...session import Session
from ...session.script_parser import FestivalScriptParser
from .dialog_utils import setup_dialog_window


class SessionSettingsDialog:
    """Dialog for viewing current session settings (read-only)."""

    def __init__(self, parent: tk.Tk, session: Session):
        """Initialize the session settings dialog.

        Args:
            parent: Parent window
            session: Current session to display
        """
        self.parent = parent
        self.session = session

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.resizable(False, False)

        # Create UI
        self._create_widgets()
        setup_dialog_window(
            self.dialog,
            self.parent,
            title="Session Settings",
            width=1200,
            height=600,
            center_on_parent=True,
        )

        # Setup keyboard bindings
        self.dialog.bind("<Escape>", lambda e: self._on_close())
        self.dialog.bind("<Return>", lambda e: self._on_close())

    def _create_widgets(self):
        """Create and layout dialog widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create two-column layout
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 20))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure grid weights for stretching
        left_frame.columnconfigure(1, weight=1)

        # Create sections
        row = self._create_session_info_section(left_frame, 0)
        row = self._create_speaker_section(left_frame, row)
        row = self._create_audio_section(left_frame, row)
        self._create_script_section(left_frame, row, right_frame)

        # Close button
        self._create_close_button(right_frame)

    def _create_session_info_section(self, parent: ttk.Frame, row: int) -> int:
        """Create session information section."""
        row = self._add_section_header(parent, row, "Session Information")
        row = self._add_field(
            parent,
            row,
            "Session Name:",
            self.session.name if self.session.name else "Unnamed Session",
        )
        row = self._add_field(
            parent,
            row,
            "Session Directory:",
            self.session.session_dir.name if self.session.session_dir else "N/A",
        )

        if self.session.created_at:
            created_str = self.session.created_at.strftime("%Y-%m-%d %H:%M")
            row = self._add_field(parent, row, "Created:", created_str)

        return self._add_separator(parent, row)

    def _create_speaker_section(self, parent: ttk.Frame, row: int) -> int:
        """Create speaker information section."""
        if not self.session.speaker:
            return row

        row = self._add_section_header(parent, row, "Speaker Information")
        row = self._add_field(parent, row, "Speaker Name:", self.session.speaker.name)
        row = self._add_field(
            parent,
            row,
            "Gender:",
            self._format_gender(self.session.speaker.gender),
        )
        row = self._add_field(
            parent, row, "Emotion:", self.session.speaker.emotion.title()
        )
        return self._add_separator(parent, row)

    def _create_audio_section(self, parent: ttk.Frame, row: int) -> int:
        """Create audio configuration section."""
        if not self.session.audio_config:
            return row

        row = self._add_section_header(parent, row, "Audio Configuration")

        # Format device name
        device_name = self.session.audio_config.input_device
        device_display = (
            "System Default" if device_name in ("default", None) else device_name
        )

        row = self._add_field(parent, row, "Input Device:", device_display)
        row = self._add_field(
            parent,
            row,
            "Sample Rate:",
            f"{self.session.audio_config.sample_rate:,} Hz",
        )
        row = self._add_field(
            parent,
            row,
            "Bit Depth:",
            f"{self.session.audio_config.bit_depth} bit",
        )
        row = self._add_field(
            parent,
            row,
            "Channels:",
            (
                f"{self.session.audio_config.channels} (Mono)"
                if self.session.audio_config.channels == 1
                else f"{self.session.audio_config.channels}"
            ),
        )
        row = self._add_field(
            parent, row, "Format:", self.session.audio_config.format.upper()
        )
        return self._add_separator(parent, row)

    def _create_script_section(
        self, parent: ttk.Frame, row: int, right_frame: ttk.Frame
    ) -> None:
        """Create script information section."""
        row = self._add_section_header(parent, row, "Script Information")
        script_path = self.session.get_script_path()

        if not script_path or not script_path.exists():
            self._add_field(parent, row, "Script File:", "Not found")
            return

        row = self._add_field(parent, row, "Script File:", script_path.name)

        # Parse script and get text lengths
        script_data = FestivalScriptParser.parse_script(script_path)
        if not script_data:
            return

        text_lengths = self._calculate_text_lengths(script_data)
        row = self._add_field(parent, row, "Total Utterances:", str(len(script_data)))

        # Add statistics
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            min_length = min(text_lengths)
            max_length = max(text_lengths)

            row = self._add_field(
                parent, row, "Average Length:", f"{avg_length:.1f} chars"
            )
            self._add_field(
                parent,
                row,
                "Min/Max Length:",
                f"{min_length} / {max_length} chars",
            )

            # Add histogram to right frame
            self._add_histogram(right_frame, text_lengths)

    def _create_close_button(self, parent: ttk.Frame) -> None:
        """Create close button."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        ttk.Button(
            button_frame, text="Close", command=self._on_close, default=tk.ACTIVE
        ).pack(side=tk.RIGHT)

    @staticmethod
    def _add_section_header(parent: ttk.Frame, row: int, text: str) -> int:
        """Add a section header."""
        label = ttk.Label(parent, text=text, font=("TkDefaultFont", 0, "bold"))
        label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(8, 5))
        return row + 1

    @staticmethod
    def _add_field(parent: ttk.Frame, row: int, label: str, value: str) -> int:
        """Add a labeled field."""
        # Label for the field name
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky=tk.W, padx=(20, 10), pady=3
        )

        # Label for the value (instead of Entry)
        value_label = ttk.Label(parent, text=value, foreground="#333333")
        value_label.grid(row=row, column=1, sticky=tk.W, pady=3)

        return row + 1

    @staticmethod
    def _add_separator(parent: ttk.Frame, row: int) -> int:
        """Add a horizontal separator."""
        row += 1
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="we", pady=5
        )
        return row + 1

    @staticmethod
    def _format_gender(gender: str) -> str:
        """Format gender code for display."""
        gender_map = {"M": "Male", "F": "Female", "X": "Other"}
        return gender_map.get(gender, gender)

    @staticmethod
    def _calculate_text_lengths(script_data: Dict[str, str]) -> List[int]:
        """Calculate lengths of clean text (without emotion prefix)."""
        return [
            len(FestivalScriptParser.extract_intensity_and_text(text)[1])
            for text in script_data.values()
        ]

    @staticmethod
    def _add_histogram(parent: ttk.Frame, text_lengths: List[int]) -> None:
        """Add a graphical histogram of text lengths using Canvas.

        Args:
            parent: Parent frame
            text_lengths: List of text lengths
        """
        # Add title
        title_label = ttk.Label(
            parent, text="Text Length Distribution", font=("TkDefaultFont", 12, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Create histogram frame
        hist_frame = ttk.Frame(parent)
        hist_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas dimensions - much larger
        canvas_width = 750
        canvas_height = 450
        margin_left = 70
        margin_right = 40
        margin_top = 30
        margin_bottom = 80

        # Create canvas
        canvas = tk.Canvas(
            hist_frame,
            width=canvas_width,
            height=canvas_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        canvas.pack()

        # Calculate histogram data
        max_length = max(text_lengths)

        # Determine bin size and count
        bin_size = 25
        num_bins = min((max_length // bin_size) + 1, 20)

        if max_length // bin_size > 20:
            bin_size = (max_length // 19) + 1
            num_bins = (max_length // bin_size) + 1

        # Count frequencies
        bins = [0] * num_bins
        bin_ranges = [(i * bin_size, (i + 1) * bin_size) for i in range(num_bins)]

        for length in text_lengths:
            bins[min(length // bin_size, num_bins - 1)] += 1

        max_freq = max(bins) if bins else 1
        plot_width = canvas_width - margin_left - margin_right
        plot_height = canvas_height - margin_top - margin_bottom

        # Draw axes
        canvas.create_line(
            margin_left, margin_top, margin_left, margin_top + plot_height, width=2
        )
        canvas.create_line(
            margin_left,
            margin_top + plot_height,
            margin_left + plot_width,
            margin_top + plot_height,
            width=2,
        )

        # Draw bars
        bar_width = plot_width / num_bins
        bar_spacing = bar_width * 0.1
        actual_bar_width = bar_width - bar_spacing

        for i, freq in enumerate(bins):
            if freq > 0:
                bar_height = (freq / max_freq) * plot_height
                x1 = margin_left + i * bar_width + bar_spacing / 2
                y1 = margin_top + plot_height - bar_height
                x2 = x1 + actual_bar_width
                y2 = margin_top + plot_height

                canvas.create_rectangle(
                    x1, y1, x2, y2, fill="#4a90e2", outline="#3a7bc8"
                )
                canvas.create_text(
                    (x1 + x2) / 2, y1 - 5, text=str(freq), font=("TkDefaultFont", 10)
                )

        # Draw X-axis labels
        for i in range(0, num_bins, max(1, num_bins // 10)):
            x = margin_left + i * bar_width + bar_width / 2
            y = margin_top + plot_height + 15

            start, end = bin_ranges[i]
            label = (
                f"{start}+"
                if i == num_bins - 1 and end > max_length
                else f"{start}-{end}"
            )

            canvas.create_text(
                x, y, text=label, anchor="n", font=("TkDefaultFont", 10), angle=45
            )

        # Draw Y-axis labels
        for i in range(0, 5):
            freq_val = int((max_freq * i) / 4)
            y = margin_top + plot_height - (plot_height * i / 4)

            canvas.create_line(margin_left - 5, y, margin_left, y, width=1)
            canvas.create_text(
                margin_left - 10,
                y,
                text=str(freq_val),
                anchor="e",
                font=("TkDefaultFont", 10),
            )

        # Axis labels
        canvas.create_text(
            canvas_width // 2,
            canvas_height - 5,
            text="Text Length (characters)",
            font=("TkDefaultFont", 11),
        )
        canvas.create_text(
            20, canvas_height // 2, text="Count", font=("TkDefaultFont", 11), angle=90
        )

    def _on_close(self):
        """Handle close button."""
        self.dialog.destroy()

    def show(self):
        """Show the dialog."""
        self.dialog.wait_window()
