"""Dialog for importing raw text files to Festival script format."""

import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
import numpy as np
from pathlib import Path
from typing import Optional, List
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ...utils.text_importer import TextImporter
from .dialog_utils import setup_dialog_window


class ImportTextDialog:
    """Dialog for importing text files and converting to Festival script format."""

    MIN_WIDTH = 700
    DEFAULT_HEIGHT = 900
    MAX_WIDTH = 1000

    PADDING = {
        "label_column": 150,
        "browse_button": 80,
        "entry_borders": 40,
        "dialog_margins": 60,
    }
    PADDING_STANDARD = 10
    PADDING_SMALL = 5
    PADDING_FRAME = "10"

    def __init__(
        self, parent: tk.Tk, default_dir: Optional[Path] = None, settings_manager=None
    ):
        """Initialize the import text dialog.

        Args:
            parent: Parent window
            default_dir: Default directory for file operations
            settings_manager: Settings manager for saving/loading preferences
        """
        self.parent = parent
        self.settings_manager = settings_manager
        self.result = None

        self._load_directories(default_dir)

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Import Text to Script")

        self._initialize_variables()
        self._create_widgets()

        self.input_file_var.set(str(self.input_dir) + "/")
        self.output_file_var.set(str(self.output_dir / "script.txt"))

        optimal_width = self._calculate_optimal_width()
        setup_dialog_window(
            self.dialog,
            self.parent,
            title="Import Text to Script",
            center_on_parent=True,
            width=optimal_width,
            height=self.DEFAULT_HEIGHT,
        )
        self.dialog.minsize(self.MIN_WIDTH, self.DEFAULT_HEIGHT)

    def _load_directories(self, default_dir: Optional[Path]) -> None:
        """Load saved directories from settings."""
        if self.settings_manager:
            self.input_dir = Path(
                getattr(self.settings_manager.settings, "import_input_dir", None)
                or default_dir
                or Path.home()
            )
            self.output_dir = Path(
                getattr(self.settings_manager.settings, "import_output_dir", None)
                or default_dir
                or Path.home()
            )
        else:
            self.input_dir = self.output_dir = default_dir or Path.home()

    def _initialize_variables(self):
        """Initialize tkinter variables."""
        self.input_file_var = tk.StringVar()
        self.output_file_var = tk.StringVar()

        self.max_length_var = tk.IntVar(value=80)
        self.split_mode_var = tk.StringVar(value="lines")
        self.max_length_label = None

        self.add_emotion_var = tk.BooleanVar(value=False)
        self.emotion_mode_var = tk.StringVar(value="fixed")
        self.fixed_level_var = tk.IntVar(value=3)

        self.dist_mean_var = tk.DoubleVar(value=2.5)
        self.dist_std_var = tk.DoubleVar(value=1.0)
        self.dist_min_var = tk.IntVar(value=1)
        self.dist_max_var = tk.IntVar(value=5)

        self.preview_text = None
        self.stats_label = None
        self.figure = None
        self.canvas = None

        self.emotion_spinboxes = []
        self.emotion_radio_buttons = []
        self.fixed_level_combo = None

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=self.PADDING_FRAME)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._create_file_section(main_frame)
        self._create_processing_section(main_frame)
        self._create_emotion_section(main_frame)
        self._create_preview_section(main_frame)
        self._create_buttons(main_frame)

        # Initialize split mode explanation
        self._on_split_mode_change()

    def _create_file_section(self, parent):
        """Create file selection section."""
        frame = ttk.LabelFrame(parent, text="Files", padding=str(self.PADDING_SMALL))
        frame.pack(fill=tk.X, pady=(0, self.PADDING_STANDARD))

        def create_file_row(
            row: int,
            label: str,
            var: tk.StringVar,
            command: callable,
            pad_top: bool = False,
        ):
            pady = (self.PADDING_SMALL, 0) if pad_top else 0
            ttk.Label(frame, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, self.PADDING_SMALL), pady=pady
            )
            ttk.Entry(frame, textvariable=var, width=40).grid(
                row=row,
                column=1,
                sticky=(tk.W, tk.E),
                padx=(0, self.PADDING_SMALL),
                pady=pady,
            )
            ttk.Button(frame, text="Browse...", command=command).grid(
                row=row, column=2, pady=pady
            )

        create_file_row(0, "Input Text File:", self.input_file_var, self._browse_input)
        create_file_row(
            1,
            "Output Script File:",
            self.output_file_var,
            self._browse_output,
            pad_top=True,
        )
        frame.columnconfigure(1, weight=1)

    def _create_processing_section(self, parent):
        """Create text processing options section."""
        frame = ttk.LabelFrame(
            parent, text="Text Processing", padding=str(self.PADDING_SMALL)
        )
        frame.pack(fill=tk.X, pady=(0, self.PADDING_STANDARD))

        # Max length spinbox
        length_frame = ttk.Frame(frame)
        length_frame.pack(fill=tk.X)

        ttk.Label(length_frame, text="Max length:").pack(side=tk.LEFT)

        spinbox = ttk.Spinbox(
            length_frame,
            from_=20,
            to=500,
            textvariable=self.max_length_var,
            width=10,
            command=self._on_max_length_change,
        )
        spinbox.pack(side=tk.LEFT, padx=(self.PADDING_SMALL, 0))
        spinbox.bind("<KeyRelease>", lambda e: self._on_max_length_change())
        spinbox.bind("<FocusOut>", lambda e: self._on_max_length_change())
        self.max_length_label = ttk.Label(length_frame, text="characters per sentence")
        self.max_length_label.pack(side=tk.LEFT, padx=(self.PADDING_SMALL, 0))

        # Split mode radio buttons
        ttk.Label(frame, text="Split mode:").pack(
            anchor=tk.W, pady=(self.PADDING_SMALL, 0)
        )

        split_frame = ttk.Frame(frame)
        split_frame.pack(fill=tk.X, padx=(20, 0))

        modes = [
            ("Lines", "lines"),
            ("Sentences", "sentences"),
            ("Paragraphs", "paragraphs"),
        ]
        for i, (text, value) in enumerate(modes):
            padx = (0, 15) if i < len(modes) - 1 else 0
            ttk.Radiobutton(
                split_frame,
                text=text,
                variable=self.split_mode_var,
                value=value,
                command=self._on_split_mode_change,
            ).pack(side=tk.LEFT, padx=padx)

        # Explanation label for split mode
        self.split_explanation_label = ttk.Label(
            frame,
            text="",
            font=("TkDefaultFont", 9, "italic"),
            foreground="gray",
            wraplength=550,
        )
        self.split_explanation_label.pack(anchor=tk.W, padx=(20, 0), pady=(2, 0))

        self.stats_label = ttk.Label(frame, text="", foreground="darkgreen")
        self.stats_label.pack(anchor=tk.W, pady=(self.PADDING_SMALL, 0))

    def _create_emotion_section(self, parent):
        """Create emotion level options section."""
        frame = ttk.LabelFrame(
            parent, text="Emotion Levels", padding=str(self.PADDING_SMALL)
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, self.PADDING_STANDARD))

        ttk.Checkbutton(
            frame,
            text="Add emotion levels",
            variable=self.add_emotion_var,
            command=self._on_emotion_toggle,
        ).pack(anchor=tk.W)

        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.BOTH, expand=True, pady=(self.PADDING_SMALL, 0))

        # Fixed level option
        fixed_frame = ttk.Frame(options_frame)
        fixed_frame.pack(fill=tk.X)

        fixed_radio = ttk.Radiobutton(
            fixed_frame,
            text="Fixed level:",
            variable=self.emotion_mode_var,
            value="fixed",
            command=self._on_mode_change,
        )
        fixed_radio.pack(side=tk.LEFT)
        self.emotion_radio_buttons.append(fixed_radio)

        self.fixed_level_combo = ttk.Combobox(
            fixed_frame,
            textvariable=self.fixed_level_var,
            values=list(range(0, 100)),  # Allow 0 to 99
            width=5,
            state="readonly",
        )
        self.fixed_level_combo.pack(side=tk.LEFT, padx=(self.PADDING_SMALL, 0))

        # Distribution option
        dist_frame = ttk.Frame(options_frame)
        dist_frame.pack(fill=tk.X, pady=(self.PADDING_SMALL, 0))

        dist_radio = ttk.Radiobutton(
            dist_frame,
            text="Normal distribution:",
            variable=self.emotion_mode_var,
            value="distribution",
            command=self._on_mode_change,
        )
        dist_radio.pack(anchor=tk.W)
        self.emotion_radio_buttons.append(dist_radio)

        # Distribution parameters
        params_frame = ttk.Frame(options_frame)
        params_frame.pack(fill=tk.X, padx=(20, 0))

        params = [
            (
                ttk.Frame(params_frame),
                [
                    ("Mean:", self.dist_mean_var, 0.0, 99.0, 0.1),
                    ("Std Dev:", self.dist_std_var, 0.1, 50.0, 0.1),
                ],
            ),
            (
                ttk.Frame(params_frame),
                [
                    ("Min:", self.dist_min_var, 0, 99, 1),
                    ("Max:", self.dist_max_var, 0, 99, 1),
                ],
            ),
        ]

        for row_frame, row_params in params:
            row_frame.pack(
                fill=tk.X,
                pady=(0 if row_frame == params[0][0] else self.PADDING_SMALL, 0),
            )
            for i, (label, var, from_val, to_val, inc) in enumerate(row_params):
                padx_label = (20, 0) if i > 0 else 0
                ttk.Label(row_frame, text=label).pack(side=tk.LEFT, padx=padx_label)

                def make_validate_func(min_v, max_v, is_int):
                    def validate(value):
                        if value == "" or value == "-":
                            return True
                        try:
                            val = int(value) if is_int else float(value)
                            return min_v <= val <= max_v
                        except ValueError:
                            return False

                    return validate

                is_integer = label in ["Min:", "Max:"]
                validate_func = make_validate_func(from_val, to_val, is_integer)
                vcmd_dist = (self.dialog.register(validate_func), "%P")

                spinbox = ttk.Spinbox(
                    row_frame,
                    from_=from_val,
                    to=to_val,
                    increment=inc,
                    textvariable=var,
                    width=8,
                    validate="key",
                    validatecommand=vcmd_dist,
                )
                spinbox.pack(side=tk.LEFT, padx=(self.PADDING_SMALL, 0))
                var.trace("w", lambda *args: self._on_distribution_change())
                self.emotion_spinboxes.append(spinbox)

        # Matplotlib plot
        plot_frame = ttk.Frame(options_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(self.PADDING_STANDARD, 0))

        self.figure = Figure(figsize=(6, 3), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._set_emotion_widgets_state("disabled")

    def _create_preview_section(self, parent):
        """Create preview section."""
        frame = ttk.LabelFrame(parent, text="Preview", padding=str(self.PADDING_SMALL))
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, self.PADDING_STANDARD))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.preview_text = tk.Text(
            text_frame,
            height=6,
            width=60,
            yscrollcommand=scrollbar.set,
            state="disabled",
            wrap=tk.WORD,
        )
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.preview_text.yview)

    def _create_buttons(self, parent):
        """Create dialog buttons."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(self.PADDING_STANDARD, 0))

        ttk.Button(frame, text="Cancel", command=self._cancel).pack(
            side=tk.RIGHT, padx=(self.PADDING_SMALL, 0)
        )
        ttk.Button(frame, text="Import", command=self._import).pack(side=tk.RIGHT)

    def _browse_input(self):
        """Browse for input text file."""
        filename = filedialog.askopenfilename(
            parent=self.dialog,
            title="Select Text File",
            initialdir=self.input_dir,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if filename:
            self._handle_input_selection(filename)

    def _handle_input_selection(self, filename: str):
        """Handle input file selection."""
        self.input_file_var.set(filename)
        input_path = Path(filename)

        self.input_dir = input_path.parent
        if self.settings_manager:
            self.settings_manager.update_setting(
                "import_input_dir", str(self.input_dir)
            )

        self.output_file_var.set(str(self.output_dir / f"{input_path.stem}_script.txt"))

        self._update_statistics()
        self._preview()

    def _browse_output(self):
        """Browse for output script file."""
        current = self.output_file_var.get()
        if current:
            initial_dir = Path(current).parent
            initial_file = Path(current).name
        else:
            initial_dir = self.output_dir
            initial_file = "script.txt"

        filename = filedialog.asksaveasfilename(
            parent=self.dialog,
            title="Save Script File As",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if filename:
            self.output_file_var.set(filename)
            self.output_dir = Path(filename).parent
            if self.settings_manager:
                self.settings_manager.update_setting(
                    "import_output_dir", str(self.output_dir)
                )

    def _calculate_optimal_width(self) -> int:
        """Calculate optimal dialog width based on directory paths."""
        entry_font = font.nametofont("TkDefaultFont")
        longer_path = max(str(self.input_dir), str(self.output_dir), key=len)
        text_width = entry_font.measure(longer_path)

        total_width = text_width + sum(self.PADDING.values())
        return max(self.MIN_WIDTH, min(self.MAX_WIDTH, total_width))

    def _update_statistics(self):
        """Update statistics label based on selected file and split mode."""
        input_file = self.input_file_var.get()
        if not input_file:
            self.stats_label.config(text="")
            return

        path = Path(input_file)
        if not path.exists() or path.is_dir():
            self.stats_label.config(text="")
            return

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()

            mode = self.split_mode_var.get()

            if mode == "sentences":
                items = re.split(r"(?<=[.!?])\s+", text)
                mode_name = "sentences"
            elif mode == "paragraphs":
                items = re.split(r"\n\s*\n", text)
                mode_name = "paragraphs"
            else:  # lines
                items = text.split("\n")
                mode_name = "lines"

            items = [item.strip() for item in items if item.strip()]

            if items:
                lengths = [len(item) for item in items]
                self.stats_label.config(
                    text=f"Found {len(items)} {mode_name}: max {max(lengths)} chars, avg {sum(lengths)/len(lengths):.0f} chars"
                )
            else:
                self.stats_label.config(text=f"No {mode_name} found")

        except Exception as e:
            self.stats_label.config(text=f"Error reading file: {str(e)}")

    def _on_max_length_change(self):
        """Handle max length change."""
        input_file = self.input_file_var.get()
        if input_file:
            path = Path(input_file)
            if path.exists() and path.is_file():
                self._preview()

    def _on_split_mode_change(self):
        """Handle split mode change."""
        mode = self.split_mode_var.get()
        if self.max_length_label:
            text_map = {
                "lines": "characters per line",
                "sentences": "characters per sentence",
                "paragraphs": "characters per paragraph",
            }
            self.max_length_label.config(text=text_map.get(mode, ""))

        # Update split mode explanation
        explanations = {
            "lines": "Text is split at original line breaks. Each line becomes a separate utterance, split further if exceeding max length.",
            "sentences": "Text is split at sentence boundaries (. ! ?) followed by whitespace. Each sentence becomes a separate utterance, split further only if exceeding max length.",
            "paragraphs": "Text is split at paragraph boundaries (double line breaks). Long paragraphs are further split at sentence boundaries if needed.",
        }
        if hasattr(self, "split_explanation_label"):
            self.split_explanation_label.config(text=explanations.get(mode, ""))

        # Only update statistics and preview if a valid file is selected
        input_file = self.input_file_var.get()
        if input_file:
            path = Path(input_file)
            if path.exists() and path.is_file():
                self._update_statistics()
                self._preview()

    def _on_emotion_toggle(self):
        """Handle emotion checkbox toggle."""
        state = "normal" if self.add_emotion_var.get() else "disabled"
        self._set_emotion_widgets_state(state)
        if state == "normal":
            self._on_mode_change()

    def _on_mode_change(self):
        """Handle emotion mode change."""
        if not self.add_emotion_var.get():
            return

        is_fixed = self.emotion_mode_var.get() == "fixed"
        self.fixed_level_combo.config(state="readonly" if is_fixed else "disabled")

        for spinbox in self.emotion_spinboxes:
            spinbox.config(state="disabled" if is_fixed else "normal")

        if is_fixed:
            self._show_plot_placeholder()
        else:
            self._update_distribution_plot()

    def _on_distribution_change(self):
        """Handle distribution parameter change."""
        if self.emotion_mode_var.get() == "distribution":
            self._update_distribution_plot()

    def _show_plot_placeholder(self):
        """Show placeholder text in the plot area."""
        if self.figure:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Select 'Normal distribution' to see histogram",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()

    def _set_emotion_widgets_state(self, state):
        """Set state of emotion-related widgets."""
        # Set state for radio buttons
        for radio in self.emotion_radio_buttons:
            radio.config(state=state)

        # Set state for combo and spinboxes
        self.fixed_level_combo.config(state=state)
        for spinbox in self.emotion_spinboxes:
            spinbox.config(state=state)

        # Update plot visibility
        if state == "disabled":
            self._show_plot_placeholder()
        elif state == "normal":
            self._on_mode_change()

    def _update_distribution_plot(self):
        """Update the distribution plot."""
        if not self.figure:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            mean = float(self.dist_mean_var.get() or 0)
            std_dev = float(self.dist_std_var.get() or 1)
            min_val = int(self.dist_min_var.get() or -3)
            max_val = int(self.dist_max_var.get() or 3)

            if std_dev <= 0:
                raise ValueError("Standard deviation must be positive")
            if min_val >= max_val:
                raise ValueError("Min must be less than Max")
            if mean < min_val or mean > max_val:
                raise ValueError("Mean must be between Min and Max")

            samples = TextImporter.truncated_normal(
                100, mean, std_dev, min_val, max_val
            )

            bins = list(range(int(min_val), int(max_val) + 2))
            if len(bins) > 1:
                ax.hist(
                    samples,
                    bins=bins,
                    alpha=0.7,
                    density=True,
                    edgecolor="black",
                    align="left",
                )

            x = np.linspace(min_val, max_val, 100)
            pdf = TextImporter.calculate_truncated_normal_pdf(
                x, mean, std_dev, min_val, max_val
            )
            ax.plot(x, pdf, "r-", linewidth=2, label="Theoretical")

            ax.set_ylabel("Probability")
            ax.set_title("Emotion Level Distribution")
            # Show reasonable number of ticks even for large ranges
            tick_range = max_val - min_val
            if tick_range <= 20:
                ax.set_xticks(range(min_val, max_val + 1))
            else:
                # Show at most 20 ticks for readability
                step = max(1, tick_range // 20)
                ax.set_xticks(range(min_val, max_val + 1, step))
            ax.legend()
            ax.grid(True, alpha=0.3)

        except (tk.TclError, ValueError) as e:
            ax.text(
                0.5,
                0.5,
                str(e) if isinstance(e, ValueError) else "Please enter valid values",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        self.canvas.draw()

    def _generate_preview(self) -> List[str]:
        """Generate preview utterances."""
        input_file = self.input_file_var.get()
        if not input_file:
            return []

        path = Path(input_file)
        if not path.exists() or path.is_dir():
            return []

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()

            utterances = TextImporter.split_text(
                text, self.max_length_var.get(), self.split_mode_var.get()
            )

            if not utterances:
                return ["No utterances generated from input text"]

            preview_utterances = utterances[:5]
            labels = TextImporter.generate_labels(len(preview_utterances))

            if self.add_emotion_var.get():
                mode = self.emotion_mode_var.get()
                if mode == "fixed":
                    preview_utterances = TextImporter.add_emotion_levels(
                        preview_utterances,
                        mode="fixed",
                        fixed_level=self.fixed_level_var.get(),
                    )
                else:
                    try:
                        params = {
                            "mean": float(self.dist_mean_var.get() or 0),
                            "std_dev": float(self.dist_std_var.get() or 1),
                            "min_val": int(self.dist_min_var.get() or -3),
                            "max_val": int(self.dist_max_var.get() or 3),
                        }

                        if (
                            params["std_dev"] <= 0
                            or params["min_val"] >= params["max_val"]
                        ):
                            return ["Error: Invalid distribution parameters"]

                        preview_utterances = TextImporter.add_emotion_levels(
                            preview_utterances,
                            mode="distribution",
                            distribution_params=params,
                        )
                    except (ValueError, tk.TclError) as e:
                        return [f"Error: Invalid distribution parameters - {str(e)}"]

            preview_lines = []
            for label, text in zip(labels, preview_utterances):
                text = text.replace('"', '\\"')
                preview_lines.append(f'( {label} "{text}" )')

            if len(utterances) > 5:
                preview_lines.append(f"... and {len(utterances) - 5} more utterances")

            return preview_lines

        except Exception as e:
            return [f"Error generating preview: {str(e)}"]

    def _preview(self):
        """Show preview of generated script."""
        preview_lines = self._generate_preview()

        self.preview_text.config(state="normal")
        self.preview_text.delete(1.0, tk.END)
        if preview_lines:
            self.preview_text.insert(1.0, "\n".join(preview_lines))
        self.preview_text.config(state="disabled")

    def _import(self):
        """Perform the import."""
        input_file = self.input_file_var.get()
        output_file = self.output_file_var.get()

        if not input_file:
            messagebox.showerror("Error", "Please select an input file")
            return

        if not output_file:
            messagebox.showerror("Error", "Please specify an output file")
            return

        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            messagebox.showerror("Error", f"Input file does not exist: {input_file}")
            return

        if input_path.is_dir():
            messagebox.showerror(
                "Error", f"Selected path is a directory, not a file: {input_file}"
            )
            return

        if output_path.exists():
            if not messagebox.askyesno(
                "File Exists",
                f"Output file already exists:\n{output_file}\n\nOverwrite?",
            ):
                return

        try:
            emotion_mode = "none"
            fixed_level = 0
            distribution_params = None

            if self.add_emotion_var.get():
                mode = self.emotion_mode_var.get()
                if mode == "fixed":
                    emotion_mode = "fixed"
                    fixed_level = self.fixed_level_var.get()
                else:
                    emotion_mode = "distribution"
                    try:
                        distribution_params = {
                            "mean": float(self.dist_mean_var.get() or 0),
                            "std_dev": float(self.dist_std_var.get() or 1),
                            "min_val": int(self.dist_min_var.get() or -3),
                            "max_val": int(self.dist_max_var.get() or 3),
                        }

                        if distribution_params["std_dev"] <= 0:
                            messagebox.showerror(
                                "Error", "Standard deviation must be positive"
                            )
                            return
                        if (
                            distribution_params["min_val"]
                            >= distribution_params["max_val"]
                        ):
                            messagebox.showerror("Error", "Min must be less than Max")
                            return
                        if (
                            distribution_params["mean"] < distribution_params["min_val"]
                            or distribution_params["mean"]
                            > distribution_params["max_val"]
                        ):
                            messagebox.showerror(
                                "Error", "Mean must be between Min and Max"
                            )
                            return
                    except (ValueError, tk.TclError) as e:
                        messagebox.showerror(
                            "Error", f"Invalid distribution parameters: {str(e)}"
                        )
                        return

            num_utterances = TextImporter.import_text_file(
                input_path,
                output_path,
                max_length=self.max_length_var.get(),
                split_mode=self.split_mode_var.get(),
                emotion_mode=emotion_mode,
                fixed_level=fixed_level,
                distribution_params=distribution_params,
            )

            self.result = output_path
            messagebox.showinfo(
                "Success",
                f"Successfully imported {num_utterances} utterances to:\n{output_file}",
            )

            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import text:\n{str(e)}")

    def _cancel(self):
        """Cancel the dialog."""
        self.result = None
        self.dialog.destroy()

    def show(self) -> Optional[Path]:
        """Show the dialog and return the result."""
        self.dialog.wait_window()
        return self.result
