"""Base class for dialogs displaying utterance lists."""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Tuple
import re
from enum import Enum

from .dialog_utils import setup_dialog_window


class SortDirection(Enum):
    """Sort direction for utterance lists."""

    UP = "up"  # Ascending (A-Z, 0-9)
    DOWN = "down"  # Descending (Z-A, 9-0)


class UtteranceListDialog:
    """Base class for dialogs that display and interact with utterance lists."""

    # Dialog dimensions
    DIALOG_WIDTH_RATIO = 0.7  # 70% of parent width
    DIALOG_HEIGHT_RATIO = 0.8  # 80% of parent height
    MIN_WIDTH = 700
    MIN_HEIGHT = 500

    # Column widths (min, max) - will be calculated dynamically
    COLUMN_LIMITS = {
        "#0": (70, 120),  # Label column (min, max)
        "emotion": (50, 80),  # Emotion column
        "text": (200, 500),  # Text column
        "recordings": (50, 80),  # Takes column
    }

    # Character width approximation for column sizing
    CHAR_WIDTH = 7  # Approximate pixels per character

    # Padding values
    PADDING_STANDARD = 10
    PADDING_SMALL = 5
    PADDING_FRAME = "10"

    def __init__(
        self,
        parent: tk.Widget,
        utterances: List[str],
        labels: List[str],
        file_manager,
        title: str = "Utterance List",
        show_search: bool = True,
        show_filters: bool = True,
        utterance_order: List[int] = None,
        default_sort: str = "order",
        default_sort_direction: SortDirection = SortDirection.UP,
        current_index: int = None,
    ):
        """Initialize the dialog base.

        Args:
            parent: Parent widget
            utterances: List of utterance texts
            labels: List of utterance labels
            file_manager: RecordingFileManager for scanning files
            title: Dialog window title
            show_search: Whether to show search field
            show_filters: Whether to show filter checkboxes
            utterance_order: Display order for utterances (list of indices)
            default_sort: Default sort column ("order", "label", etc.)
            default_sort_direction: Default sort direction (UP or DOWN)
            current_index: Currently selected utterance index to highlight
        """
        self.parent = parent
        self.utterances = utterances
        self.labels = labels
        self.file_manager = file_manager
        # Dynamically scan takes (excluding trash)
        take_files = file_manager.scan_all_take_files(labels)
        self.takes = {label: len(files) for label, files in take_files.items()}
        self.show_search = show_search
        self.show_filters = show_filters
        self.utterance_order = utterance_order or list(range(len(utterances)))
        self.current_index = current_index

        # Sorting state
        self.sort_column = default_sort  # Default sort column
        self.sort_reverse = default_sort_direction == SortDirection.DOWN
        self.title = title

        # Create dialog window
        self.dialog = tk.Toplevel(parent)

        # First prepare data to calculate optimal sizes
        self._prepare_data()

        # Create UI with calculated dimensions
        self._create_ui()

        # Now populate the tree and apply filters
        self._apply_data()

        # Setup window with dynamic size calculation
        setup_dialog_window(
            self.dialog,
            self.parent,
            title=title,
            size_callback=self._calculate_optimal_size,
            min_width=self.MIN_WIDTH,
            min_height=self.MIN_HEIGHT,
            center_on_parent=True,
        )

        # Focus handling (can be overridden)
        self._set_initial_focus()

        # Lift to top
        self.dialog.lift()

    def _prepare_data(self) -> None:
        """Prepare data and calculate optimal column widths."""
        self.all_items = []

        # Create items in the display order
        for display_pos, actual_idx in enumerate(self.utterance_order):
            label = self.labels[actual_idx]
            utterance = self.utterances[actual_idx]

            # Get take count
            take_count = self.takes.get(label, 0)

            # Extract emotion label if present
            emotion, clean_text = self._extract_emotion(utterance)

            # Truncate text for display (first 60 chars)
            display_text = clean_text[:60]
            if len(clean_text) > 60:
                display_text += "..."

            # Store item data with both display position and actual index
            item_data = {
                "index": actual_idx,  # Original index in utterances list
                "display_pos": display_pos,  # Position in the display order
                "label": label,
                "emotion": emotion,
                "text": display_text,
                "full_text": utterance,
                "clean_text": clean_text,
                "takes": take_count,
            }
            self.all_items.append(item_data)

        # Calculate optimal column widths
        self.column_widths = self._calculate_column_widths()

    def _extract_emotion(self, text: str) -> Tuple[str, str]:
        """Extract emotion label from text if present.

        Args:
            text: The utterance text

        Returns:
            Tuple of (emotion_label, clean_text)
        """
        # Pattern to match "N: " at the beginning where N is a natural number
        emotion_pattern = r"^(\d+):\s*(.*)$"
        match = re.match(emotion_pattern, text)

        if match:
            emotion = match.group(1)
            clean_text = match.group(2)
            return emotion, clean_text
        return "", text

    def _calculate_optimal_size(self) -> Tuple[int, int]:
        """Calculate optimal dialog size based on content.

        Returns:
            Tuple of (width, height) in pixels
        """
        # Update parent to ensure we get correct dimensions
        self.parent.update_idletasks()

        # Get parent window position and size
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate dialog width based on column widths
        # Add up column widths plus padding and scrollbar
        content_width = (
            self.column_widths["#0"]
            + self.column_widths["emotion"]
            + self.column_widths["text"]
            + self.column_widths["recordings"]
            + 50  # Extra for padding, borders, scrollbar
        )

        # Add padding for the dialog frame
        dialog_width = content_width + 40

        # Ensure minimum width
        dialog_width = max(dialog_width, self.MIN_WIDTH)

        # Limit to parent width if too large
        max_width = int(parent_width * 0.9)
        dialog_width = min(dialog_width, max_width)

        # For height: Since the dialog will be centered, we need to ensure
        # it doesn't extend beyond parent bounds when centered
        # Maximum height should be at most the parent height
        # (centering will place it within parent bounds)

        # Start with desired height based on ratio
        desired_height = int(parent_height * self.DIALOG_HEIGHT_RATIO)

        # But limit to parent height with some margin
        # Using 95% to leave a small visual margin
        max_height = int(parent_height * 0.95)

        # Apply constraints
        dialog_height = max(self.MIN_HEIGHT, min(desired_height, max_height))

        # Final check: if dialog would be taller than parent, limit it
        if dialog_height > parent_height:
            dialog_height = parent_height
        return dialog_width, dialog_height

    def _calculate_column_widths(self) -> Dict[str, int]:
        """Calculate optimal column widths based on content.

        Returns:
            Dictionary mapping column names to calculated widths
        """
        widths = {}

        # Calculate maximum content width for each column
        # Label column
        max_label = max((len(item["label"]) for item in self.all_items), default=10)
        widths["#0"] = min(
            max(max_label * self.CHAR_WIDTH + 20, self.COLUMN_LIMITS["#0"][0]),
            self.COLUMN_LIMITS["#0"][1],
        )

        # Emotion column
        max_emotion = max((len(item["emotion"]) for item in self.all_items), default=0)
        if max_emotion > 0:
            widths["emotion"] = min(
                max(
                    max_emotion * self.CHAR_WIDTH + 20, self.COLUMN_LIMITS["emotion"][0]
                ),
                self.COLUMN_LIMITS["emotion"][1],
            )
        else:
            # No emotions, use minimum width
            widths["emotion"] = self.COLUMN_LIMITS["emotion"][0]

        # Text column - use the display text length
        max_text = max((len(item["text"]) for item in self.all_items), default=40)
        widths["text"] = min(
            max(max_text * self.CHAR_WIDTH + 20, self.COLUMN_LIMITS["text"][0]),
            self.COLUMN_LIMITS["text"][1],
        )

        # Recordings column
        max_takes = max((len(str(item["takes"])) for item in self.all_items), default=1)
        widths["recordings"] = min(
            max(max_takes * self.CHAR_WIDTH + 30, self.COLUMN_LIMITS["recordings"][0]),
            self.COLUMN_LIMITS["recordings"][1],
        )

        return widths

    def _create_ui(self) -> None:
        """Create the UI elements."""
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, padding=self.PADDING_FRAME)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Optional info text (can be overridden by subclasses)
        self._create_info_frame(main_frame)

        # Optional search frame
        if self.show_search:
            self._create_search_frame(main_frame)

        # Optional filter frame
        if self.show_filters:
            self._create_filter_frame(main_frame)

        # List frame with scrollbar
        self._create_list_frame(main_frame)

        # Button frame (to be implemented by subclasses)
        self._create_button_frame(main_frame)

        # Bind common events
        self._bind_events()

    def _create_info_frame(self, parent: ttk.Frame) -> None:
        """Create info frame. Can be overridden by subclasses to add info text."""
        pass

    def _create_search_frame(self, parent: ttk.Frame) -> None:
        """Create the search frame."""
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=tk.X, pady=(0, self.PADDING_STANDARD))

        ttk.Label(search_frame, text="Search:").pack(
            side=tk.LEFT, padx=(0, self.PADDING_SMALL)
        )

        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search_changed)
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_filter_frame(self, parent: ttk.Frame) -> None:
        """Create the filter frame."""
        filter_frame = ttk.Frame(parent)
        filter_frame.pack(fill=tk.X, pady=(0, self.PADDING_STANDARD))

        self.show_with_recordings = tk.BooleanVar(value=True)
        self.show_without_recordings = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            filter_frame,
            text="With Recordings",
            variable=self.show_with_recordings,
            command=self._apply_filters,
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Checkbutton(
            filter_frame,
            text="Without Recordings",
            variable=self.show_without_recordings,
            command=self._apply_filters,
        ).pack(side=tk.LEFT)

    def _create_list_frame(self, parent: ttk.Frame) -> None:
        """Create the list frame with tree view."""
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, self.PADDING_STANDARD))

        # Create treeview for utterance list with emotion column
        self.tree = ttk.Treeview(
            list_frame,
            columns=("emotion", "text", "recordings"),
            show="tree headings",
            selectmode="browse",
        )

        # Configure style for sortable column headers
        self.style = ttk.Style()
        self.style.configure(
            "Treeview.Heading",
            relief="flat",
            background="#d9d9d9",
            foreground="black",
            borderwidth=1,
        )
        self.style.map("Treeview.Heading", background=[("active", "#c0c0c0")])

        # Configure columns with sorting
        self.tree.heading("#0", text="Label", command=lambda: self._sort_by("label"))
        self.tree.heading(
            "emotion", text="Emotion", command=lambda: self._sort_by("emotion")
        )
        self.tree.heading("text", text="Text", command=lambda: self._sort_by("text"))
        self.tree.heading(
            "recordings", text="Takes", command=lambda: self._sort_by("recordings")
        )

        # Initially set minimum widths - will be adjusted after data is loaded
        self.tree.column(
            "#0",
            width=self.COLUMN_LIMITS["#0"][0],
            minwidth=self.COLUMN_LIMITS["#0"][0],
        )
        self.tree.column(
            "emotion",
            width=self.COLUMN_LIMITS["emotion"][0],
            minwidth=self.COLUMN_LIMITS["emotion"][0],
        )
        self.tree.column(
            "text",
            width=self.COLUMN_LIMITS["text"][0],
            minwidth=self.COLUMN_LIMITS["text"][0],
        )
        self.tree.column(
            "recordings",
            width=self.COLUMN_LIMITS["recordings"][0],
            minwidth=self.COLUMN_LIMITS["recordings"][0],
        )

        # Scrollbars
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Pack tree and scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

    def _create_button_frame(self, parent: ttk.Frame) -> None:
        """Create the button frame. To be overridden by subclasses."""
        pass

    def _bind_events(self) -> None:
        """Bind common events."""
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())

    def _set_initial_focus(self) -> None:
        """Set initial focus. Can be overridden by subclasses."""
        if self.show_search and hasattr(self, "search_entry"):
            self.search_entry.focus_set()

    def _apply_data(self) -> None:
        """Apply the prepared data to the tree widget."""
        # Set column widths
        self.tree.column("#0", width=self.column_widths["#0"], minwidth=50)
        self.tree.column("emotion", width=self.column_widths["emotion"], minwidth=40)
        self.tree.column("text", width=self.column_widths["text"], minwidth=150)
        self.tree.column(
            "recordings", width=self.column_widths["recordings"], minwidth=40
        )

        # Sort by default column
        self._sort_items()
        self._update_column_headers()  # Show initial sort indicators
        self._apply_filters()

        # Select current utterance if provided
        if self.current_index is not None:
            self.select_utterance_by_index(self.current_index)

    def _sort_items(self) -> None:
        """Sort items based on current sort column and direction.

        For all columns except label/index, use label as secondary sort key
        to ensure stable and predictable ordering when primary values are equal.
        """
        if self.sort_column == "order":
            # Sort by display order (session order), then by label
            self.all_items.sort(
                key=lambda x: (x["display_pos"], x["label"]), reverse=self.sort_reverse
            )
        elif self.sort_column == "index" or self.sort_column == "label":
            # Sort by label/index only
            self.all_items.sort(key=lambda x: x["label"], reverse=self.sort_reverse)
        elif self.sort_column == "emotion":
            # Sort by emotion (empty values last), then by label
            self.all_items.sort(
                key=lambda x: (x["emotion"] == "", x["emotion"], x["label"]),
                reverse=self.sort_reverse,
            )
        elif self.sort_column == "text":
            # Sort by text alphabetically, then by label
            self.all_items.sort(
                key=lambda x: (x["clean_text"].lower(), x["label"]),
                reverse=self.sort_reverse,
            )
        elif self.sort_column == "recordings":
            # Sort by number of recordings, then by label
            self.all_items.sort(
                key=lambda x: (x["takes"], x["label"]), reverse=self.sort_reverse
            )

    def _sort_by(self, column: str) -> None:
        """Sort tree by specified column."""
        # Remember currently selected utterance
        selected_index = None
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            tags = item.get("tags")
            if tags:
                selected_index = int(tags[0])

        # Toggle sort direction if same column, otherwise reset
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        # Sort items
        self._sort_items()

        # Update column headers
        self._update_column_headers()

        # Reapply filters to update display
        self._apply_filters()

        # Restore selection if we had one
        if selected_index is not None:
            self.select_utterance_by_index(selected_index)

    def select_utterance_by_index(self, utterance_index: int) -> None:
        """Select and scroll to a specific utterance by its index.

        Args:
            utterance_index: The index of the utterance to select
        """
        if not self.tree:
            return

        for item_id in self.tree.get_children():
            tags = self.tree.item(item_id, "tags")
            if tags and int(tags[0]) == utterance_index:
                self.tree.selection_set(item_id)
                self.tree.see(item_id)
                self.tree.focus(item_id)
                break

    def _update_column_headers(self) -> None:
        """Update column headers to show sort indicators."""
        # Unicode arrows
        arrow_up = " ▲"
        arrow_down = " ▼"

        columns = {
            "label": ("Label", "#0"),
            "emotion": ("Emotion", "emotion"),
            "text": ("Text", "text"),
            "recordings": ("Takes", "recordings"),
        }

        # Map 'order' and 'index' to 'label' column for display purposes
        display_sort_column = self.sort_column
        if self.sort_column in ("order", "index"):
            display_sort_column = "label"

        # Update all column headers
        for col_key, (display_name, tree_col) in columns.items():
            if col_key == display_sort_column:
                # Active column with prominent arrow and brackets
                arrow = arrow_down if self.sort_reverse else arrow_up
                text = f"[ {display_name}{arrow} ]"
            else:
                # Inactive columns with subtle indicator
                text = f"  {display_name} ◦"

            self.tree.heading(tree_col, text=text)

    def _apply_filters(self) -> None:
        """Apply search and checkbox filters to the list."""
        # Clear current items
        self.tree.delete(*self.tree.get_children())

        search_term = (
            getattr(self, "search_var", tk.StringVar()).get().lower()
            if self.show_search
            else ""
        )
        show_with = (
            getattr(self, "show_with_recordings", tk.BooleanVar(value=True)).get()
            if self.show_filters
            else True
        )
        show_without = (
            getattr(self, "show_without_recordings", tk.BooleanVar(value=True)).get()
            if self.show_filters
            else True
        )

        for item in self.all_items:
            # Apply recording filter
            if self.show_filters:
                has_recordings = item["takes"] > 0
                if has_recordings and not show_with:
                    continue
                if not has_recordings and not show_without:
                    continue

            # Apply search filter
            if search_term:
                # Search in label, emotion, and both full and clean text
                if (
                    search_term not in item["label"].lower()
                    and search_term not in item["full_text"].lower()
                    and search_term not in item["emotion"]
                ):
                    continue

            # Add to tree with emotion column
            self.tree.insert(
                "",
                "end",
                text=item["label"],
                values=(item["emotion"], item["text"], str(item["takes"])),
                tags=(str(item["index"]),),
            )

    def _on_search_changed(self, *args) -> None:
        """Handle search text changes."""
        self._apply_filters()

    def _on_cancel(self) -> None:
        """Handle cancel button."""
        self.dialog.destroy()

    def show(self):
        """Show the dialog and wait for it to close."""
        self.dialog.wait_window()
