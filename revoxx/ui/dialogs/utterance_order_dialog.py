"""Dialog for managing utterance order within a session."""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional
from .utterance_list_base import UtteranceListDialog, SortDirection


class UtteranceOrderDialog(UtteranceListDialog):
    """Dialog for reordering utterances within a session."""

    def __init__(
        self,
        parent: tk.Widget,
        utterances: List[str],
        labels: List[str],
        file_manager,
        current_order: Optional[List[int]] = None,
        current_sort_column: str = "label",
        current_sort_direction: SortDirection = SortDirection.UP,
        current_index: Optional[int] = None,
    ):
        """Initialize the utterance order dialog.

        Args:
            parent: Parent widget
            utterances: List of utterance texts
            labels: List of utterance labels
            file_manager: RecordingFileManager for scanning files
            current_order: Current order as list of indices, None for original order
            current_sort_column: Current sort column from session
            current_sort_direction: Current sort direction from session
            current_index: Currently selected utterance index to highlight
        """
        self.current_order = current_order or list(range(len(utterances)))
        self.original_order = list(self.current_order)  # Keep copy of original
        self.result_order = None
        self.current_index = current_index

        # Initialize base class with current sort settings
        super().__init__(
            parent=parent,
            utterances=utterances,
            labels=labels,
            file_manager=file_manager,
            title="Utterance Order",
            show_search=False,  # No search needed
            show_filters=False,  # No filters needed for ordering
            utterance_order=self.current_order,
            default_sort=current_sort_column,
            default_sort_direction=current_sort_direction,
            current_index=current_index,
        )

    def _create_info_frame(self, parent: ttk.Frame) -> None:
        """Add info text at the top of the dialog."""
        info_label = ttk.Label(
            parent,
            text="Click column headers to sort. The current sort order will be saved as the new utterance order.",
            wraplength=600,
        )
        info_label.pack(fill=tk.X, pady=(0, 10))

    def _create_button_frame(self, parent: ttk.Frame) -> None:
        """Create the button frame with action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)

        # Right side: Action buttons
        action_frame = ttk.Frame(button_frame)
        action_frame.pack(side=tk.RIGHT)

        self.reset_btn = ttk.Button(
            action_frame, text="Reset", command=self._reset_order
        )
        self.reset_btn.pack(side=tk.RIGHT, padx=(5, 5))

        ttk.Button(action_frame, text="Save Order", command=self._on_save).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        ttk.Button(action_frame, text="Cancel", command=self._on_cancel).pack(
            side=tk.RIGHT
        )

    def _reset_order(self) -> None:
        """Reset to original order."""
        result = messagebox.askyesno(
            "Reset Order",
            "Reset to original order? This will undo any sorting.",
            parent=self.dialog,
        )

        if result:
            # Reset to original order sorting
            self.sort_column = "order"
            self.sort_reverse = False

            # Re-sort and update display
            self._sort_items()
            self._update_column_headers()
            self._apply_filters()

    def _on_save(self) -> None:
        """Handle save button - save the current sort settings."""
        # Return the current sort column and direction
        self.result_order = (self.sort_column, self.sort_reverse)
        self.dialog.destroy()

    def show(self):
        """Show the dialog and return the sort settings.

        Returns:
            Tuple of (sort_column, sort_reverse), or None if cancelled
        """
        super().show()
        return self.result_order
