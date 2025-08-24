"""Find dialog for searching and navigating to utterances."""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Callable
from .utterance_list_base import UtteranceListDialog, SortDirection


class FindDialog(UtteranceListDialog):
    """Dialog for finding specific utterances with search and filter capabilities."""

    def __init__(
        self,
        parent: tk.Widget,
        utterances: List[str],
        labels: List[str],
        file_manager,
        current_index: int,
        on_find: Callable[[int], None],
        utterance_order: List[int],
        current_sort_column: str = "label",
        current_sort_direction: SortDirection = SortDirection.UP,
    ):
        """Initialize the find dialog.

        Args:
            parent: Parent widget
            utterances: List of utterance texts
            labels: List of utterance labels
            file_manager: RecordingFileManager for scanning files
            current_index: Currently selected utterance index
            on_find: Callback function when finding an utterance
            utterance_order: Display order for utterances (list of indices)
            current_sort_column: Current sort column from session
            current_sort_direction: Current sort direction from session
        """
        self.current_index = current_index
        self.on_find = on_find
        self.result_index = None

        # Initialize base class with current sort settings
        super().__init__(
            parent=parent,
            utterances=utterances,
            labels=labels,
            file_manager=file_manager,
            title="Find Utterance",
            show_search=True,
            show_filters=True,
            utterance_order=utterance_order,
            default_sort=current_sort_column,
            default_sort_direction=current_sort_direction,
            current_index=current_index,
        )

    def _create_button_frame(self, parent: ttk.Frame) -> None:
        """Create the button frame with Find and Cancel buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Find", command=self._on_find).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(
            side=tk.RIGHT
        )

    def _bind_events(self) -> None:
        """Bind dialog events."""
        super()._bind_events()
        self.tree.bind("<Double-Button-1>", lambda e: self._on_find())
        self.tree.bind("<Return>", lambda e: self._on_find())

    def _on_find(self) -> None:
        """Handle find button or double-click."""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            tags = item["tags"]
            if tags:
                index = int(tags[0])
                self.result_index = index
                self.dialog.destroy()
                if self.on_find:
                    self.on_find(index)

    def show(self) -> Optional[int]:
        """Show the dialog and return the selected index.

        Returns:
            Selected utterance index or None if cancelled
        """
        super().show()
        return self.result_index
