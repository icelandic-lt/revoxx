"""Active recordings cache and sorting management."""

from typing import Dict, List, Optional
import re
from .file_manager import RecordingFileManager


class ActiveRecordings:
    """Manages cached recording data and sorted iteration.

    This class sits between the application and FileManager, providing:
    - Cached access to expensive disk operations (take counts)
    - Sorted iteration based on configurable criteria
    - Automatic cache invalidation on recording events
    - Efficient navigation in sorted order
    """

    def __init__(self, file_manager: RecordingFileManager):
        """Initialize with a file manager.

        Args:
            file_manager: The underlying file manager for disk operations
        """
        self.file_manager = file_manager

        # Cache management
        self._takes_cache: Dict[str, List[str]] = {}  # label -> list of filenames
        self._cache_valid = False

        # Data
        self._labels: List[str] = []
        self._utterances: List[str] = []

        # Sort configuration
        self.sort_column = "label"  # Default: alphabetical by label
        self.sort_reverse = False

        # Sorted indices cache
        self._sorted_indices: Optional[List[int]] = None
        self._sort_valid = False

    def set_data(self, labels: List[str], utterances: List[str]) -> None:
        """Update the utterance data and invalidate caches.

        Args:
            labels: List of utterance labels
            utterances: List of utterance texts
        """
        if labels != self._labels or utterances != self._utterances:
            self._labels = labels
            self._utterances = utterances
            self._cache_valid = False
            self._sort_valid = False

    def set_sort(self, column: str, reverse: bool = False) -> None:
        """Update sort criteria.

        Args:
            column: Sort column ("label", "text", "emotion", "recordings")
            reverse: Whether to reverse sort order
        """
        if column != self.sort_column or reverse != self.sort_reverse:
            self.sort_column = column
            self.sort_reverse = reverse
            self._sort_valid = False

    def invalidate_cache(self) -> None:
        """Invalidate cache after recording/deletion events.

        This should be called whenever:
        - A new recording is created
        - A recording is deleted/moved to trash
        - The recording directory structure changes
        """
        self._cache_valid = False
        # Also invalidate sort if it depends on recordings
        if self.sort_column == "recordings":
            self._sort_valid = False

    def _ensure_cache(self) -> None:
        """Refresh cache from disk if invalid."""
        if not self._cache_valid and self._labels:
            # Cache actual filenames
            self._takes_cache = self.file_manager.scan_all_take_files(self._labels)
            self._cache_valid = True

    @staticmethod
    def _extract_take_number(filename: str) -> int:
        """Extract take number from a filename.

        Args:
            filename: Filename like "take_001.flac"

        Returns:
            Take number, or 0 if invalid
        """
        try:
            # Remove extension and get the number part
            stem = filename.split(".")[0]  # "take_001"
            take_str = stem.split("_")[1]  # "001"
            return int(take_str)
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _extract_emotion(text: str) -> str:
        """Extract emotion label from utterance text.

        Args:
            text: The utterance text

        Returns:
            Emotion label if present, empty string otherwise
        """
        # Pattern to match "N: " at the beginning where N is a natural number
        match = re.match(r"^(\d+):\s*", text)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_clean_text(text: str) -> str:
        """Extract clean text without emotion label.

        Args:
            text: The utterance text

        Returns:
            Text without emotion label prefix
        """
        # Remove emotion prefix if present
        match = re.match(r"^\d+:\s*(.*)$", text)
        return match.group(1) if match else text

    def _get_sort_key(self):
        """Get the sort key function for current sort column.

        For all columns except label, use label as secondary sort key
        to ensure stable and predictable ordering when primary values are equal.
        """
        if self.sort_column == "label":
            return lambda item: item["label"].lower()
        elif self.sort_column == "text":
            # Sort by text, then by label for stable ordering
            return lambda item: (item["text"].lower(), item["label"].lower())
        elif self.sort_column == "emotion":
            # Empty emotions sort last, then by emotion, then by label
            return lambda item: (
                item["emotion"] == "",
                item["emotion"],
                item["label"].lower(),
            )
        elif self.sort_column == "recordings":
            # Sort by takes, then by label for stable ordering
            return lambda item: (item["takes"], item["label"].lower())
        else:
            # Default to index (no sort)
            return lambda item: item["index"]

    def _ensure_sort(self) -> None:
        """Update sort order if needed."""
        if not self._sort_valid:
            items = []

            # Load takes cache only if sorting by recordings
            if self.sort_column == "recordings":
                self._ensure_cache()

            # Build sortable items
            for i, label in enumerate(self._labels):
                item = {
                    "index": i,
                    "label": label,  # Always include label for secondary sorting
                }

                # Add fields as needed based on sort column
                if self.sort_column == "text":
                    item["text"] = self._extract_clean_text(self._utterances[i])
                elif self.sort_column == "emotion":
                    item["emotion"] = self._extract_emotion(self._utterances[i])
                elif self.sort_column == "recordings":
                    filenames = self._takes_cache.get(label, [])
                    item["takes"] = len(filenames)

                items.append(item)

            # Sort and extract indices
            key_func = self._get_sort_key()
            items.sort(key=key_func, reverse=self.sort_reverse)
            self._sorted_indices = [item["index"] for item in items]
            self._sort_valid = True

    # Public query methods

    def get_takes(self, label: str) -> int:
        """Get take count for a specific label (cached).

        Args:
            label: The utterance label

        Returns:
            Number of takes for this label
        """
        self._ensure_cache()
        filenames = self._takes_cache.get(label, [])
        return len(filenames)

    def get_all_takes(self) -> Dict[str, int]:
        """Get all take counts (cached).

        Returns:
            Dictionary mapping labels to take counts
        """
        self._ensure_cache()
        return {label: len(filenames) for label, filenames in self._takes_cache.items()}

    def _get_take_numbers_from_cache(self, label: str) -> List[int]:
        """Extract take numbers from cached filenames for a label.

        Args:
            label: The utterance label

        Returns:
            List of take numbers (unsorted)
        """
        self._ensure_cache()
        filenames = self._takes_cache.get(label, [])
        if not filenames:
            return []

        take_numbers = []
        for filename in filenames:
            take_num = self._extract_take_number(filename)
            if take_num > 0:
                take_numbers.append(take_num)

        return take_numbers

    def get_highest_take(self, label: str) -> int:
        """Get the highest take number for a specific label (cached).

        Args:
            label: The utterance label

        Returns:
            Highest take number for this label (0 if no takes)
        """
        take_numbers = self._get_take_numbers_from_cache(label)
        return max(take_numbers) if take_numbers else 0

    def get_existing_takes(self, label: str) -> List[int]:
        """Get list of existing take numbers for a specific label (cached).

        Args:
            label: The utterance label

        Returns:
            Sorted list of existing take numbers
        """
        take_numbers = self._get_take_numbers_from_cache(label)
        return sorted(take_numbers)

    def find_next_best_take(self, label: str, current_take: int) -> int:
        """Find the next best take after deleting the current one.

        Logic:
        1. Try to find the next higher take number
        2. If not available, find the highest lower take number
        3. Return 0 if no takes remain

        Args:
            label: The utterance label
            current_take: The take number being deleted

        Returns:
            Next best take number, or 0 if no takes remain
        """
        existing_takes = self.get_existing_takes(label)

        # Remove the current take from the list (as it will be deleted)
        remaining_takes = [t for t in existing_takes if t != current_take]

        if not remaining_takes:
            return 0

        # Find next higher take
        higher_takes = [t for t in remaining_takes if t > current_take]
        if higher_takes:
            return min(higher_takes)  # Get the lowest of the higher takes

        # No higher takes, return the highest lower take
        return max(remaining_takes)

    def get_sorted_indices(self) -> List[int]:
        """Get indices in current sort order (cached).

        Returns:
            List of indices in sorted order
        """
        self._ensure_sort()
        return self._sorted_indices.copy() if self._sorted_indices else []

    def get_display_position(self, actual_index: int) -> int:
        """Convert actual index to display position.

        Args:
            actual_index: The actual index in the utterances list

        Returns:
            Display position (1-based) in current sort order
        """
        self._ensure_sort()
        if self._sorted_indices:
            try:
                return self._sorted_indices.index(actual_index) + 1
            except ValueError:
                pass
        return actual_index + 1

    def get_actual_index(self, display_position: int) -> Optional[int]:
        """Convert display position to actual index.

        Args:
            display_position: Display position (1-based)

        Returns:
            Actual index in utterances list, or None if invalid position
        """
        self._ensure_sort()
        if self._sorted_indices:
            pos = display_position - 1
            if 0 <= pos < len(self._sorted_indices):
                return self._sorted_indices[pos]
        return None

    def navigate(self, current_index: int, direction: int) -> Optional[int]:
        """Navigate to next/previous utterance in sorted order.

        Args:
            current_index: Current actual index
            direction: Navigation direction (+1 for next, -1 for previous)

        Returns:
            New actual index, or None if at boundary
        """
        self._ensure_sort()
        if not self._sorted_indices:
            return None

        try:
            current_pos = self._sorted_indices.index(current_index)
        except ValueError:
            # Current index not in sorted list, start from beginning
            return self._sorted_indices[0] if self._sorted_indices else None

        new_pos = current_pos + direction
        if 0 <= new_pos < len(self._sorted_indices):
            return self._sorted_indices[new_pos]

        return None

    def iterate_sorted(self):
        """Generator for iterating in sorted order.

        Yields:
            Tuples of (actual_index, label, utterance)
        """
        self._ensure_sort()
        if self._sorted_indices:
            for idx in self._sorted_indices:
                if 0 <= idx < len(self._labels):
                    yield idx, self._labels[idx], self._utterances[idx]

    # Event notification methods

    def on_recording_completed(self, label: str) -> None:
        """Notify that a recording was completed.

        Args:
            label: Label of the completed recording
        """
        self.invalidate_cache()

    def on_recording_deleted(self, label: str, take: int) -> None:
        """Notify that a recording was deleted.

        Args:
            label: Label of the deleted recording
            take: Take number that was deleted
        """
        self.invalidate_cache()

    def on_session_changed(self) -> None:
        """Notify that the session has changed."""
        self._labels = []
        self._utterances = []
        self._cache_valid = False
        self._sort_valid = False
        self._sorted_indices = None
        self._takes_cache = {}
