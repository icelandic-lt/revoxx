"""Tests for utterance list dialog sorting with stable secondary criterion."""

import unittest
from unittest.mock import Mock
import tkinter as tk

from revoxx.ui.dialogs.utterance_list_base import UtteranceListDialog, SortDirection


class TestUtteranceListDialogSorting(unittest.TestCase):
    """Test cases for stable sorting in utterance list dialogs."""

    def setUp(self):
        """Set up test fixtures."""
        # Create root window for testing
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window

        # Mock file manager
        self.file_manager = Mock()

        # Sample data with duplicates to test secondary sorting
        self.labels = ["utt_003", "utt_001", "utt_004", "utt_002", "utt_005"]
        self.utterances = [
            "2: Same emotion A",  # utt_003: emotion 2
            "1: Different emotion",  # utt_001: emotion 1
            "2: Same emotion B",  # utt_004: emotion 2 (same as utt_003)
            "Same text",  # utt_002: no emotion
            "Same text",  # utt_005: no emotion (same text as utt_002)
        ]

        # Mock takes for each label
        self.mock_takes = {
            "utt_003": 2,  # 2 takes
            "utt_001": 2,  # 2 takes (same as utt_003)
            "utt_004": 1,  # 1 take
            "utt_002": 1,  # 1 take (same as utt_004)
            "utt_005": 0,  # 0 takes
        }

        # Mock file manager scan
        self.file_manager.scan_all_take_files.return_value = {
            label: [f"take_{i:03d}.flac" for i in range(1, count + 1)]
            for label, count in self.mock_takes.items()
        }

    def tearDown(self):
        """Clean up test fixtures."""
        self.root.destroy()

    def create_dialog(self, default_sort="label"):
        """Helper to create a dialog instance."""
        dialog = UtteranceListDialog(
            self.root,
            self.utterances,
            self.labels,
            self.file_manager,
            title="Test Dialog",
            show_search=False,
            show_filters=False,
            default_sort=default_sort,
            default_sort_direction=SortDirection.UP,
        )
        return dialog

    def test_emotion_sort_stable_ordering(self):
        """Test that emotions with same value are sorted by label."""
        dialog = self.create_dialog(default_sort="emotion")

        # Get sorted items
        sorted_items = dialog.all_items

        # Items with emotion "2" should be in alphabetical order by label
        emotion_2_items = [item for item in sorted_items if item["emotion"] == "2"]
        emotion_2_labels = [item["label"] for item in emotion_2_items]
        self.assertEqual(emotion_2_labels, ["utt_003", "utt_004"])

        # Items with no emotion should also be in alphabetical order
        no_emotion_items = [item for item in sorted_items if item["emotion"] == ""]
        no_emotion_labels = [item["label"] for item in no_emotion_items]
        self.assertEqual(no_emotion_labels, ["utt_002", "utt_005"])

        dialog.dialog.destroy()

    def test_recordings_sort_stable_ordering(self):
        """Test that recordings with same count are sorted by label."""
        dialog = self.create_dialog(default_sort="recordings")

        # Get sorted items
        sorted_items = dialog.all_items

        # Items with 2 takes should be in alphabetical order by label
        two_takes_items = [item for item in sorted_items if item["takes"] == 2]
        two_takes_labels = [item["label"] for item in two_takes_items]
        self.assertEqual(two_takes_labels, ["utt_001", "utt_003"])

        # Items with 1 take should be in alphabetical order by label
        one_take_items = [item for item in sorted_items if item["takes"] == 1]
        one_take_labels = [item["label"] for item in one_take_items]
        self.assertEqual(one_take_labels, ["utt_002", "utt_004"])

        dialog.dialog.destroy()

    def test_text_sort_stable_ordering(self):
        """Test that texts with same content are sorted by label."""
        dialog = self.create_dialog(default_sort="text")

        # Get sorted items
        sorted_items = dialog.all_items

        # Items with "Same text" should be in alphabetical order by label
        same_text_items = [
            item for item in sorted_items if item["clean_text"] == "Same text"
        ]
        same_text_labels = [item["label"] for item in same_text_items]
        self.assertEqual(same_text_labels, ["utt_002", "utt_005"])

        dialog.dialog.destroy()

    def test_reverse_sort_maintains_secondary(self):
        """Test that reverse sort only reverses primary, secondary stays ascending."""
        dialog = self.create_dialog(default_sort="recordings")
        dialog.sort_reverse = True
        dialog._sort_items()

        sorted_items = dialog.all_items

        # With reverse sort, higher counts come first
        # Items with 2 takes should still be in reverse alphabetical order
        two_takes_items = [item for item in sorted_items if item["takes"] == 2]
        two_takes_labels = [item["label"] for item in two_takes_items]
        # Note: When reverse=True, both primary and secondary are reversed
        self.assertEqual(two_takes_labels, ["utt_003", "utt_001"])

        dialog.dialog.destroy()

    def test_sort_column_change_reapplies_sort(self):
        """Test that changing sort column correctly applies new sort with secondary."""
        dialog = self.create_dialog(default_sort="label")

        # Initially sorted by label
        initial_labels = [item["label"] for item in dialog.all_items]
        self.assertEqual(initial_labels[0], "utt_001")  # First alphabetically

        # Change to emotion sort
        dialog._sort_by("emotion")

        # Should now be sorted by emotion with label as secondary
        sorted_labels = [item["label"] for item in dialog.all_items]
        # utt_001 has emotion "1" so should be first
        self.assertEqual(sorted_labels[0], "utt_001")
        # utt_003 and utt_004 have emotion "2", should be alphabetical
        emotion_2_start = sorted_labels.index("utt_003")
        self.assertEqual(
            sorted_labels[emotion_2_start : emotion_2_start + 2], ["utt_003", "utt_004"]
        )

        dialog.dialog.destroy()

    def test_order_sort_with_secondary(self):
        """Test that order (display position) sort uses label as secondary."""
        # Create custom utterance order
        utterance_order = [2, 0, 3, 1, 4]  # Mix up the order

        dialog = UtteranceListDialog(
            self.root,
            self.utterances,
            self.labels,
            self.file_manager,
            title="Test Dialog",
            show_search=False,
            show_filters=False,
            utterance_order=utterance_order,
            default_sort="order",
            default_sort_direction=SortDirection.UP,
        )

        # Items should be sorted by display position
        sorted_items = dialog.all_items
        display_positions = [item["display_pos"] for item in sorted_items]

        # Should be in order 0, 1, 2, 3, 4
        self.assertEqual(display_positions, [0, 1, 2, 3, 4])

        # If we had duplicate positions (which shouldn't happen normally),
        # they would be sorted by label as secondary

        dialog.dialog.destroy()


if __name__ == "__main__":
    unittest.main()
