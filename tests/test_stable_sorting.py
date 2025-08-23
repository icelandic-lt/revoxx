"""Tests for stable sorting with secondary label criterion."""

import unittest
from unittest.mock import Mock

from revoxx.utils.active_recordings import ActiveRecordings
from revoxx.utils.file_manager import RecordingFileManager


class TestStableSorting(unittest.TestCase):
    """Test cases for stable sorting with label as secondary criterion."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock file manager
        self.file_manager = Mock(spec=RecordingFileManager)

        # Create ActiveRecordings instance
        self.active_recordings = ActiveRecordings(self.file_manager)

        # Sample data with duplicate values to test secondary sorting
        self.labels = ["utt_003", "utt_001", "utt_004", "utt_002", "utt_005"]
        self.utterances = [
            "2: Same emotion A",  # utt_003: emotion 2
            "1: Different emotion",  # utt_001: emotion 1
            "2: Same emotion B",  # utt_004: emotion 2 (same as utt_003)
            "Same text",  # utt_002: no emotion
            "Same text",  # utt_005: no emotion (same text as utt_002)
        ]

        # Mock take counts for recordings sort test
        self.mock_takes = {
            "utt_003": ["take_001.flac", "take_002.flac"],  # 2 takes
            "utt_001": ["take_001.flac", "take_002.flac"],  # 2 takes (same as utt_003)
            "utt_004": ["take_001.flac"],  # 1 take
            "utt_002": ["take_001.flac"],  # 1 take (same as utt_004)
            "utt_005": [],  # 0 takes
        }

    def test_emotion_sort_with_secondary_label(self):
        """Test that emotion sort uses label as secondary criterion."""
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("emotion", False)

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [self.labels[i] for i in sorted_indices]

        # Expected order:
        # 1. Emotion "1": utt_001
        # 2. Emotion "2": utt_003, utt_004 (alphabetical by label)
        # 3. No emotion (empty): utt_002, utt_005 (alphabetical by label)
        expected = ["utt_001", "utt_003", "utt_004", "utt_002", "utt_005"]
        self.assertEqual(sorted_labels, expected)

    def test_emotion_sort_reverse_with_secondary_label(self):
        """Test that reverse emotion sort still uses label as secondary (non-reversed)."""
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("emotion", True)  # Reverse

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [self.labels[i] for i in sorted_indices]

        # Expected order (reverse primary, but secondary still ascending):
        # 1. No emotion (empty): utt_005, utt_002 (reverse alphabetical)
        # 2. Emotion "2": utt_004, utt_003 (reverse alphabetical)
        # 3. Emotion "1": utt_001
        expected = ["utt_005", "utt_002", "utt_004", "utt_003", "utt_001"]
        self.assertEqual(sorted_labels, expected)

    def test_text_sort_with_secondary_label(self):
        """Test that text sort uses label as secondary criterion."""
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("text", False)

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [self.labels[i] for i in sorted_indices]

        # Expected order (alphabetical by clean text, then by label):
        # 1. "Different emotion": utt_001
        # 2. "Same emotion A": utt_003
        # 3. "Same emotion B": utt_004
        # 4. "Same text": utt_002, utt_005 (alphabetical by label)
        expected = ["utt_001", "utt_003", "utt_004", "utt_002", "utt_005"]
        self.assertEqual(sorted_labels, expected)

    def test_recordings_sort_with_secondary_label(self):
        """Test that recordings sort uses label as secondary criterion."""
        # Mock the file scan
        self.file_manager.scan_all_take_files.return_value = self.mock_takes

        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("recordings", False)

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [self.labels[i] for i in sorted_indices]

        # Expected order (by take count, then by label):
        # 1. 0 takes: utt_005
        # 2. 1 take: utt_002, utt_004 (alphabetical by label)
        # 3. 2 takes: utt_001, utt_003 (alphabetical by label)
        expected = ["utt_005", "utt_002", "utt_004", "utt_001", "utt_003"]
        self.assertEqual(sorted_labels, expected)

    def test_recordings_sort_reverse_with_secondary_label(self):
        """Test that reverse recordings sort uses label as secondary criterion."""
        # Mock the file scan
        self.file_manager.scan_all_take_files.return_value = self.mock_takes

        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("recordings", True)  # Reverse

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [self.labels[i] for i in sorted_indices]

        # Expected order (reverse by take count, then reverse by label):
        # 1. 2 takes: utt_003, utt_001 (reverse alphabetical)
        # 2. 1 take: utt_004, utt_002 (reverse alphabetical)
        # 3. 0 takes: utt_005
        expected = ["utt_003", "utt_001", "utt_004", "utt_002", "utt_005"]
        self.assertEqual(sorted_labels, expected)

    def test_label_sort_no_secondary(self):
        """Test that label sort does not use secondary criterion (only primary)."""
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("label", False)

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [self.labels[i] for i in sorted_indices]

        # Expected order (alphabetical by label only):
        expected = ["utt_001", "utt_002", "utt_003", "utt_004", "utt_005"]
        self.assertEqual(sorted_labels, expected)

    def test_navigation_respects_stable_sort(self):
        """Test that navigation follows the stable sorted order."""
        # Mock the file scan for recordings-based sort
        self.file_manager.scan_all_take_files.return_value = self.mock_takes

        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("recordings", False)

        # Start at utt_002 (index 3 in original, has 1 take)
        current_index = 3  # utt_002

        # Navigate forward - should go to utt_004 (also 1 take, next alphabetically)
        next_index = self.active_recordings.navigate(current_index, 1)
        self.assertEqual(self.labels[next_index], "utt_004")

        # Navigate forward again - should go to utt_001 (2 takes)
        next_index = self.active_recordings.navigate(next_index, 1)
        self.assertEqual(self.labels[next_index], "utt_001")

        # Navigate backward - should go back to utt_004
        prev_index = self.active_recordings.navigate(next_index, -1)
        self.assertEqual(self.labels[prev_index], "utt_004")

    def test_empty_emotion_sorts_last(self):
        """Test that empty emotions always sort last, regardless of label."""
        labels = ["z_last", "a_first", "m_middle"]
        utterances = [
            "No emotion Z",  # z_last: no emotion
            "1: Has emotion A",  # a_first: emotion 1
            "No emotion M",  # m_middle: no emotion
        ]

        self.active_recordings.set_data(labels, utterances)
        self.active_recordings.set_sort("emotion", False)

        sorted_indices = self.active_recordings.get_sorted_indices()
        sorted_labels = [labels[i] for i in sorted_indices]

        # Expected: emotion first, then no-emotion sorted by label
        expected = ["a_first", "m_middle", "z_last"]
        self.assertEqual(sorted_labels, expected)


if __name__ == "__main__":
    unittest.main()
