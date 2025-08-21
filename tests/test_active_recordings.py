"""Tests for ActiveRecordings cache and sorting management."""

import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock

from revoxx.utils.active_recordings import ActiveRecordings
from revoxx.utils.file_manager import RecordingFileManager


class TestActiveRecordings(unittest.TestCase):
    """Test cases for ActiveRecordings class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for tests
        self.test_dir = Path(tempfile.mkdtemp())

        # Create mock file manager
        self.file_manager = Mock(spec=RecordingFileManager)

        # Create ActiveRecordings instance
        self.active_recordings = ActiveRecordings(self.file_manager)

        # Sample data
        self.labels = ["utt_001", "utt_002", "utt_003", "utt_004"]
        self.utterances = [
            "Hello world",
            "2: How are you?",  # Has emotion
            "Testing recording",
            "1: Another emotional text",  # Has emotion
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test ActiveRecordings initialization."""
        self.assertIsNotNone(self.active_recordings)
        self.assertEqual(self.active_recordings.sort_column, "label")
        self.assertFalse(self.active_recordings.sort_reverse)
        self.assertFalse(self.active_recordings._cache_valid)
        self.assertFalse(self.active_recordings._sort_valid)

    def test_set_data(self):
        """Test setting utterance data."""
        self.active_recordings.set_data(self.labels, self.utterances)

        self.assertEqual(self.active_recordings._labels, self.labels)
        self.assertEqual(self.active_recordings._utterances, self.utterances)
        self.assertFalse(self.active_recordings._cache_valid)
        self.assertFalse(self.active_recordings._sort_valid)

    def test_set_sort(self):
        """Test setting sort criteria."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Change sort column
        self.active_recordings.set_sort("text", False)
        self.assertEqual(self.active_recordings.sort_column, "text")
        self.assertFalse(self.active_recordings.sort_reverse)
        self.assertFalse(self.active_recordings._sort_valid)

        # Change sort direction
        self.active_recordings.set_sort("text", True)
        self.assertTrue(self.active_recordings.sort_reverse)
        self.assertFalse(self.active_recordings._sort_valid)

    def test_extract_take_number_from_filename(self):
        """Test extracting take number from filename."""
        ar = self.active_recordings

        self.assertEqual(ar._extract_take_number("take_001.flac"), 1)
        self.assertEqual(ar._extract_take_number("take_042.wav"), 42)
        self.assertEqual(ar._extract_take_number("take_999.flac"), 999)
        self.assertEqual(ar._extract_take_number("invalid.flac"), 0)
        self.assertEqual(ar._extract_take_number("take_.flac"), 0)
        self.assertEqual(ar._extract_take_number(""), 0)

    def test_extract_emotion(self):
        """Test extracting emotion labels from text."""
        ar = self.active_recordings

        self.assertEqual(ar._extract_emotion("1: Happy text"), "1")
        self.assertEqual(ar._extract_emotion("2: Sad text"), "2")
        self.assertEqual(ar._extract_emotion("42: Some emotion"), "42")
        self.assertEqual(ar._extract_emotion("No emotion here"), "")
        self.assertEqual(ar._extract_emotion(""), "")

    def test_extract_clean_text(self):
        """Test extracting clean text without emotion labels."""
        ar = self.active_recordings

        self.assertEqual(ar._extract_clean_text("1: Happy text"), "Happy text")
        self.assertEqual(ar._extract_clean_text("2: Sad text"), "Sad text")
        self.assertEqual(ar._extract_clean_text("No emotion here"), "No emotion here")
        self.assertEqual(ar._extract_clean_text(""), "")

    def test_get_takes_with_cached_filenames(self):
        """Test getting take count from cached filenames."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Mock file manager to return filenames
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac", "take_002.flac"],
            "utt_002": ["take_001.flac"],
            "utt_003": [],
            "utt_004": ["take_001.flac", "take_003.flac", "take_005.flac"],
        }

        # Test individual takes
        self.assertEqual(self.active_recordings.get_takes("utt_001"), 2)
        self.assertEqual(self.active_recordings.get_takes("utt_002"), 1)
        self.assertEqual(self.active_recordings.get_takes("utt_003"), 0)
        self.assertEqual(self.active_recordings.get_takes("utt_004"), 3)

        # Verify cache was populated
        self.assertTrue(self.active_recordings._cache_valid)
        self.file_manager.scan_all_take_files.assert_called_once()

    def test_get_existing_takes(self):
        """Test getting list of existing take numbers."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Mock file manager to return filenames with gaps
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac", "take_003.flac", "take_007.flac"],
            "utt_002": ["take_002.flac"],
            "utt_003": [],
            "utt_004": ["take_001.flac", "take_005.flac", "take_010.flac"],
        }

        # Test existing takes lists
        self.assertEqual(
            self.active_recordings.get_existing_takes("utt_001"), [1, 3, 7]
        )
        self.assertEqual(self.active_recordings.get_existing_takes("utt_002"), [2])
        self.assertEqual(self.active_recordings.get_existing_takes("utt_003"), [])
        self.assertEqual(
            self.active_recordings.get_existing_takes("utt_004"), [1, 5, 10]
        )
        self.assertEqual(self.active_recordings.get_existing_takes("unknown"), [])

    def test_get_highest_take(self):
        """Test getting the highest take number from cached filenames."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Mock file manager to return filenames with gaps
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac", "take_002.flac", "take_007.flac"],
            "utt_002": ["take_003.flac"],
            "utt_003": [],
            "utt_004": ["take_001.flac", "take_005.flac", "take_010.flac"],
        }

        # Test highest take numbers
        self.assertEqual(self.active_recordings.get_highest_take("utt_001"), 7)
        self.assertEqual(self.active_recordings.get_highest_take("utt_002"), 3)
        self.assertEqual(self.active_recordings.get_highest_take("utt_003"), 0)
        self.assertEqual(self.active_recordings.get_highest_take("utt_004"), 10)
        self.assertEqual(self.active_recordings.get_highest_take("unknown"), 0)

    def test_get_all_takes(self):
        """Test getting all take counts."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Mock file manager
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac", "take_002.flac"],
            "utt_002": ["take_001.flac"],
            "utt_003": [],
            "utt_004": ["take_001.flac", "take_003.flac"],
        }

        all_takes = self.active_recordings.get_all_takes()

        self.assertEqual(all_takes["utt_001"], 2)
        self.assertEqual(all_takes["utt_002"], 1)
        self.assertEqual(all_takes["utt_003"], 0)
        self.assertEqual(all_takes["utt_004"], 2)

    def test_sorting_by_label(self):
        """Test sorting by label."""
        labels = ["utt_003", "utt_001", "utt_004", "utt_002"]
        utterances = ["Third", "First", "Fourth", "Second"]

        self.active_recordings.set_data(labels, utterances)
        self.active_recordings.set_sort("label", False)

        sorted_indices = self.active_recordings.get_sorted_indices()

        # Should be sorted alphabetically by label
        self.assertEqual(
            sorted_indices, [1, 3, 0, 2]
        )  # utt_001, utt_002, utt_003, utt_004

    def test_sorting_by_text(self):
        """Test sorting by text content."""
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("text", False)

        sorted_indices = self.active_recordings.get_sorted_indices()

        # Verify text is sorted alphabetically
        sorted_texts = [self.utterances[i] for i in sorted_indices]
        self.assertEqual(sorted_texts[0].lower(), "1: another emotional text".lower())

    def test_sorting_by_emotion(self):
        """Test sorting by emotion."""
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.set_sort("emotion", False)

        sorted_indices = self.active_recordings.get_sorted_indices()

        # Items with emotions (1, 2) should come before items without
        # utt_004 has emotion "1", utt_002 has emotion "2"
        emotions = [
            self.active_recordings._extract_emotion(self.utterances[i])
            for i in sorted_indices
        ]

        # First two should have emotions
        self.assertIn(emotions[0], ["1", "2"])
        self.assertIn(emotions[1], ["1", "2"])
        # Last two should have no emotion
        self.assertEqual(emotions[2], "")
        self.assertEqual(emotions[3], "")

    def test_sorting_by_recordings(self):
        """Test sorting by number of recordings."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Mock file manager
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac", "take_002.flac", "take_003.flac"],
            "utt_002": ["take_001.flac"],
            "utt_003": [],
            "utt_004": ["take_001.flac", "take_002.flac"],
        }

        self.active_recordings.set_sort("recordings", False)
        sorted_indices = self.active_recordings.get_sorted_indices()

        # Should be sorted by take count: utt_003(0), utt_002(1), utt_004(2), utt_001(3)
        takes = [
            len(self.file_manager.scan_all_take_files.return_value[self.labels[i]])
            for i in sorted_indices
        ]
        self.assertEqual(takes, [0, 1, 2, 3])

    def test_reverse_sorting(self):
        """Test reverse sorting."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Normal sort
        self.active_recordings.set_sort("label", False)
        normal_indices = self.active_recordings.get_sorted_indices()

        # Reverse sort
        self.active_recordings.set_sort("label", True)
        reverse_indices = self.active_recordings.get_sorted_indices()

        # Should be reversed
        self.assertEqual(reverse_indices, list(reversed(normal_indices)))

    def test_display_position_conversion(self):
        """Test converting between actual index and display position."""
        labels = ["utt_003", "utt_001", "utt_004", "utt_002"]
        utterances = ["Third", "First", "Fourth", "Second"]

        self.active_recordings.set_data(labels, utterances)
        self.active_recordings.set_sort("label", False)

        # utt_001 is at actual index 1, but should be display position 1 (first alphabetically)
        self.assertEqual(self.active_recordings.get_display_position(1), 1)

        # utt_003 is at actual index 0, but should be display position 3
        self.assertEqual(self.active_recordings.get_display_position(0), 3)

        # Test reverse conversion
        self.assertEqual(
            self.active_recordings.get_actual_index(1), 1
        )  # First position -> utt_001
        self.assertEqual(
            self.active_recordings.get_actual_index(3), 0
        )  # Third position -> utt_003

    def test_find_next_best_take(self):
        """Test finding the next best take after deletion."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Mock file manager to return specific takes
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": [
                "take_001.flac",
                "take_002.flac",
                "take_003.flac",
                "take_004.flac",
            ],
            "utt_002": ["take_001.flac"],
            "utt_003": [],
            "utt_004": ["take_001.flac", "take_002.flac"],
        }

        # Test deleting take 3 of 4 - should return take 4
        next_take = self.active_recordings.find_next_best_take("utt_001", 3)
        self.assertEqual(next_take, 4)

        # Test deleting take 4 (highest) - should return take 3
        next_take = self.active_recordings.find_next_best_take("utt_001", 4)
        self.assertEqual(next_take, 3)

        # Test deleting take 1 (lowest) - should return take 2
        next_take = self.active_recordings.find_next_best_take("utt_001", 1)
        self.assertEqual(next_take, 2)

        # Test deleting only take - should return 0
        next_take = self.active_recordings.find_next_best_take("utt_002", 1)
        self.assertEqual(next_take, 0)

        # Test deleting middle take 2 of 2 - should return take 1
        next_take = self.active_recordings.find_next_best_take("utt_004", 2)
        self.assertEqual(next_take, 1)

        # Test with no takes at all - should return 0
        next_take = self.active_recordings.find_next_best_take("utt_003", 1)
        self.assertEqual(next_take, 0)

    def test_navigation(self):
        """Test navigation in sorted order."""
        labels = ["utt_003", "utt_001", "utt_004", "utt_002"]
        utterances = ["Third", "First", "Fourth", "Second"]

        self.active_recordings.set_data(labels, utterances)
        self.active_recordings.set_sort("label", False)

        # Start at utt_001 (actual index 1)
        # Next should be utt_002 (actual index 3)
        next_index = self.active_recordings.navigate(1, 1)
        self.assertEqual(next_index, 3)

        # Previous from utt_002 should be utt_001
        prev_index = self.active_recordings.navigate(3, -1)
        self.assertEqual(prev_index, 1)

        # Navigate past boundaries
        self.assertIsNone(self.active_recordings.navigate(2, 1))  # Past last
        self.assertIsNone(self.active_recordings.navigate(1, -1))  # Before first

    def test_iterate_sorted(self):
        """Test iterating in sorted order."""
        labels = ["utt_003", "utt_001", "utt_004", "utt_002"]
        utterances = ["Third", "First", "Fourth", "Second"]

        self.active_recordings.set_data(labels, utterances)
        self.active_recordings.set_sort("label", False)

        sorted_items = list(self.active_recordings.iterate_sorted())

        # Should iterate in alphabetical order by label
        self.assertEqual(len(sorted_items), 4)
        self.assertEqual(sorted_items[0], (1, "utt_001", "First"))
        self.assertEqual(sorted_items[1], (3, "utt_002", "Second"))
        self.assertEqual(sorted_items[2], (0, "utt_003", "Third"))
        self.assertEqual(sorted_items[3], (2, "utt_004", "Fourth"))

    def test_cache_invalidation(self):
        """Test cache invalidation on events."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Set up initial cache
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac"],
            "utt_002": [],
            "utt_003": [],
            "utt_004": [],
        }

        # Access cache to populate it
        self.active_recordings.get_takes("utt_001")
        self.assertTrue(self.active_recordings._cache_valid)

        # Recording completed should invalidate cache
        self.active_recordings.on_recording_completed("utt_001")
        self.assertFalse(self.active_recordings._cache_valid)

        # Re-populate cache
        self.active_recordings.get_takes("utt_001")
        self.assertTrue(self.active_recordings._cache_valid)

        # Recording deleted should invalidate cache
        self.active_recordings.on_recording_deleted("utt_001", 1)
        self.assertFalse(self.active_recordings._cache_valid)

        # Session change should clear everything
        self.active_recordings.set_data(self.labels, self.utterances)
        self.active_recordings.on_session_changed()
        self.assertEqual(self.active_recordings._labels, [])
        self.assertEqual(self.active_recordings._utterances, [])
        self.assertFalse(self.active_recordings._cache_valid)
        self.assertFalse(self.active_recordings._sort_valid)

    def test_sort_invalidation_on_recordings_change(self):
        """Test that sort is invalidated when recordings change if sorting by recordings."""
        self.active_recordings.set_data(self.labels, self.utterances)

        # Set up to sort by recordings
        self.file_manager.scan_all_take_files.return_value = {
            "utt_001": ["take_001.flac"],
            "utt_002": [],
            "utt_003": [],
            "utt_004": [],
        }

        self.active_recordings.set_sort("recordings", False)
        self.active_recordings.get_sorted_indices()  # Populate sort
        self.assertTrue(self.active_recordings._sort_valid)

        # Invalidate cache should also invalidate sort when sorting by recordings
        self.active_recordings.invalidate_cache()
        self.assertFalse(self.active_recordings._cache_valid)
        self.assertFalse(self.active_recordings._sort_valid)

        # But not when sorting by other columns
        self.active_recordings.set_sort("label", False)
        self.active_recordings.get_sorted_indices()  # Populate sort
        self.assertTrue(self.active_recordings._sort_valid)

        self.active_recordings.invalidate_cache()
        self.assertFalse(self.active_recordings._cache_valid)
        self.assertTrue(self.active_recordings._sort_valid)  # Should remain valid


if __name__ == "__main__":
    unittest.main()
