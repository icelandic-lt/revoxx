"""Tests for dataset exporter functionality."""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import soundfile as sf
import numpy as np

from revoxx.dataset.exporter import DatasetExporter


class TestDatasetExporter(unittest.TestCase):
    """Test cases for DatasetExporter."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True)

        # Create test session directories
        self.session1_dir = Path(self.temp_dir) / "session1.revoxx"
        self.session2_dir = Path(self.temp_dir) / "session2.revoxx"
        self.session3_dir = Path(self.temp_dir) / "session3.revoxx"

        for session_dir in [self.session1_dir, self.session2_dir, self.session3_dir]:
            session_dir.mkdir(parents=True)
            (session_dir / "recordings").mkdir()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_session(
        self,
        session_dir: Path,
        speaker_name: str,
        emotion: str,
        utterances: list,
        intensity_prefix: bool = False,
    ):
        """Create a test session with metadata and recordings."""
        # Create session.json
        session_data = {
            "speaker": {
                "name": speaker_name,
                "emotion": emotion,
                "age": 30,
                "gender": "F",
            },
            "audio_config": {"sample_rate": 48000, "bit_depth": 24, "channels": 1},
        }

        with open(session_dir / "session.json", "w") as f:
            json.dump(session_data, f)

        # Create script.txt with utterances
        script_lines = []
        for utt_id, text in utterances:
            if intensity_prefix:
                # Add intensity prefix for non-neutral emotions
                if emotion != "neutral":
                    text = f"3: {text}"
            script_lines.append(f'({utt_id} "{text}")')

        with open(session_dir / "script.txt", "w") as f:
            f.write("\n".join(script_lines))

        # Create actual recording files
        for utt_id, _ in utterances:
            utt_dir = session_dir / "recordings" / utt_id
            utt_dir.mkdir()

            # Create dummy audio file (take_001.flac)
            audio_data = np.random.randn(48000).astype(np.float32) * 0.1
            audio_file = utt_dir / "take_001.flac"
            sf.write(str(audio_file), audio_data, 48000)

            # Create additional takes for some utterances
            if utt_id == "utt_001":
                # Create take_002.flac (should use this as highest)
                sf.write(str(utt_dir / "take_002.flac"), audio_data, 48000)

    def test_export_single_emotion_session(self):
        """Test exporting a single session with one emotion."""
        # Create test session
        utterances = [
            ("utt_001", "This is the first utterance"),
            ("utt_002", "This is the second utterance"),
            ("utt_003", "This is the third utterance"),
        ]
        self._create_test_session(
            self.session1_dir, "Test Speaker", "happy", utterances
        )

        # Export
        exporter = DatasetExporter(self.output_dir, audio_format="flac")
        output_paths, stats = exporter.export_sessions([self.session1_dir])

        # Verify output structure
        self.assertEqual(len(output_paths), 1)
        dataset_dir = output_paths[0]
        self.assertTrue(dataset_dir.exists())
        self.assertEqual(dataset_dir.name, "test_speaker")

        # Check emotion directory
        emotion_dir = dataset_dir / "happy"
        self.assertTrue(emotion_dir.exists())

        # Check audio files are correctly named
        audio_files = list(emotion_dir.glob("*.flac"))
        self.assertEqual(len(audio_files), 3)
        expected_files = [
            "test_speaker_happy_001.flac",
            "test_speaker_happy_002.flac",
            "test_speaker_happy_003.flac",
        ]
        actual_files = sorted([f.name for f in audio_files])
        self.assertEqual(actual_files, expected_files)

        # Check index.tsv
        index_file = dataset_dir / "index.tsv"
        self.assertTrue(index_file.exists())

        with open(index_file, "r") as f:
            index_lines = f.readlines()

        self.assertEqual(len(index_lines), 3)

        # Verify index format (with intensity column by default)
        # Collect all texts from index to verify they're all there
        all_texts = []
        for line in index_lines:
            parts = line.strip().split("\t")
            self.assertEqual(
                len(parts), 5
            )  # filename, speaker, emotion, intensity, text
            # Check common fields
            self.assertTrue(parts[0].startswith("test_speaker_happy_"))
            self.assertEqual(parts[1], "test_speaker")
            self.assertEqual(parts[2], "happy")
            self.assertEqual(parts[3], "0")  # Default intensity
            all_texts.append(parts[4])

        # Check all utterances are present (order not guaranteed)
        self.assertSetEqual(
            set(all_texts),
            {
                "This is the first utterance",
                "This is the second utterance",
                "This is the third utterance",
            },
        )

        # Check README exists
        readme_file = dataset_dir / "README.txt"
        self.assertTrue(readme_file.exists())

    def test_export_multiple_emotions(self):
        """Test exporting sessions with multiple emotions from same speaker."""
        # Create sessions with different emotions
        self._create_test_session(
            self.session1_dir,
            "Anna",
            "happy",
            [("utt_001", "Happy utterance"), ("utt_002", "Another happy one")],
        )

        self._create_test_session(
            self.session2_dir,
            "Anna",
            "sad",
            [("utt_003", "Sad utterance"), ("utt_004", "Another sad one")],
        )

        self._create_test_session(
            self.session3_dir, "Anna", "neutral", [("utt_005", "Neutral utterance")]
        )

        # Export
        exporter = DatasetExporter(self.output_dir)
        output_paths, stats = exporter.export_sessions(
            [self.session1_dir, self.session2_dir, self.session3_dir]
        )

        # Verify single dataset with multiple emotion directories
        self.assertEqual(len(output_paths), 1)
        dataset_dir = output_paths[0]

        # Check all emotion directories exist
        self.assertTrue((dataset_dir / "happy").exists())
        self.assertTrue((dataset_dir / "sad").exists())
        self.assertTrue((dataset_dir / "neutral").exists())

        # Verify files in each emotion directory
        happy_files = list((dataset_dir / "happy").glob("*.flac"))
        sad_files = list((dataset_dir / "sad").glob("*.flac"))
        neutral_files = list((dataset_dir / "neutral").glob("*.flac"))

        self.assertEqual(len(happy_files), 2)
        self.assertEqual(len(sad_files), 2)
        self.assertEqual(len(neutral_files), 1)

        # Check index contains all files
        with open(dataset_dir / "index.tsv", "r") as f:
            index_lines = f.readlines()

        self.assertEqual(len(index_lines), 5)

        # Verify statistics
        self.assertEqual(stats["total_utterances"], 5)
        self.assertEqual(stats["sessions_processed"], 3)
        self.assertEqual(stats["datasets_created"], 1)
        self.assertEqual(len(stats["speakers"]), 1)

        speaker_stats = stats["speakers"][0]
        self.assertEqual(speaker_stats["name"], "anna")
        self.assertSetEqual(set(speaker_stats["emotions"]), {"happy", "sad", "neutral"})

    def test_intensity_handling(self):
        """Test that intensity levels are correctly extracted and handled."""
        # Create session with intensity prefixes
        utterances = [
            ("utt_001", "1: Low intensity"),
            ("utt_002", "3: Medium intensity"),
            ("utt_003", "5: High intensity"),
            ("utt_004", "No intensity prefix"),
        ]

        self._create_test_session(
            self.session1_dir,
            "Test Speaker",
            "angry",
            utterances,
            intensity_prefix=False,  # We're adding prefixes manually
        )

        # Export with intensity
        exporter = DatasetExporter(self.output_dir, include_intensity=True)
        output_paths, _ = exporter.export_sessions([self.session1_dir])

        dataset_dir = output_paths[0]
        with open(dataset_dir / "index.tsv", "r") as f:
            index_lines = f.readlines()

        # Check intensity values - create map of text to intensity
        intensity_map = {}
        for line in index_lines:
            parts = line.strip().split("\t")
            text = parts[4]
            intensity = parts[3]
            intensity_map[text] = intensity

        # Verify intensities are correctly extracted
        self.assertEqual(intensity_map["Low intensity"], "1")
        self.assertEqual(intensity_map["Medium intensity"], "3")
        self.assertEqual(intensity_map["High intensity"], "5")
        self.assertEqual(intensity_map["No intensity prefix"], "0")

    def test_neutral_emotion_zero_intensity(self):
        """Test that neutral emotion always gets intensity 0."""
        # Create neutral session with intensity prefixes (should be ignored)
        utterances = [
            ("utt_001", "3: Should be zero intensity"),
            ("utt_002", "5: Also zero intensity"),
        ]

        self._create_test_session(
            self.session1_dir,
            "Test Speaker",
            "neutral",
            utterances,
            intensity_prefix=False,
        )

        # Export
        exporter = DatasetExporter(self.output_dir, zero_intensity_emotions=["neutral"])
        output_paths, _ = exporter.export_sessions([self.session1_dir])

        dataset_dir = output_paths[0]
        with open(dataset_dir / "index.tsv", "r") as f:
            index_lines = f.readlines()

        # Both should have intensity 0 despite prefixes
        for line in index_lines:
            parts = line.strip().split("\t")
            self.assertEqual(parts[3], "0")  # Intensity should be 0
            # Text should still be cleaned (no prefix)
            self.assertNotIn(":", parts[4])

    def test_multiple_speakers_separate_datasets(self):
        """Test that different speakers create separate datasets."""
        # Create sessions for different speakers
        self._create_test_session(
            self.session1_dir,
            "Speaker A",
            "happy",
            [("utt_001", "Speaker A utterance")],
        )

        self._create_test_session(
            self.session2_dir,
            "Speaker B",
            "happy",
            [("utt_002", "Speaker B utterance")],
        )

        # Export
        exporter = DatasetExporter(self.output_dir)
        output_paths, stats = exporter.export_sessions(
            [self.session1_dir, self.session2_dir]
        )

        # Should create two separate datasets
        self.assertEqual(len(output_paths), 2)
        self.assertEqual(stats["datasets_created"], 2)

        # Check dataset names
        dataset_names = sorted([p.name for p in output_paths])
        self.assertEqual(dataset_names, ["speaker_a", "speaker_b"])

        # Each should have its own index.tsv
        for dataset_dir in output_paths:
            self.assertTrue((dataset_dir / "index.tsv").exists())
            self.assertTrue((dataset_dir / "README.txt").exists())

    def test_highest_take_selection(self):
        """Test that the highest take number is selected for export."""
        session_dir = self.session1_dir
        utterances = [("utt_001", "Test utterance")]

        # Create session
        self._create_test_session(session_dir, "Test Speaker", "happy", utterances)

        # Add multiple takes for utt_001
        utt_dir = session_dir / "recordings" / "utt_001"
        audio_data = np.random.randn(48000).astype(np.float32) * 0.1

        # Create takes 1, 3, 5 (missing 2 and 4)
        sf.write(str(utt_dir / "take_001.flac"), audio_data, 48000)
        sf.write(str(utt_dir / "take_003.flac"), audio_data * 1.1, 48000)
        sf.write(str(utt_dir / "take_005.flac"), audio_data * 1.2, 48000)

        # Export
        exporter = DatasetExporter(self.output_dir)
        output_paths, _ = exporter.export_sessions([session_dir])

        # The exported file should be from take_005 (highest)
        dataset_dir = output_paths[0]
        emotion_dir = dataset_dir / "happy"
        audio_files = list(emotion_dir.glob("*.flac"))

        self.assertEqual(len(audio_files), 1)

        # Verify it used take_005 by checking file size (different data)
        exported_file = audio_files[0]
        exported_data, _ = sf.read(str(exported_file))

        # Compare with take_005 data
        take_005_data, _ = sf.read(str(utt_dir / "take_005.flac"))
        np.testing.assert_array_almost_equal(exported_data, take_005_data)

    def test_missing_recordings_handling(self):
        """Test handling of missing recording files."""
        # Create session with script but missing recordings
        session_dir = self.session1_dir
        utterances = [
            ("utt_001", "Has recording"),
            ("utt_002", "Missing recording"),
            ("utt_003", "Also has recording"),
        ]

        self._create_test_session(session_dir, "Test Speaker", "happy", utterances)

        # Remove utt_002 recording
        shutil.rmtree(session_dir / "recordings" / "utt_002")

        # Export
        exporter = DatasetExporter(self.output_dir)
        output_paths, stats = exporter.export_sessions([session_dir])

        # Check statistics - only utterances with recordings are counted
        self.assertEqual(stats["total_utterances"], 2)
        self.assertEqual(stats["missing_recordings"], 0)

        # Check only 2 files exported
        dataset_dir = output_paths[0]
        audio_files = list((dataset_dir / "happy").glob("*.flac"))
        self.assertEqual(len(audio_files), 2)

        # Check index only has 2 entries
        with open(dataset_dir / "index.tsv", "r") as f:
            index_lines = f.readlines()
        self.assertEqual(len(index_lines), 2)

    def test_export_without_intensity_column(self):
        """Test exporting without intensity column in index.tsv."""
        # Create test session
        utterances = [
            ("utt_001", "3: With intensity prefix"),
            ("utt_002", "Without prefix"),
        ]

        self._create_test_session(
            self.session1_dir,
            "Test Speaker",
            "happy",
            utterances,
            intensity_prefix=False,
        )

        # Export without intensity
        exporter = DatasetExporter(self.output_dir, include_intensity=False)
        output_paths, _ = exporter.export_sessions([self.session1_dir])

        dataset_dir = output_paths[0]
        with open(dataset_dir / "index.tsv", "r") as f:
            index_lines = f.readlines()

        # Check format without intensity column
        all_texts = []
        for line in index_lines:
            parts = line.strip().split("\t")
            self.assertEqual(len(parts), 4)  # filename, speaker, emotion, text
            self.assertTrue(parts[0].startswith("test_speaker_happy_"))
            self.assertEqual(parts[1], "test_speaker")
            self.assertEqual(parts[2], "happy")
            # Text has intensity prefix removed
            all_texts.append(parts[3])

        # Check all texts are present without intensity prefixes
        self.assertSetEqual(set(all_texts), {"With intensity prefix", "Without prefix"})

    def test_audio_format_conversion(self):
        """Test converting FLAC to WAV during export."""
        # Create test session with FLAC recordings
        utterances = [("utt_001", "Test utterance")]
        self._create_test_session(
            self.session1_dir, "Test Speaker", "happy", utterances
        )

        # Export as WAV
        exporter = DatasetExporter(self.output_dir, audio_format="wav")
        output_paths, _ = exporter.export_sessions([self.session1_dir])

        # Check output is WAV
        dataset_dir = output_paths[0]
        audio_files = list((dataset_dir / "happy").glob("*.wav"))
        self.assertEqual(len(audio_files), 1)
        self.assertTrue(audio_files[0].suffix == ".wav")

        # Verify file is valid WAV
        data, sr = sf.read(str(audio_files[0]))
        self.assertEqual(sr, 48000)

    def test_dataset_name_override(self):
        """Test overriding dataset name."""
        # Create single speaker session
        self._create_test_session(
            self.session1_dir, "Original Name", "happy", [("utt_001", "Test")]
        )

        # Export with custom name
        exporter = DatasetExporter(self.output_dir)
        output_paths, _ = exporter.export_sessions(
            [self.session1_dir], dataset_name="custom_dataset"
        )

        # Check dataset uses custom name
        self.assertEqual(output_paths[0].name, "custom_dataset")

        # Check files use custom name
        audio_files = list((output_paths[0] / "happy").glob("*.flac"))
        self.assertTrue(audio_files[0].name.startswith("custom_dataset_"))

    def test_readme_content(self):
        """Test that README contains correct information."""
        # Create test session
        self._create_test_session(
            self.session1_dir, "Test Speaker", "happy", [("utt_001", "Test")]
        )

        # Export with intensity
        exporter = DatasetExporter(
            self.output_dir, audio_format="flac", include_intensity=True
        )
        output_paths, _ = exporter.export_sessions([self.session1_dir])

        # Check README content
        readme_path = output_paths[0] / "README.txt"
        with open(readme_path, "r") as f:
            readme_content = f.read()

        # Verify key information is present
        self.assertIn("FLAC", readme_content)
        self.assertIn("48000", readme_content)  # Sample rate
        self.assertIn("24", readme_content)  # Bit depth
        # Check that intensity column is described
        self.assertIn("intensity", readme_content)
        self.assertIn("5 tab-separated columns", readme_content)

    def test_progress_callback(self):
        """Test that progress callback is called correctly."""
        # Create test session
        utterances = [("utt_001", "First"), ("utt_002", "Second"), ("utt_003", "Third")]
        self._create_test_session(
            self.session1_dir, "Test Speaker", "happy", utterances
        )

        # Track progress calls
        progress_calls = []

        def progress_callback(count):
            progress_calls.append(count)

        # Export with callback
        exporter = DatasetExporter(self.output_dir)
        exporter.export_sessions(
            [self.session1_dir], progress_callback=progress_callback
        )

        # Should be called for each utterance
        self.assertEqual(len(progress_calls), 3)
        self.assertEqual(progress_calls, [1, 2, 3])

    def test_empty_session_handling(self):
        """Test handling of session with no recordings."""
        # Create session with recordings directory but no actual recordings
        session_data = {
            "speaker": {"name": "Test", "emotion": "happy"},
            "audio_config": {"sample_rate": 48000, "bit_depth": 24},
        }

        with open(self.session1_dir / "session.json", "w") as f:
            json.dump(session_data, f)

        with open(self.session1_dir / "script.txt", "w") as f:
            f.write('(utt_001 "Test")')

        # Create utterance directory but no audio files
        utt_dir = self.session1_dir / "recordings" / "utt_001"
        utt_dir.mkdir(parents=True)

        # Export
        exporter = DatasetExporter(self.output_dir)
        output_paths, stats = exporter.export_sessions([self.session1_dir])

        # Should create dataset structure but no audio files
        self.assertEqual(len(output_paths), 1)
        # No utterances are processed because no take files exist
        self.assertEqual(stats["total_utterances"], 0)
        self.assertEqual(stats["missing_recordings"], 0)

        # Check empty index
        with open(output_paths[0] / "index.tsv", "r") as f:
            content = f.read()
        self.assertEqual(content, "")


if __name__ == "__main__":
    unittest.main()
