"""Dataset exporter for converting Revoxx sessions to Talrómur 3 format."""

import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import soundfile as sf

from ..session.inspector import SessionInspector
from ..session.script_parser import FestivalScriptParser


class DatasetExporter:
    """Export Revoxx sessions to Talrómur 3 dataset format.

    The Talrómur 3 format organizes recordings by emotion with an index.tsv file:
    - voice-name/emotion/voice-name_emotion_001.wav
    - voice-name/index.tsv with metadata

    Intensity Level Handling:
    - Text format "N: text" where N (1-5) is the intensity level
    - Neutral emotion always gets intensity 0 (regardless of text prefix)
    - Other emotions preserve the intensity from the text prefix
    - Missing intensity prefix defaults to 0
    """

    def __init__(
        self,
        output_dir: Path,
        audio_format: str = "flac",
        zero_intensity_emotions: List[str] = None,
        include_intensity: bool = True,
    ):
        """Initialize dataset exporter.

        Args:
            output_dir: Base output directory for datasets
            audio_format: Output audio format ('wav' or 'flac')
            zero_intensity_emotions: List of emotions to set intensity to 0
            include_intensity: Whether to include intensity column in index.tsv
        """
        self.output_dir = Path(output_dir)
        self.format = audio_format.lower()
        self.zero_intensity_emotions = zero_intensity_emotions or ["neutral"]
        self.include_intensity = include_intensity

    def _group_sessions_by_speaker(self, session_paths: List[Path]) -> Dict:
        """Group sessions by speaker name.

        Args:
            session_paths: List of session paths

        Returns:
            Dictionary mapping speaker names to session data
        """
        speaker_groups = {}
        for session_path in session_paths:
            session_data = self._load_session(session_path)
            if session_data:
                speaker_name = session_data.get("speaker", {}).get("name", "unknown")
                speaker_name_normalized = speaker_name.lower().replace(" ", "_")

                if speaker_name_normalized not in speaker_groups:
                    speaker_groups[speaker_name_normalized] = []
                speaker_groups[speaker_name_normalized].append(
                    (session_path, session_data)
                )
        return speaker_groups

    def export_sessions(
        self,
        session_paths: List[Path],
        dataset_name: str = None,
        progress_callback=None,
    ) -> Tuple[List[Path], Dict]:
        """Export Revoxx sessions grouped by speaker name.

        Sessions with the same speaker name will be grouped into one dataset.

        Args:
            session_paths: List of paths to .revoxx session directories
            dataset_name: Optional override for dataset name (if None, uses speaker names)
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (list of output_paths, statistics_dict)
        """
        if not session_paths:
            raise ValueError("No sessions provided")

        # Load session metadata and group by speaker name
        speaker_groups = self._group_sessions_by_speaker(session_paths)

        if not speaker_groups:
            raise ValueError("No valid sessions found")

        # Process each speaker group as a separate dataset
        all_datasets = []
        total_statistics = {
            "total_utterances": 0,
            "missing_recordings": 0,
            "datasets_created": 0,
            "sessions_processed": 0,
            "speakers": [],
        }

        for speaker_name_normalized, sessions in speaker_groups.items():
            # Use override name if provided and only one speaker group
            if dataset_name and len(speaker_groups) == 1:
                current_dataset_name = dataset_name
            else:
                current_dataset_name = speaker_name_normalized

            # Create output directory for this speaker
            dataset_dir = self.output_dir / current_dataset_name
            if dataset_dir.exists():
                # For now, we'll overwrite. In UI, we'll ask for confirmation
                shutil.rmtree(dataset_dir)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Process sessions for this speaker
            index_data = []
            file_counts = Counter()
            speaker_statistics = {
                "speaker": speaker_name_normalized,
                "emotions": set(),
                "sessions": len(sessions),
            }

            # Group sessions by emotion
            emotion_sessions = {}
            for session_path, session_data in sessions:
                emotion = session_data.get("speaker", {}).get("emotion", "unknown")
                if emotion not in emotion_sessions:
                    emotion_sessions[emotion] = []
                emotion_sessions[emotion].append((session_path, session_data))
                speaker_statistics["emotions"].add(emotion)

            # Process each emotion group
            for emotion, emotion_session_list in emotion_sessions.items():
                self._process_emotion_group(
                    emotion,
                    emotion_session_list,
                    dataset_dir,
                    current_dataset_name,
                    index_data,
                    file_counts,
                    total_statistics,
                    progress_callback,
                )

            # Write index file for this speaker
            index_path = dataset_dir / "index.tsv"
            with open(index_path, "w", encoding="utf-8") as f:
                f.writelines(index_data)

            # Get audio properties from first session
            audio_properties = self._get_audio_properties(sessions[0][1])

            # Write README file with format documentation
            self._write_readme(dataset_dir, audio_properties)

            all_datasets.append(dataset_dir)
            total_statistics["datasets_created"] += 1
            total_statistics["sessions_processed"] += len(sessions)
            total_statistics["speakers"].append(
                {
                    "name": speaker_name_normalized,
                    "emotions": list(speaker_statistics["emotions"]),
                    "file_counts": dict(file_counts),
                    "output_path": str(dataset_dir),
                }
            )

        return all_datasets, total_statistics

    def _process_emotion_group(
        self,
        emotion: str,
        emotion_session_list: List,
        dataset_dir: Path,
        dataset_name: str,
        index_data: List,
        file_counts: Counter,
        total_statistics: Dict,
        progress_callback=None,
    ) -> None:
        """Process all sessions for a specific emotion.

        Args:
            emotion: Emotion name
            emotion_session_list: List of sessions for this emotion
            dataset_dir: Output directory for dataset
            dataset_name: Name of the dataset
            index_data: List to append index entries to
            file_counts: Counter for file numbering
            total_statistics: Statistics dictionary to update
            progress_callback: Optional progress callback
        """
        emotion_dir = dataset_dir / emotion
        emotion_dir.mkdir(exist_ok=True)

        # Process all utterances for this emotion
        utterance_map = self._collect_utterances(emotion_session_list)

        for utterance_id, (session_path, take_num, text) in utterance_map.items():
            total_statistics["total_utterances"] += 1

            # Extract intensity and clean text
            intensity, clean_text = self._extract_intensity_and_text(text)
            if emotion in self.zero_intensity_emotions:
                intensity = "0"

            # Process audio file
            source_file = (
                session_path / "recordings" / utterance_id / f"take_{take_num:03d}.flac"
            )

            if source_file.exists():
                file_counter = file_counts[emotion] + 1
                output_filename = (
                    f"{dataset_name}_{emotion}_{file_counter:03d}.{self.format}"
                )
                output_path = emotion_dir / output_filename

                # Copy or convert audio
                if self.format == "flac" and source_file.suffix == ".flac":
                    shutil.copy2(source_file, output_path)
                else:
                    self._convert_audio(source_file, output_path)

                file_counts[emotion] += 1

                # Add to index
                if self.include_intensity:
                    index_data.append(
                        f"{output_filename}\t{dataset_name}\t{emotion}\t{intensity}\t{clean_text}\n"
                    )
                else:
                    index_data.append(
                        f"{output_filename}\t{dataset_name}\t{emotion}\t{clean_text}\n"
                    )
            else:
                total_statistics["missing_recordings"] += 1

            if progress_callback:
                progress_callback(total_statistics["total_utterances"])

    @staticmethod
    def _load_session(session_path: Path) -> Optional[Dict]:
        """Load session metadata from session.json."""
        return SessionInspector.load_metadata(session_path)

    def _collect_utterances(self, emotion_sessions: List[Tuple[Path, Dict]]) -> Dict:
        """Collect all utterances from sessions, choosing best take for each.

        Returns:
            Dict mapping utterance_id to (session_path, take_number, text)
        """
        utterances = {}

        for session_path, session_data in emotion_sessions:
            recordings_dir = session_path / "recordings"
            script_file = session_path / "script.txt"
            script_data = self._load_script(script_file)

            # Find all recordings
            if recordings_dir.exists():
                for utterance_dir in recordings_dir.iterdir():
                    if utterance_dir.is_dir():
                        utterance_id = utterance_dir.name

                        # Find highest take number
                        takes = list(utterance_dir.glob("take_*.flac"))
                        takes.extend(list(utterance_dir.glob("take_*.wav")))

                        if takes:
                            # Extract take numbers and find highest
                            take_numbers = []
                            for take_file in takes:
                                try:
                                    take_num = int(take_file.stem.split("_")[1])
                                    take_numbers.append(take_num)
                                except (IndexError, ValueError):
                                    continue

                            if take_numbers:
                                highest_take = max(take_numbers)
                                text = script_data.get(utterance_id, "")
                                utterances[utterance_id] = (
                                    session_path,
                                    highest_take,
                                    text,
                                )

        return utterances

    @staticmethod
    def _load_script(script_file: Path) -> Dict[str, str]:
        """Load script file and return mapping of utterance_id to text."""
        return FestivalScriptParser.parse_script(script_file)

    @staticmethod
    def _get_audio_properties(session_data: Dict) -> Dict[str, Any]:
        """Extract audio properties from session data.

        Returns:
            Dict with sample_rate and bit_depth
        """
        audio_config = session_data.get("audio_config", {})
        return {
            "sample_rate": audio_config.get("sample_rate"),
            "bit_depth": audio_config.get("bit_depth"),
        }

    @staticmethod
    def _extract_intensity_and_text(text: str) -> Tuple[str, str]:
        """Extract intensity level and clean text from utterance.

        The text format can be:
        - "N: actual text" where N is intensity level 1-5
        - Just "actual text" for no intensity (defaults to "0")

        Note: The calling code may override the intensity to "0" for
        specific emotions like "neutral" regardless of the text prefix.

        Returns:
            Tuple of (intensity, clean_text)
        """
        return FestivalScriptParser.extract_intensity_and_text(text)

    @staticmethod
    def _convert_audio(source_path: Path, dest_path: Path):
        """Convert audio file to target format."""
        try:
            data, samplerate = sf.read(str(source_path))
            sf.write(str(dest_path), data, samplerate)
        except Exception:
            # If conversion fails, try to copy as-is
            shutil.copy2(source_path, dest_path)

    def _write_readme(self, dataset_dir: Path, audio_properties: Dict[str, Any]):
        """Write README.txt with TSV format documentation from template.

        Args:
            dataset_dir: Directory to write README to
            audio_properties: Dict with sample_rate and bit_depth
        """
        # Load main template
        template_dir = Path(__file__).parent.parent / "resources" / "templates"
        main_template_path = template_dir / "dataset_readme.txt"

        with open(main_template_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Load appropriate index format template
        if self.include_intensity:
            index_template_path = template_dir / "index_format_with_intensity.txt"
        else:
            index_template_path = template_dir / "index_format_without_intensity.txt"

        with open(index_template_path, "r", encoding="utf-8") as f:
            index_format = f.read()

        # Prepare audio properties strings
        sample_rate_str = (
            str(audio_properties["sample_rate"])
            if audio_properties["sample_rate"]
            else "Not specified"
        )
        bit_depth_str = (
            str(audio_properties["bit_depth"])
            if audio_properties["bit_depth"]
            else "Not specified"
        )

        # Fill in template variables
        readme_content = readme_content.format(
            index_format=index_format,
            audio_format=self.format.upper(),
            file_extension=self.format,
            sample_rate=sample_rate_str,
            bit_depth=bit_depth_str,
        )

        # Write README file
        readme_path = dataset_dir / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
