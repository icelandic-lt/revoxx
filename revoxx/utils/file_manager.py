"""File management utilities for the recorder."""

from pathlib import Path
from typing import Optional, Tuple, List, Set
import soundfile as sf
import numpy as np

from ..constants import FileConstants


class RecordingFileManager:
    """Manages recording files and directory structure.

    This class handles all file operations related to audio recordings,
    including file naming, path generation, existence checking, and
    scanning for existing recordings. It supports robust navigation
    through recordings even with gaps in take numbers.

    Attributes:
        recording_dir: Base directory for all recordings

    File Naming Convention:
        Session structure: recordings/<utterance-id>/take_XXX.wav
        Example: recordings/utt_001/take_001.wav, recordings/utt_001/take_002.wav
    """

    def __init__(self, recording_dir: Path):
        """Initialize the file manager.

        Args:
            recording_dir: Directory for storing recordings (created if not exists)
        """
        self.recording_dir = Path(recording_dir)
        self.recording_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _extract_take_number(file_path: Path) -> Optional[int]:
        """Extract take number from a file path.

        Args:
            file_path: Path to a take file

        Returns:
            Take number if valid, None otherwise
        """
        try:
            # Filename format: take_XXX.ext
            take_str = file_path.stem.split("_")[1]
            return int(take_str)
        except (ValueError, IndexError):
            return None

    def _get_take_files(self, label: str, include_trash: bool = False) -> List[Path]:
        """Get all take files for a label.

        Args:
            label: Script label/ID for the utterance
            include_trash: Whether to include files in trash directory

        Returns:
            List of Path objects for all take files
        """
        files = []

        # Check main directory
        utterance_dir = self.recording_dir / label
        if utterance_dir.exists():
            # Check for both FLAC and WAV files
            flac_pattern = f"take_*{FileConstants.AUDIO_FILE_EXTENSION}"
            wav_pattern = f"take_*{FileConstants.LEGACY_AUDIO_FILE_EXTENSION}"
            files.extend(utterance_dir.glob(flac_pattern))
            files.extend(utterance_dir.glob(wav_pattern))

        # Check trash directory if requested
        if include_trash:
            trash_dir = self.recording_dir.parent / "trash" / label
            if trash_dir.exists():
                files.extend(trash_dir.glob("take_*.*"))

        return files

    def _get_take_numbers(self, label: str, include_trash: bool = False) -> Set[int]:
        """Get all take numbers for a label.

        Args:
            label: Script label/ID for the utterance
            include_trash: Whether to include files in trash directory

        Returns:
            Set of take numbers
        """
        take_numbers = set()
        files = self._get_take_files(label, include_trash)

        for file in files:
            take_num = self._extract_take_number(file)
            if take_num is not None:
                take_numbers.add(take_num)

        return take_numbers

    def get_recording_path(self, label: str, take: int) -> Path:
        """Get the path for a recording file.

        Args:
            label: Script label/ID for the utterance
            take: Take number (1-based)

        Returns:
            Path: Full path to the recording file
        """
        # Session structure: recordings/<utterance-id>/take_XXX.wav
        utterance_dir = self.recording_dir / label
        utterance_dir.mkdir(exist_ok=True, parents=True)

        # Format take number with leading zeros
        take_str = f"{take:03d}"

        # Check for existing files (FLAC or WAV)
        wav_filename = f"take_{take_str}{FileConstants.LEGACY_AUDIO_FILE_EXTENSION}"
        wav_path = utterance_dir / wav_filename
        if wav_path.exists():
            return wav_path

        flac_filename = f"take_{take_str}{FileConstants.AUDIO_FILE_EXTENSION}"
        return utterance_dir / flac_filename

    def recording_exists(self, label: str, take: int) -> bool:
        """Check if a recording exists.

        Args:
            label: Script label/ID for the utterance
            take: Take number to check

        Returns:
            bool: True if the recording file exists
        """
        utterance_dir = self.recording_dir / label
        if not utterance_dir.exists():
            return False

        take_str = f"{take:03d}"
        flac_filename = f"take_{take_str}{FileConstants.AUDIO_FILE_EXTENSION}"
        wav_filename = f"take_{take_str}{FileConstants.LEGACY_AUDIO_FILE_EXTENSION}"

        return (utterance_dir / flac_filename).exists() or (
            utterance_dir / wav_filename
        ).exists()

    def get_highest_take(self, label: str) -> int:
        """Find the highest take number including trash (for avoiding conflicts).

        Args:
            label: Script label/ID for the utterance

        Returns:
            int: Highest take number found (0 if no recordings)
        """
        take_numbers = self._get_take_numbers(label, include_trash=True)
        return max(take_numbers, default=0)

    def scan_all_take_files(self, labels: List[str]) -> dict[str, List[str]]:
        """Scan for all existing take filenames for given labels.

        Efficiently scans the recording directory to find all take
        filenames for each label in the provided list.
        Note: This excludes files in the trash directory.

        Args:
            labels: List of script labels to scan

        Returns:
            dict: Mapping of label to list of filenames (excluding trash)
        """
        takes = {}
        for label in labels:
            files = self._get_take_files(label, include_trash=False)
            # Extract just the filenames, not full paths
            filenames = [
                f.name for f in files if self._extract_take_number(f) is not None
            ]
            takes[label] = sorted(filenames)
        return takes

    @staticmethod
    def get_file_info(file_path: Path) -> Optional[Tuple[int, int, str, int, float]]:
        """Get audio file information.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (sample_rate, bit_depth, format, channels, duration) or None
        """
        try:
            info = sf.info(str(file_path))

            # Determine bit depth from subtype
            bit_depth = 16  # default
            if "PCM_24" in info.subtype or "FLAC" in info.subtype:
                bit_depth = 24
            elif "PCM_16" in info.subtype:
                bit_depth = 16

            # Format
            format_name = "FLAC" if info.format == "FLAC" else "WAV"

            return (
                info.samplerate,
                bit_depth,
                format_name,
                info.channels,
                info.duration,
            )
        except Exception as e:
            print(f"Error reading file info: {e}")
            return None

    @staticmethod
    def load_audio(filepath: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return data with sample rate.

        Loads audio data using soundfile, automatically converting
        stereo to mono if necessary.

        Args:
            filepath: Path to the audio file

        Returns:
            Tuple[np.ndarray, int]: Audio data (normalized -1 to 1) and sample rate

        Raises:
            FileNotFoundError: If the audio file doesn't exist

        Note:
            Audio data is returned normalized between -1 and 1,
            regardless of the original bit depth.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        data, sample_rate = sf.read(str(filepath))

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        return data, sample_rate

    @staticmethod
    def save_audio(
        filepath: Path, data: np.ndarray, sample_rate: int, subtype: str
    ) -> None:
        """Save audio data to file.

        Args:
            filepath: Output file path
            data: Audio data array
            sample_rate: Sample rate in Hz
            subtype: Audio subtype (e.g., 'PCM_16', 'PCM_24', or None for FLAC)
        """
        if subtype:
            sf.write(str(filepath), data, sample_rate, subtype=subtype)
        else:
            # For FLAC, let soundfile determine format from extension
            sf.write(str(filepath), data, sample_rate)

    def move_to_trash(self, label: str, take: int) -> bool:
        """Move a recording to the trash directory.

        Uses the session-level trash directory structure:
        session_dir/trash/<label>/take_XXX.ext

        Args:
            label: Script label/ID for the utterance
            take: Take number to move

        Returns:
            bool: True if successful, False otherwise
        """
        source_path = self.get_recording_path(label, take)
        if not source_path.exists():
            return False

        # Use session-level trash directory
        # recordings/../trash/<label>/
        trash_dir = self.recording_dir.parent / "trash" / label
        trash_dir.mkdir(exist_ok=True, parents=True)

        # Keep original filename when moving to trash
        dest_path = trash_dir / source_path.name

        try:
            source_path.rename(dest_path)
            return True
        except Exception:
            return False

    def get_next_take_number(self, label: str) -> int:
        """Get the next available take number, considering both active and trash files.

        This ensures no conflicts when creating new recordings.

        Args:
            label: Script label/ID for the utterance

        Returns:
            int: Next available take number (1-based)
        """
        # Get highest from both active and trash
        highest = self.get_highest_take(label)  # This already includes trash
        return highest + 1


class ScriptFileManager:
    """Manages script files in Festival data format.

    This class handles loading and saving script files that contain
    utterances to be recorded. Scripts use the Festival data format:
    (label "utterance text")
    """

    @staticmethod
    def load_script(filepath: Path) -> Tuple[List[str], List[str]]:
        """Load and parse script file in Festival data format.

        Parses files in the format:
        (label1 "utterance text 1")
        (label2 "utterance text 2")

        Args:
            filepath: Path to the script file

        Returns:
            Tuple[List[str], List[str]]: Lists of labels and utterances

        Raises:
            FileNotFoundError: If script file doesn't exist
            ValueError: If file format is invalid
        """
        if not filepath:
            raise FileNotFoundError("No script file specified")

        if not filepath.exists():
            raise FileNotFoundError(f"Script file not found: {filepath}")

        labels = []
        utterances = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse Festival format: (label "text")
                if not line.startswith("(") or not line.endswith(")"):
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue

                # Remove parentheses
                content = line[1:-1].strip()

                # Find the label (first word)
                parts = content.split(None, 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue

                label = parts[0]
                text_part = parts[1].strip()

                # Extract text from quotes
                if text_part.startswith('"') and text_part.endswith('"'):
                    text = text_part[1:-1]
                else:
                    print(f"Warning: Text not in quotes at line {line_num}: {line}")
                    text = text_part

                labels.append(label)
                utterances.append(text)

        if not labels:
            raise ValueError(f"No valid utterances found in {filepath}")

        return labels, utterances

    @staticmethod
    def save_script(filepath: Path, labels: List[str], utterances: List[str]) -> None:
        """Save script in Festival data format.

        Args:
            filepath: Output file path
            labels: List of utterance labels
            utterances: List of utterance texts

        Raises:
            ValueError: If labels and utterances have different lengths
        """
        if len(labels) != len(utterances):
            raise ValueError("Labels and utterances must have the same length")

        with open(filepath, "w", encoding="utf-8") as f:
            for label, text in zip(labels, utterances):
                f.write(f'({label} "{text}")\n')

    @staticmethod
    def validate_script(filepath: Path) -> Tuple[bool, List[str]]:
        """Validate a script file without fully loading it.

        Args:
            filepath: Path to script file

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []

        if not filepath.exists():
            return False, ["Script file does not exist"]

        try:
            labels, utterances = ScriptFileManager.load_script(filepath)
            if not labels:
                errors.append("Script contains no valid utterances")
            return len(errors) == 0, errors
        except Exception as e:
            return False, [str(e)]
