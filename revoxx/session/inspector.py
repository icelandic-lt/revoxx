"""Session inspection utilities for analyzing Revoxx sessions."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class SessionInfo:
    """Information about a single session."""

    path: Path
    name: str
    speaker: str
    emotion: str
    created_at: str
    total_utterances: int
    recorded_utterances: int
    recording_files: int

    @property
    def is_complete(self) -> bool:
        """Check if all utterances have been recorded."""
        return self.recorded_utterances >= self.total_utterances

    @property
    def is_empty(self) -> bool:
        """Check if session has no recordings."""
        return self.recorded_utterances == 0

    @property
    def missing_utterances(self) -> int:
        """Get number of missing utterances."""
        return max(0, self.total_utterances - self.recorded_utterances)


class ValidationResult(NamedTuple):
    """Result of session validation."""

    valid_sessions: List[Path]
    incomplete_sessions: List[Dict]
    empty_sessions: List[Dict]


class SessionInspector:
    """Utilities for inspecting and validating Revoxx sessions."""

    @staticmethod
    def load_metadata(session_path: Path) -> Optional[Dict]:
        """Load session metadata from session.json.

        Args:
            session_path: Path to .revoxx session directory

        Returns:
            Session metadata dict or None if not found/invalid
        """
        session_file = session_path / "session.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    @staticmethod
    def count_utterances_from_script(script_path: Path) -> int:
        """Count total utterances in a script file.

        Args:
            script_path: Path to script.txt file

        Returns:
            Number of utterances in the script
        """
        if not script_path.exists():
            return 0

        count = 0
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Festival format: (utterance_id "text")
                    if line.startswith("(") and line.endswith(")"):
                        count += 1
        except IOError:
            return 0

        return count

    @staticmethod
    def count_recorded_utterances(recordings_dir: Path) -> Tuple[int, int]:
        """Count recorded utterances and total recording files.

        Args:
            recordings_dir: Path to recordings directory

        Returns:
            Tuple of (utterances_with_recordings, total_recording_files)
        """
        if not recordings_dir.exists():
            return 0, 0

        utterances_with_recordings = 0
        total_files = 0

        for utterance_dir in recordings_dir.iterdir():
            if not utterance_dir.is_dir():
                continue

            # Check for take files
            takes = list(utterance_dir.glob("take_*.flac"))
            takes.extend(list(utterance_dir.glob("take_*.wav")))

            if takes:
                utterances_with_recordings += 1
                total_files += len(takes)

        return utterances_with_recordings, total_files

    @classmethod
    def inspect_session(cls, session_path: Path) -> Optional[SessionInfo]:
        """Inspect a single session and gather comprehensive information.

        Args:
            session_path: Path to .revoxx session directory

        Returns:
            SessionInfo object or None if session is invalid
        """
        if not session_path.is_dir() or session_path.suffix != ".revoxx":
            return None

        # Load metadata
        metadata = cls.load_metadata(session_path)
        if not metadata:
            return None

        # Extract basic info
        speaker_info = metadata.get("speaker", {})
        speaker_name = speaker_info.get("name", "Unknown")
        emotion = speaker_info.get("emotion", "unknown")
        created_at = metadata.get("created_at", "")

        # Count utterances
        script_path = session_path / "script.txt"
        total_utterances = cls.count_utterances_from_script(script_path)

        # Count recordings
        recordings_dir = session_path / "recordings"
        recorded_utterances, recording_files = cls.count_recorded_utterances(
            recordings_dir
        )

        return SessionInfo(
            path=session_path,
            name=session_path.name,
            speaker=speaker_name,
            emotion=emotion,
            created_at=created_at,
            total_utterances=total_utterances,
            recorded_utterances=recorded_utterances,
            recording_files=recording_files,
        )

    @classmethod
    def validate_sessions(cls, session_paths: List[Path]) -> ValidationResult:
        """Validate multiple sessions for completeness.

        Args:
            session_paths: List of paths to .revoxx session directories

        Returns:
            ValidationResult with categorized sessions
        """
        valid_sessions = []
        incomplete_sessions = []
        empty_sessions = []

        for session_path in session_paths:
            info = cls.inspect_session(session_path)
            if not info:
                continue

            if info.is_empty:
                empty_sessions.append(
                    {
                        "path": session_path,
                        "name": info.name,
                        "total": info.total_utterances,
                        "recorded": 0,
                    }
                )
            elif not info.is_complete:
                incomplete_sessions.append(
                    {
                        "path": session_path,
                        "name": info.name,
                        "total": info.total_utterances,
                        "recorded": info.recorded_utterances,
                        "missing": info.missing_utterances,
                    }
                )
                valid_sessions.append(session_path)
            else:
                valid_sessions.append(session_path)

        return ValidationResult(
            valid_sessions=valid_sessions,
            incomplete_sessions=incomplete_sessions,
            empty_sessions=empty_sessions,
        )

    @classmethod
    def find_sessions(cls, base_dir: Path) -> List[SessionInfo]:
        """Find all valid sessions in a directory.

        Args:
            base_dir: Directory to search for .revoxx sessions

        Returns:
            List of SessionInfo objects for valid sessions
        """
        if not base_dir.exists():
            return []

        sessions = []
        for item in sorted(base_dir.iterdir()):
            if item.is_dir() and item.suffix == ".revoxx":
                info = cls.inspect_session(item)
                if info:
                    sessions.append(info)

        return sessions
