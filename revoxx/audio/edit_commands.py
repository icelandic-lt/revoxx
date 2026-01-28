"""Edit commands for undo/redo support.

This module provides command classes that encapsulate audio editing operations.
Each command stores the data needed to undo the operation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..files.file_manager import FileManager


class EditCommand(ABC):
    """Abstract base class for edit commands.

    Each command represents an audio editing operation that can be
    undone and redone. Commands store the data needed to reverse
    the operation.
    """

    def __init__(self, filepath: Path, sample_rate: int, subtype: Optional[str] = None):
        """Initialize the command.

        Args:
            filepath: Path to the audio file being edited
            sample_rate: Sample rate of the audio
            subtype: Audio subtype for saving (e.g., "PCM_16")
        """
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.subtype = subtype

    @abstractmethod
    def execute(self, file_manager: "FileManager") -> bool:
        """Execute the command.

        Args:
            file_manager: FileManager instance for audio I/O

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def inverse(self) -> "EditCommand":
        """Create the inverse command for undo.

        Returns:
            A command that reverses this operation
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """Get a human-readable description of this command.

        Returns:
            Description string (e.g., "Delete Range")
        """
        pass


class AudioSnapshotCommand(EditCommand):
    """Command that stores complete audio snapshots for undo/redo.

    This approach ensures exact restoration without re-applying cross-fade.
    Stores both the audio before and after the operation.
    """

    def __init__(
        self,
        filepath: Path,
        sample_rate: int,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        subtype: Optional[str] = None,
        selection_start_time: Optional[float] = None,
        selection_end_time: Optional[float] = None,
        marker_after_edit: Optional[float] = None,
        operation_description: str = "Edit",
    ):
        """Initialize audio snapshot command.

        Args:
            filepath: Path to the audio file
            sample_rate: Sample rate of the audio
            audio_before: Complete audio data before the operation
            audio_after: Complete audio data after the operation
            subtype: Audio subtype for saving
            selection_start_time: Selection to restore on undo
            selection_end_time: Selection to restore on undo
            marker_after_edit: Marker to set after redo
            operation_description: Description of the operation
        """
        super().__init__(filepath, sample_rate, subtype)
        self.audio_before = audio_before
        self.audio_after = audio_after
        self.selection_start_time = selection_start_time
        self.selection_end_time = selection_end_time
        self.marker_after_edit = marker_after_edit
        self.operation_description = operation_description

    def execute(self, file_manager: "FileManager") -> bool:
        """Write audio_after to file (for redo)."""
        try:
            file_manager.save_audio(
                self.filepath, self.audio_after, self.sample_rate, self.subtype
            )
            return True
        except (OSError, ValueError):
            return False

    def inverse(self) -> "AudioSnapshotCommand":
        """Create inverse command that restores audio_before."""
        return AudioSnapshotCommand(
            filepath=self.filepath,
            sample_rate=self.sample_rate,
            audio_before=self.audio_after,
            audio_after=self.audio_before,
            subtype=self.subtype,
            selection_start_time=self.selection_start_time,
            selection_end_time=self.selection_end_time,
            marker_after_edit=None,
            operation_description=self.operation_description,
        )

    def description(self) -> str:
        """Get description of this command."""
        return self.operation_description


# Convenience aliases for specific operations (all use AudioSnapshotCommand internally)
DeleteRangeCommand = AudioSnapshotCommand
InsertCommand = AudioSnapshotCommand
ReplaceRangeCommand = AudioSnapshotCommand


class TrashClipCommand(EditCommand):
    """Command for moving a clip to trash.

    Uses the existing trash system (move_to_trash/restore_from_trash).
    """

    def __init__(
        self,
        filepath: Path,
        sample_rate: int,
        label: str,
        take: int,
    ):
        """Initialize trash clip command.

        Args:
            filepath: Path to the audio file
            sample_rate: Sample rate of the audio
            label: Script label/ID for the utterance
            take: Take number
        """
        super().__init__(filepath, sample_rate, None)
        self.label = label
        self.take = take

    def execute(self, file_manager: "FileManager") -> bool:
        """Move the clip to trash."""
        return file_manager.move_to_trash(self.label, self.take)

    def inverse(self) -> "RestoreFromTrashCommand":
        """Create a restore command to undo this deletion."""
        return RestoreFromTrashCommand(
            filepath=self.filepath,
            sample_rate=self.sample_rate,
            label=self.label,
            take=self.take,
        )

    def description(self) -> str:
        """Get description of this command."""
        return f"Delete Clip (take {self.take})"


class RestoreFromTrashCommand(EditCommand):
    """Command for restoring a clip from trash."""

    def __init__(
        self,
        filepath: Path,
        sample_rate: int,
        label: str,
        take: int,
    ):
        """Initialize restore from trash command."""
        super().__init__(filepath, sample_rate, None)
        self.label = label
        self.take = take

    def execute(self, file_manager: "FileManager") -> bool:
        """Restore the clip from trash."""
        return file_manager.restore_from_trash(self.label, self.take)

    def inverse(self) -> "TrashClipCommand":
        """Create a trash command to undo this restoration."""
        return TrashClipCommand(
            filepath=self.filepath,
            sample_rate=self.sample_rate,
            label=self.label,
            take=self.take,
        )

    def description(self) -> str:
        """Get description of this command."""
        return f"Restore Clip (take {self.take})"


class DeleteClipCommand(EditCommand):
    """Command for deleting an entire audio clip.

    Stores the complete audio data so it can be restored on undo.
    Note: For trash-based deletion, use TrashClipCommand instead.
    """

    def __init__(
        self,
        filepath: Path,
        sample_rate: int,
        audio_data: np.ndarray,
        subtype: Optional[str] = None,
    ):
        """Initialize delete clip command.

        Args:
            filepath: Path to the audio file
            sample_rate: Sample rate of the audio
            audio_data: The complete audio data of the deleted clip
            subtype: Audio subtype for saving (e.g., "PCM_16")
        """
        super().__init__(filepath, sample_rate)
        self.audio_data = audio_data
        self.subtype = subtype

    def execute(self, file_manager: "FileManager") -> bool:
        """Execute the delete clip operation (delete the file).

        Args:
            file_manager: FileManager instance for audio I/O

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.filepath.exists():
                self.filepath.unlink()
            return True
        except OSError:
            return False

    def inverse(self) -> "RestoreClipCommand":
        """Create a restore command to undo this deletion.

        Returns:
            RestoreClipCommand that recreates the deleted file
        """
        return RestoreClipCommand(
            filepath=self.filepath,
            sample_rate=self.sample_rate,
            audio_data=self.audio_data,
            subtype=self.subtype,
        )

    def description(self) -> str:
        """Get description of this command."""
        duration = len(self.audio_data) / self.sample_rate
        return f"Delete Clip ({duration:.2f}s)"


class RestoreClipCommand(EditCommand):
    """Command for restoring a deleted audio clip.

    Used as the inverse of DeleteClipCommand.
    """

    def __init__(
        self,
        filepath: Path,
        sample_rate: int,
        audio_data: np.ndarray,
        subtype: Optional[str] = None,
    ):
        """Initialize restore clip command.

        Args:
            filepath: Path to the audio file
            sample_rate: Sample rate of the audio
            audio_data: The audio data to restore
            subtype: Audio subtype for saving (e.g., "PCM_16")
        """
        super().__init__(filepath, sample_rate)
        self.audio_data = audio_data
        self.subtype = subtype

    def execute(self, file_manager: "FileManager") -> bool:
        """Execute the restore operation (recreate the file).

        Args:
            file_manager: FileManager instance for audio I/O

        Returns:
            True if successful, False otherwise
        """
        try:
            file_manager.save_audio(
                self.filepath, self.audio_data, self.sample_rate, self.subtype
            )
            return True
        except (OSError, ValueError):
            return False

    def inverse(self) -> "DeleteClipCommand":
        """Create a delete command to undo this restoration.

        Returns:
            DeleteClipCommand that deletes the file again
        """
        return DeleteClipCommand(
            filepath=self.filepath,
            sample_rate=self.sample_rate,
            audio_data=self.audio_data,
            subtype=self.subtype,
        )

    def description(self) -> str:
        """Get description of this command."""
        duration = len(self.audio_data) / self.sample_rate
        return f"Restore Clip ({duration:.2f}s)"
