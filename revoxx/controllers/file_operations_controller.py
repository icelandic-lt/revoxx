"""File operations controller for managing recordings and file operations."""

from typing import Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
from tkinter import messagebox

from ..constants import MsgType

if TYPE_CHECKING:
    from ..app import Revoxx


class FileOperationsController:
    """Handles file operations for recordings and exports.

    This controller manages:
    - Recording deletion and restoration
    - File exports (session and individual)
    - Audio file operations
    - Recording status checks
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the file operations controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app

    def delete_current_recording(self) -> None:
        """Delete the current recording (move to trash)."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            return

        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take == 0:
            self.app.window.set_status("No recording to delete", MsgType.TEMPORARY)
            return

        # Ask for confirmation
        result = messagebox.askyesno(
            "Delete Recording",
            "Delete recording ?",
            parent=self.app.window.window,
        )

        if not result:
            return

        # Move file to trash
        success = self.app.file_manager.move_to_trash(current_label, current_take)

        if success:
            # Invalidate cache for this label
            if self.app.active_recordings:
                self.app.active_recordings.on_recording_deleted(
                    current_label, current_take
                )
                # Update takes from active recordings
                self.app.state.recording.takes = (
                    self.app.active_recordings.get_all_takes()
                )
                # Find next available take
                existing_takes = self.app.active_recordings.get_existing_takes(
                    current_label
                )
            else:
                existing_takes = []

            if existing_takes:
                # Set to the highest remaining take
                new_take = existing_takes[-1]
                self.app.state.recording.set_displayed_take(current_label, new_take)
            else:
                # No more takes, reset to 0
                self.app.state.recording.set_displayed_take(current_label, 0)

            # Clear mel spectrogram
            if hasattr(self.app.window, "mel_spectrogram"):
                self.app.window.mel_spectrogram.clear()

            # Show the current recording if one exists
            self.app.display_controller.show_saved_recording()

            # Update take status
            self.app.navigation_controller.update_take_status()

            # Update info panel if visible
            if self.app.window.info_panel_visible:
                self.app.display_controller.update_info_panel()

            self.app.window.set_status(
                f"Recording moved to trash: take_{current_take:03d}", MsgType.TEMPORARY
            )
        else:
            self.app.window.set_status("Failed to delete recording", MsgType.ERROR)

    def restore_deleted_recording(self) -> None:
        """Restore the last deleted recording from trash."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            return

        # Find deleted takes for this label
        deleted_takes = self.app.file_manager.get_deleted_takes(current_label)

        if not deleted_takes:
            self.app.window.set_status(
                "No deleted recordings to restore", MsgType.TEMPORARY
            )
            return

        # Restore the most recent deletion (highest take number)
        take_to_restore = max(deleted_takes)
        success = self.app.file_manager.restore_from_trash(
            current_label, take_to_restore
        )

        if success:
            # Invalidate cache for this label
            if self.app.active_recordings:
                self.app.active_recordings.on_recording_restored(current_label)
                # Update takes from active recordings
                self.app.state.recording.takes = (
                    self.app.active_recordings.get_all_takes()
                )

            # Set to the restored take
            self.app.state.recording.set_displayed_take(current_label, take_to_restore)

            # Show the restored recording
            self.app.display_controller.show_saved_recording()

            # Update take status
            self.app.navigation_controller.update_take_status()

            # Update info panel if visible
            if self.app.window.info_panel_visible:
                self.app.display_controller.update_info_panel()

            self.app.window.set_status(
                f"Recording restored: take_{take_to_restore:03d}", MsgType.TEMPORARY
            )
        else:
            self.app.window.set_status("Failed to restore recording", MsgType.ERROR)

    def export_session(self, export_path: Path) -> None:
        """Export the entire session to a directory.

        Args:
            export_path: Path to export directory
        """
        try:
            # Create export directory
            export_path.mkdir(parents=True, exist_ok=True)

            # Export script
            script_dest = export_path / "script.txt"
            with open(script_dest, "w", encoding="utf-8") as f:
                for utterance in self.app.state.recording.utterances:
                    f.write(f"{utterance.get('id', '')}\t{utterance.get('text', '')}\n")

            # Export recordings
            recordings_dir = export_path / "recordings"
            recordings_dir.mkdir(exist_ok=True)

            exported_count = 0
            for label, take_count in self.app.state.recording.takes.items():
                if take_count > 0:
                    # Get the highest take for export
                    if self.app.active_recordings:
                        highest_take = self.app.active_recordings.get_highest_take(
                            label
                        )
                    else:
                        highest_take = 0
                    if highest_take > 0:
                        source = self.app.file_manager.get_recording_path(
                            label, highest_take
                        )
                        if source.exists():
                            dest = recordings_dir / f"{label}.wav"
                            shutil.copy2(source, dest)
                            exported_count += 1

            self.app.window.set_status(
                f"Session exported: {exported_count} recordings", MsgType.TEMPORARY
            )

        except (OSError, IOError) as e:
            self.app.window.set_status(f"Export failed: {e}", MsgType.ERROR)

    def export_current_recording(self, export_path: Path) -> None:
        """Export the current recording to a file.

        Args:
            export_path: Path to export file
        """
        current_label = self.app.state.recording.current_label
        if not current_label:
            self.app.window.set_status("No recording to export", MsgType.TEMPORARY)
            return

        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take == 0:
            self.app.window.set_status("No recording to export", MsgType.TEMPORARY)
            return

        try:
            source = self.app.file_manager.get_recording_path(
                current_label, current_take
            )
            if source.exists():
                shutil.copy2(source, export_path)
                self.app.window.set_status(
                    f"Recording exported to {export_path.name}", MsgType.TEMPORARY
                )
            else:
                self.app.window.set_status("Recording file not found", MsgType.ERROR)
        except (OSError, IOError) as e:
            self.app.window.set_status(f"Export failed: {e}", MsgType.ERROR)

    def check_recording_exists(self, label: str, take: int) -> bool:
        """Check if a recording exists.

        Args:
            label: Recording label
            take: Take number

        Returns:
            True if recording exists
        """
        if take == 0:
            return False

        filepath = self.app.file_manager.get_recording_path(label, take)
        return filepath.exists()

    def get_recording_info(self, label: str, take: int) -> Optional[dict]:
        """Get information about a recording.

        Args:
            label: Recording label
            take: Take number

        Returns:
            Dictionary with recording info or None if not found
        """
        if not self.check_recording_exists(label, take):
            return None

        filepath = self.app.file_manager.get_recording_path(label, take)

        try:
            with sf.SoundFile(filepath) as f:
                return {
                    "duration": len(f) / f.samplerate,
                    "sample_rate": f.samplerate,
                    "channels": f.channels,
                    "file_size": filepath.stat().st_size,
                    "file_path": str(filepath),
                }
        except (OSError, ValueError):
            return None

    def load_audio_data(
        self, label: str, take: int
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio data for a recording.

        Args:
            label: Recording label
            take: Take number

        Returns:
            Tuple of (audio_data, sample_rate) or None if not found
        """
        if not self.check_recording_exists(label, take):
            return None

        filepath = self.app.file_manager.get_recording_path(label, take)

        try:
            return self.app.file_manager.load_audio(filepath)
        except (OSError, ValueError):
            return None

    def get_total_recording_count(self) -> int:
        """Get the total number of recordings in the session.

        Returns:
            Total recording count
        """
        count = 0
        for label, take_count in self.app.state.recording.takes.items():
            if take_count > 0:
                count += 1
        return count

    def get_session_size(self) -> int:
        """Get the total size of all recordings in bytes.

        Returns:
            Total size in bytes
        """
        total_size = 0
        for label, take_count in self.app.state.recording.takes.items():
            if take_count > 0:
                if self.app.active_recordings:
                    highest_take = self.app.active_recordings.get_highest_take(label)
                else:
                    highest_take = 0
                if highest_take > 0:
                    filepath = self.app.file_manager.get_recording_path(
                        label, highest_take
                    )
                    if filepath.exists():
                        total_size += filepath.stat().st_size
        return total_size

    def cleanup_empty_directories(self) -> None:
        """Clean up empty recording directories."""
        if self.app.current_session:
            self.app.file_manager.cleanup_empty_directories(
                self.app.current_session.recordings_dir
            )
