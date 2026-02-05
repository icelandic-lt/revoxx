"""Edit controller for audio editing operations.

This controller handles audio editing operations such as deleting,
inserting, and replacing audio segments within recordings.
"""

from typing import TYPE_CHECKING, Optional

from ..constants import MsgType, FileConstants
from ..audio.editor import AudioEditor
from ..audio.undo_stack import UndoStack
from ..audio.edit_commands import (
    AudioSnapshotCommand,
    TrashClipCommand,
    RestoreFromTrashCommand,
)

if TYPE_CHECKING:
    import numpy as np
    from ..app import Revoxx


class EditController:
    """Handles audio editing operations.

    This controller provides:
    - Delete selection: Remove a selected range from the recording
    - Insert at position: Insert a new recording at a marker position
    - Replace selection: Replace a selected range with a new recording
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the edit controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self._pending_insert_position: Optional[float] = None
        self._pending_replace_range: Optional[tuple] = None
        self.undo_stack = UndoStack(max_size=10)

    def _get_selection_state(self):
        """Get selection state from spectrogram widget.

        Returns:
            SelectionState or None if not available
        """
        if self.app.window and self.app.window.mel_spectrogram:
            return self.app.window.mel_spectrogram.selection_state
        return None

    def _get_current_recording_info(self):
        """Get current recording label, take, and file path.

        Returns:
            Tuple of (label, take, filepath) or (None, None, None) if not available
        """
        current_label = self.app.state.recording.current_label
        if not current_label:
            return None, None, None

        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take == 0:
            return None, None, None

        filepath = self.app.file_manager.get_recording_path(current_label, current_take)
        if not filepath.exists():
            return None, None, None

        return current_label, current_take, filepath

    def delete_selection(self) -> bool:
        """Delete the currently selected range from the recording.

        Returns:
            True if deletion was successful, False otherwise
        """
        selection_state = self._get_selection_state()
        if not selection_state or not selection_state.has_selection:
            self.app.display_controller.set_status(
                "No selection to delete", MsgType.TEMPORARY
            )
            return False

        label, take, filepath = self._get_current_recording_info()
        if not filepath:
            self.app.display_controller.set_status(
                "No recording to edit", MsgType.TEMPORARY
            )
            return False

        try:
            # Load audio (this is audio_before)
            audio_before, sr = self.app.file_manager.load_audio(filepath)

            # Get selection range in samples
            sample_range = selection_state.get_selection_samples(sr)
            if not sample_range:
                return False

            start_sample, end_sample = sample_range

            # Save selection info before clearing
            sel_start = selection_state.selection_start
            sel_end = selection_state.selection_end
            deletion_position = sel_start

            # Delete the range (this creates audio_after with cross-fade)
            audio_after = AudioEditor.delete_range(
                audio_before, start_sample, end_sample, sr
            )

            # Check if deletion removed all audio
            if len(audio_after) == 0:
                # Move the now-empty recording to trash instead of saving a 0-byte file
                self.app.file_manager.move_to_trash(label, take)

                # Update active recordings cache and take counter
                if self.app.active_recordings:
                    self.app.active_recordings.on_recording_deleted(label, take)
                    self.app.state.recording.takes = (
                        self.app.active_recordings.get_all_takes()
                    )

                self.app.state.recording.set_displayed_take(label, 0)

                # Clear selection and display
                selection_state.clear_all()
                if self.app.window and self.app.window.mel_spectrogram:
                    self.app.window.mel_spectrogram.selection_visualizer.clear()

                self.app.display_controller.clear_spectrograms()
                self.app.navigation_controller.update_take_status()

                if self.app.window.info_panel_visible:
                    self.app.display_controller.update_info_panel()

                self.app.display_controller.set_status(
                    "Recording deleted (all audio removed)", MsgType.TEMPORARY
                )
                return True

            # Save the edited audio
            subtype = self._get_audio_subtype()
            self.app.file_manager.save_audio(filepath, audio_after, sr, subtype)

            # Create and push undo command with complete snapshots
            duration = (end_sample - start_sample) / sr
            cmd = AudioSnapshotCommand(
                filepath=filepath,
                sample_rate=sr,
                audio_before=audio_before.copy(),
                audio_after=audio_after.copy(),
                subtype=subtype,
                selection_start_time=sel_start,
                selection_end_time=sel_end,
                marker_after_edit=deletion_position,
                operation_description=f"Delete Range ({duration:.2f}s)",
            )
            self.undo_stack.push(cmd)

            # Clear selection
            selection_state.clear_all()
            if self.app.window and self.app.window.mel_spectrogram:
                self.app.window.mel_spectrogram.selection_visualizer.clear()

            # Refresh display
            self.app.display_controller.refresh_recording_preserving_zoom()

            # Set marker at deletion position so user can review the edit
            if deletion_position is not None:
                self._set_marker(deletion_position)

            self.app.display_controller.set_status(
                "Selection deleted", MsgType.TEMPORARY
            )
            return True

        except (OSError, ValueError) as e:
            self.app.display_controller.set_status(
                f"Error deleting selection: {e}", MsgType.ERROR
            )
            return False

    def can_insert_at_marker(self) -> bool:
        """Check if insert at marker is possible.

        Returns:
            True if a marker is set and a recording exists
        """
        selection_state = self._get_selection_state()
        if not selection_state or not selection_state.has_marker:
            return False

        label, take, filepath = self._get_current_recording_info()
        return filepath is not None

    def can_replace_selection(self) -> bool:
        """Check if replace selection is possible.

        Returns:
            True if a selection exists and a recording exists
        """
        selection_state = self._get_selection_state()
        if not selection_state or not selection_state.has_selection:
            return False

        label, take, filepath = self._get_current_recording_info()
        return filepath is not None

    def prepare_insert_recording(self) -> bool:
        """Prepare for insert recording at marker position.

        Call this before starting a new recording that will be inserted.

        Returns:
            True if insert mode was activated, False otherwise
        """
        selection_state = self._get_selection_state()
        if not selection_state or not selection_state.has_marker:
            return False

        self._pending_insert_position = selection_state.marker_position
        return True

    def prepare_replace_recording(self) -> bool:
        """Prepare for replace recording at selection.

        Call this before starting a new recording that will replace the selection.

        Returns:
            True if replace mode was activated, False otherwise
        """
        selection_state = self._get_selection_state()
        if not selection_state or not selection_state.has_selection:
            return False

        self._pending_replace_range = (
            selection_state.selection_start,
            selection_state.selection_end,
        )
        return True

    def finalize_insert_recording(
        self, new_audio: "np.ndarray", sr: int, target_filepath=None
    ) -> tuple:
        """Finalize insert by merging new recording at marker position.

        Args:
            new_audio: The newly recorded audio to insert
            sr: Sample rate of the new recording
            target_filepath: Optional path to the file to edit (if not provided,
                uses current recording)

        Returns:
            Tuple of (success, start_time, end_time) where start_time and end_time
            define the range of the newly inserted audio (for selection).
            Returns (False, None, None) on failure.
        """
        if self._pending_insert_position is None:
            return (False, None, None)

        if target_filepath is None:
            label, take, filepath = self._get_current_recording_info()
        else:
            filepath = target_filepath

        if not filepath or not filepath.exists():
            self._pending_insert_position = None
            return (False, None, None)

        try:
            # Load original audio (audio_before)
            audio_before, original_sr = self.app.file_manager.load_audio(filepath)

            # Convert insert position to samples
            insert_sample = int(self._pending_insert_position * original_sr)

            # Insert the new audio (creates audio_after with cross-fade)
            audio_after = AudioEditor.insert_at_position(
                audio_before, new_audio, insert_sample, original_sr
            )

            # Calculate the range of newly inserted audio
            insert_start = self._pending_insert_position
            insert_duration = len(new_audio) / original_sr
            insert_end = insert_start + insert_duration

            # Save
            subtype = self._get_audio_subtype()
            self.app.file_manager.save_audio(
                filepath, audio_after, original_sr, subtype
            )

            # Create and push undo command with complete snapshots
            cmd = AudioSnapshotCommand(
                filepath=filepath,
                sample_rate=original_sr,
                audio_before=audio_before.copy(),
                audio_after=audio_after.copy(),
                subtype=subtype,
                operation_description=f"Insert ({insert_duration:.2f}s)",
            )
            self.undo_stack.push(cmd)

            # Clear pending state
            self._pending_insert_position = None

            self.app.display_controller.set_status("Audio inserted", MsgType.TEMPORARY)
            return (True, insert_start, insert_end)

        except (OSError, ValueError) as e:
            self._pending_insert_position = None
            self.app.display_controller.set_status(
                f"Error inserting audio: {e}", MsgType.ERROR
            )
            return (False, None, None)

    def finalize_replace_recording(
        self, new_audio: "np.ndarray", sr: int, target_filepath=None
    ) -> tuple:
        """Finalize replace by replacing selection with new recording.

        Args:
            new_audio: The newly recorded audio to replace with
            sr: Sample rate of the new recording
            target_filepath: Optional path to the file to edit (if not provided,
                uses current recording)

        Returns:
            Tuple of (success, start_time, end_time) where start_time and end_time
            define the range of the newly inserted audio (for selection).
            Returns (False, None, None) on failure.
        """
        if self._pending_replace_range is None:
            return (False, None, None)

        if target_filepath is None:
            label, take, filepath = self._get_current_recording_info()
        else:
            filepath = target_filepath

        if not filepath or not filepath.exists():
            self._pending_replace_range = None
            return (False, None, None)

        try:
            # Load original audio (audio_before)
            audio_before, original_sr = self.app.file_manager.load_audio(filepath)

            # Convert range to samples
            start_time, end_time = self._pending_replace_range
            start_sample = int(start_time * original_sr)
            end_sample = int(end_time * original_sr)

            # Replace the range (creates audio_after with cross-fade)
            audio_after = AudioEditor.replace_range(
                audio_before, new_audio, start_sample, end_sample, original_sr
            )

            # Calculate the range of newly inserted audio
            replace_start = start_time
            replace_duration = len(new_audio) / original_sr
            replace_end = replace_start + replace_duration

            # Save
            subtype = self._get_audio_subtype()
            self.app.file_manager.save_audio(
                filepath, audio_after, original_sr, subtype
            )

            # Create and push undo command with complete snapshots
            original_duration = (end_sample - start_sample) / original_sr
            cmd = AudioSnapshotCommand(
                filepath=filepath,
                sample_rate=original_sr,
                audio_before=audio_before.copy(),
                audio_after=audio_after.copy(),
                subtype=subtype,
                operation_description=f"Replace Range ({original_duration:.2f}s)",
            )
            self.undo_stack.push(cmd)

            # Clear pending state
            self._pending_replace_range = None

            self.app.display_controller.set_status("Audio replaced", MsgType.TEMPORARY)
            return (True, replace_start, replace_end)

        except (OSError, ValueError) as e:
            self._pending_replace_range = None
            self.app.display_controller.set_status(
                f"Error replacing audio: {e}", MsgType.ERROR
            )
            return (False, None, None)

    def cancel_pending_edit(self) -> None:
        """Cancel any pending insert or replace operation."""
        self._pending_insert_position = None
        self._pending_replace_range = None

    def has_pending_edit(self) -> bool:
        """Check if there's a pending edit operation.

        Returns:
            True if an insert or replace is pending
        """
        return (
            self._pending_insert_position is not None
            or self._pending_replace_range is not None
        )

    def _clear_selection(self) -> None:
        """Clear selection state and visualizer."""
        selection_state = self._get_selection_state()
        if selection_state:
            selection_state.clear_all()
        if self.app.window and self.app.window.mel_spectrogram:
            self.app.window.mel_spectrogram.selection_visualizer.clear()
            self.app.window.mel_spectrogram.draw_idle()

    def _set_marker(self, time: float) -> None:
        """Set marker at a specific time position.

        Also adjusts the view to ensure the marker is visible.

        Args:
            time: Marker position in seconds
        """
        if not self.app.window or not self.app.window.mel_spectrogram:
            return

        spec = self.app.window.mel_spectrogram
        zoom_ctrl = spec.zoom_controller

        # Adjust view to show the marker if needed
        visible_seconds = zoom_ctrl.get_visible_seconds()
        view_start = zoom_ctrl.view_offset
        view_end = view_start + visible_seconds

        marker_visible = view_start <= time <= view_end

        if not marker_visible:
            # Center the marker in the view
            new_offset = time - (visible_seconds / 2)

            # Clamp to valid range
            recording_duration = spec.recording_display.recording_duration
            max_offset = max(0.0, recording_duration - visible_seconds)
            new_offset = max(0.0, min(new_offset, max_offset))

            zoom_ctrl.view_offset = new_offset
            spec._update_after_zoom()

        # Set the marker
        spec._set_marker(time)

        # Force immediate redraw
        if spec.canvas:
            spec.canvas.draw()

    def _select_range(self, start_time: float, end_time: float) -> None:
        """Set selection to a specific range and update visualizer.

        Also adjusts the view to ensure the selection is visible.

        Args:
            start_time: Selection start in seconds
            end_time: Selection end in seconds
        """
        if not self.app.window or not self.app.window.mel_spectrogram:
            return

        spec = self.app.window.mel_spectrogram
        zoom_ctrl = spec.zoom_controller

        # Adjust view to show the selection
        visible_seconds = zoom_ctrl.get_visible_seconds()
        view_start = zoom_ctrl.view_offset
        view_end = view_start + visible_seconds

        # Check if selection is visible
        selection_visible = start_time >= view_start and end_time <= view_end

        if not selection_visible:
            # Center the selection in the view if possible
            selection_center = (start_time + end_time) / 2
            new_offset = selection_center - (visible_seconds / 2)

            # Clamp to valid range
            recording_duration = spec.recording_display.recording_duration
            max_offset = max(0.0, recording_duration - visible_seconds)
            new_offset = max(0.0, min(new_offset, max_offset))

            zoom_ctrl.view_offset = new_offset
            spec._update_after_zoom()

        # Now set the selection
        spec._set_selection(start_time, end_time)

        # Force immediate redraw
        if spec.canvas:
            spec.canvas.draw()

    def undo(self) -> bool:
        """Undo the last edit operation.

        Returns:
            True if undo was successful, False otherwise
        """
        if not self.undo_stack.can_undo():
            self._beep()
            return False

        cmd = self.undo_stack.undo(self.app.file_manager)
        if cmd:
            if self._is_clip_command(cmd):
                self._refresh_after_clip_change(cmd)
            else:
                self.app.display_controller.refresh_recording_preserving_zoom()
                self._restore_selection_from_command(cmd)
            self.app.display_controller.set_status(
                f"Undo: {cmd.description()}", MsgType.TEMPORARY
            )
            return True

        self.app.display_controller.set_status("Undo failed", MsgType.ERROR)
        return False

    def redo(self) -> bool:
        """Redo the last undone operation.

        Returns:
            True if redo was successful, False otherwise
        """
        if not self.undo_stack.can_redo():
            self._beep()
            return False

        cmd = self.undo_stack.redo(self.app.file_manager)
        if cmd:
            if self._is_clip_command(cmd):
                self._refresh_after_clip_change(cmd)
            else:
                self.app.display_controller.refresh_recording_preserving_zoom()
                self._restore_marker_from_command(cmd)
            self.app.display_controller.set_status(
                f"Redo: {cmd.description()}", MsgType.TEMPORARY
            )
            return True

        self.app.display_controller.set_status("Redo failed", MsgType.ERROR)
        return False

    def clear_undo_history(self) -> None:
        """Clear the undo/redo history.

        Call this when switching to a different audio clip.
        """
        self.undo_stack.clear()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.undo_stack.can_undo()

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.undo_stack.can_redo()

    def _restore_selection_from_command(self, cmd) -> None:
        """Restore selection boundaries from a command if available.

        Args:
            cmd: The command that may contain selection boundaries
        """
        start = getattr(cmd, "selection_start_time", None)
        end = getattr(cmd, "selection_end_time", None)
        if start is not None and end is not None:
            self._select_range(start, end)
        else:
            self._clear_selection()

    def _restore_marker_from_command(self, cmd) -> None:
        """Restore marker position from a command if available.

        Args:
            cmd: The command that may contain a marker position
        """
        marker = getattr(cmd, "marker_after_edit", None)
        if marker is not None:
            self._clear_selection()
            self._set_marker(marker)
        else:
            self._clear_selection()

    def _is_clip_command(self, cmd) -> bool:
        """Check if command is a clip-level operation (trash/restore)."""
        return isinstance(cmd, (TrashClipCommand, RestoreFromTrashCommand))

    def _refresh_after_clip_change(self, cmd) -> None:
        """Refresh UI after a clip was trashed or restored.

        Args:
            cmd: The clip command that was executed
        """
        label = cmd.label
        take = cmd.take

        # Update active_recordings cache
        if self.app.active_recordings:
            if isinstance(cmd, TrashClipCommand):
                # Clip was trashed - remove from cache
                self.app.active_recordings.on_recording_deleted(label, take)
            else:
                # Clip was restored - refresh cache
                self.app.active_recordings.on_recording_restored(label)

            # Update takes from active recordings
            self.app.state.recording.takes = self.app.active_recordings.get_all_takes()
            existing_takes = self.app.active_recordings.get_existing_takes(label)
        else:
            existing_takes = []

        # Update displayed take
        if isinstance(cmd, RestoreFromTrashCommand):
            # Show the restored take
            self.app.state.recording.set_displayed_take(label, take)
        elif existing_takes:
            # Show highest remaining take
            self.app.state.recording.set_displayed_take(label, existing_takes[-1])
        else:
            # No takes left
            self.app.state.recording.set_displayed_take(label, 0)

        # Update display
        self.app.display_controller.show_saved_recording()
        self.app.navigation_controller.update_take_status()

        if self.app.window.info_panel_visible:
            self.app.display_controller.update_info_panel()

    def _beep(self) -> None:
        """Play system beep sound."""
        if self.app.window:
            self.app.window.window.bell()

    def _get_audio_subtype(self) -> str:
        """Get audio subtype based on current config.

        Returns:
            Audio subtype string for soundfile
        """
        bit_depth = self.app.config.audio.bit_depth
        if bit_depth == 16:
            return FileConstants.PCM_16_SUBTYPE
        elif bit_depth == 24:
            return FileConstants.PCM_24_SUBTYPE
        else:
            return FileConstants.FLAC_SUBTYPE
