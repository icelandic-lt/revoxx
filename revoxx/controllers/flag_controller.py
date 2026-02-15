"""Flag controller for marking utterances during review."""

from typing import Optional, Dict, TYPE_CHECKING

from ..constants import FlagType, MsgType

if TYPE_CHECKING:
    from ..app import Revoxx


class FlagController:
    """Handles utterance flagging for review workflows.

    Supports two flag types:
    - needs_edit: utterance needs re-recording or editing
    - rejected: utterance should be excluded from dataset export
    """

    FLAG_DISPLAY = FlagType.DISPLAY

    def __init__(self, app: "Revoxx"):
        self.app = app

    def _get_current_label(self) -> Optional[str]:
        """Get the label of the currently displayed utterance."""
        return self.app.state.recording.current_label

    def _get_flags(self) -> Dict[str, str]:
        """Get the utterance flags dict from the current session."""
        if not self.app.current_session:
            return {}
        return self.app.current_session.utterance_flags

    def get_flag(self, label: str) -> Optional[str]:
        """Return the flag value for a label, or None if unflagged."""
        return self._get_flags().get(label)

    def get_flags(self) -> Dict[str, str]:
        """Return all flags."""
        return dict(self._get_flags())

    def toggle_needs_edit(self) -> None:
        """Toggle 'needs_edit' flag on the current utterance."""
        self._toggle(FlagType.NEEDS_EDIT)

    def toggle_rejected(self) -> None:
        """Toggle 'rejected' flag on the current utterance."""
        self._toggle(FlagType.REJECTED)

    def clear_flag(self) -> None:
        """Remove any flag from the current utterance."""
        label = self._get_current_label()
        if not label or not self.app.current_session:
            return

        flags = self._get_flags()
        if label in flags:
            del flags[label]
            self._save()
            self._update_status_after_flag(label)
            self.app.display_controller.set_status(
                f"Flag cleared: {label}", MsgType.TEMPORARY
            )
        else:
            self.app.display_controller.set_status(
                f"No flag on {label}", MsgType.TEMPORARY
            )

    def jump_to_next(self, flag_type: str) -> None:
        """Jump to the next utterance with the given flag type.

        Searches forward from the current position in sorted order,
        wrapping around if necessary.
        """
        if not self.app.current_session or not self.app.active_recordings:
            return

        flags = self._get_flags()
        current_index = self.app.state.recording.current_index

        # Collect all sorted indices
        sorted_indices = self.app.active_recordings.get_sorted_indices()
        if not sorted_indices:
            return

        # Find current position in sorted order
        try:
            current_pos = sorted_indices.index(current_index)
        except ValueError:
            current_pos = 0

        total = len(sorted_indices)
        labels = self.app.state.recording.labels

        # Search forward from current position, wrapping around
        for offset in range(1, total):
            pos = (current_pos + offset) % total
            idx = sorted_indices[pos]
            label = labels[idx]
            if flags.get(label) == flag_type:
                self.app.navigation_controller.find_utterance(idx)
                display_name = self.FLAG_DISPLAY.get(flag_type, flag_type)
                self.app.display_controller.set_status(
                    f"Jumped to {display_name}: {label}", MsgType.TEMPORARY
                )
                return

        # None found
        display_name = self.FLAG_DISPLAY.get(flag_type, flag_type)
        self.app.display_controller.set_status(
            f"No {display_name} utterances found", MsgType.TEMPORARY
        )

    def _toggle(self, flag_type: str) -> None:
        """Toggle a specific flag type on the current utterance."""
        label = self._get_current_label()
        if not label or not self.app.current_session:
            return

        flags = self._get_flags()
        display_name = self.FLAG_DISPLAY.get(flag_type, flag_type)

        if flags.get(label) == flag_type:
            del flags[label]
            self._save()
            self._update_status_after_flag(label)
            self.app.display_controller.set_status(
                f"Unflagged: {label}", MsgType.TEMPORARY
            )
        else:
            flags[label] = flag_type
            self._save()
            self._update_status_after_flag(label)
            self.app.display_controller.set_status(
                f"Flagged as {display_name}: {label}", MsgType.TEMPORARY
            )

    def _save(self) -> None:
        """Persist the current session to disk."""
        if self.app.current_session:
            self.app.current_session.save()

    def _update_status_after_flag(self, label: str) -> None:
        """Update the default status text to reflect the current flag state."""
        self.app.navigation_controller.update_take_status()
