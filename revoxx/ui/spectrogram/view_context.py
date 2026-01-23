"""View context for spectrogram display.

This module provides a container for view-related parameters that are
frequently passed together to visualization components.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from .controllers import ZoomController


@dataclass
class ViewContext:
    """Encapsulates view parameters for visualization updates.

    This class bundles the parameters that are repeatedly passed to
    visualization methods, reducing parameter count and making the
    code more maintainable.

    Attributes:
        spec_frames: Number of display frames (horizontal resolution)
        n_mels: Number of mel bins (vertical resolution)
        zoom_controller: Controller for zoom and pan state
        recording_duration: Total duration of the recording in seconds
    """

    spec_frames: int
    n_mels: int
    zoom_controller: ZoomController
    recording_duration: float

    @property
    def has_recording(self) -> bool:
        """Check if a recording is loaded."""
        return self.recording_duration > 0

    @property
    def visible_seconds(self) -> float:
        """Get the currently visible time range in seconds."""
        return self.zoom_controller.get_visible_seconds()

    @property
    def view_offset(self) -> float:
        """Get the current view offset in seconds."""
        return self.zoom_controller.view_offset


@dataclass
class SavedViewState:
    """Stores view state for restoration after playback.

    This class encapsulates the information needed to restore the view
    to its previous position after playback finishes.

    Attributes:
        target_time: The time position (marker or selection start) to restore
        relative_pos: The relative position (0.0 to 1.0) of target in viewport
    """

    target_time: Optional[float] = field(default=None)
    relative_pos: Optional[float] = field(default=None)

    @property
    def is_valid(self) -> bool:
        """Check if saved state contains valid data."""
        return self.target_time is not None and self.relative_pos is not None

    def save(self, target_time: float, relative_pos: float) -> None:
        """Save view state.

        Args:
            target_time: Time position to restore
            relative_pos: Relative position in viewport (0.0 to 1.0)
        """
        self.target_time = target_time
        self.relative_pos = relative_pos

    def clear(self) -> None:
        """Clear saved state."""
        self.target_time = None
        self.relative_pos = None

    def get(self) -> Optional[Tuple[float, float]]:
        """Get saved state if valid.

        Returns:
            Tuple of (target_time, relative_pos) or None if not valid
        """
        if self.is_valid:
            return (self.target_time, self.relative_pos)
        return None
