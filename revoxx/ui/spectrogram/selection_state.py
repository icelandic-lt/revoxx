"""Selection state management for spectrogram display.

This module provides the data model for position markers and
range selections in the spectrogram view.
"""

from typing import Optional, Tuple


class SelectionState:
    """Manages marker position and selection range state.

    This class tracks:
    - A single position marker (for playback start position)
    - A selection range (start/end times for editing operations)

    All time values are in seconds.
    """

    def __init__(self):
        """Initialize selection state."""
        self._marker_position: Optional[float] = None
        self._selection_start: Optional[float] = None
        self._selection_end: Optional[float] = None

    @property
    def marker_position(self) -> Optional[float]:
        """Get current marker position in seconds."""
        return self._marker_position

    @property
    def selection_start(self) -> Optional[float]:
        """Get selection start time in seconds."""
        return self._selection_start

    @property
    def selection_end(self) -> Optional[float]:
        """Get selection end time in seconds."""
        return self._selection_end

    @property
    def has_marker(self) -> bool:
        """Check if a marker is set."""
        return self._marker_position is not None

    @property
    def has_selection(self) -> bool:
        """Check if a selection range is set."""
        return (
            self._selection_start is not None
            and self._selection_end is not None
        )

    def set_marker(self, time_seconds: float) -> None:
        """Set marker position and clear any selection.

        Args:
            time_seconds: Position in seconds
        """
        self._marker_position = max(0.0, time_seconds)
        self._selection_start = None
        self._selection_end = None

    def clear_marker(self) -> None:
        """Clear the marker position."""
        self._marker_position = None

    def set_selection(self, start: float, end: float) -> None:
        """Set selection range and clear marker.

        The start and end values are normalized so start <= end.

        Args:
            start: Selection start time in seconds
            end: Selection end time in seconds
        """
        start = max(0.0, start)
        end = max(0.0, end)

        if start > end:
            start, end = end, start

        self._selection_start = start
        self._selection_end = end
        self._marker_position = None

    def clear_selection(self) -> None:
        """Clear the selection range."""
        self._selection_start = None
        self._selection_end = None

    def clear_all(self) -> None:
        """Clear both marker and selection."""
        self._marker_position = None
        self._selection_start = None
        self._selection_end = None

    def get_play_start_position(self) -> float:
        """Get the playback start position.

        Returns:
            Marker position if set, selection start if selection exists,
            or 0.0 for start of recording.
        """
        if self._marker_position is not None:
            return self._marker_position
        if self._selection_start is not None:
            return self._selection_start
        return 0.0

    def get_play_range(self, duration: float) -> Tuple[float, float]:
        """Get the playback range.

        Args:
            duration: Total recording duration in seconds

        Returns:
            Tuple of (start, end) times in seconds.
            Returns selection range if set, otherwise (marker_position, duration)
            if marker is set, otherwise (0, duration).
        """
        if self.has_selection:
            return (self._selection_start, min(self._selection_end, duration))

        start = self.get_play_start_position()
        return (start, duration)

    def get_selection_samples(self, sample_rate: int) -> Optional[Tuple[int, int]]:
        """Get selection range in samples.

        Args:
            sample_rate: Audio sample rate in Hz

        Returns:
            Tuple of (start_sample, end_sample) or None if no selection.
        """
        if not self.has_selection:
            return None

        start_sample = int(self._selection_start * sample_rate)
        end_sample = int(self._selection_end * sample_rate)
        return (start_sample, end_sample)
