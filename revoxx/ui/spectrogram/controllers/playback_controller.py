"""Playback controller for spectrogram animation."""

import time
from enum import Enum
from typing import Optional, Tuple
from ....constants import UIConstants


class ViewportMode(Enum):
    """Viewport scrolling behavior during playback.

    SCROLL: Viewport follows playback position (3-phase animation)
    STATIC: Viewport stays fixed, only playback line moves
    """

    SCROLL = "scroll"
    STATIC = "static"


class PlaybackController:
    """Manages playback state and animation calculations.

    This controller handles the three-phase playback animation:
    1. Line moves from left to center
    2. Line stays at center while view scrolls
    3. Line moves from center to right

    When viewport_mode is STATIC, scrolling is disabled and the playback
    line simply moves across the current view.
    """

    def __init__(self):
        """Initialize playback controller."""
        self.is_playing = False
        self.playback_position = 0.0
        self.playback_duration = 0.0
        self.playback_start_time = 0.0
        self.recording_duration = 0.0
        self.start_offset = 0.0
        self.end_position: Optional[float] = None
        self.viewport_mode = ViewportMode.SCROLL
        self.static_view_offset: Optional[float] = None

    def start(
        self,
        duration: float,
        start_offset: float = 0.0,
        end_position: Optional[float] = None,
    ) -> None:
        """Start playback with given duration.

        Args:
            duration: Total playback duration in seconds
            start_offset: Start position in seconds (default 0.0)
            end_position: End position in seconds (default None = play to end)
        """
        self.is_playing = True
        self.playback_duration = duration
        self.start_offset = start_offset
        self.end_position = end_position
        self.playback_position = start_offset
        self.playback_start_time = time.time()
        self.viewport_mode = ViewportMode.SCROLL
        self.static_view_offset = None

    def set_static_viewport(self, view_offset: float) -> None:
        """Lock viewport to a fixed offset (no scrolling during playback).

        Args:
            view_offset: The view offset to lock to
        """
        self.viewport_mode = ViewportMode.STATIC
        self.static_view_offset = view_offset

    def stop(self) -> None:
        """Stop playback."""
        self.is_playing = False
        self.playback_position = 0.0
        self.playback_duration = 0.0
        self.start_offset = 0.0
        self.end_position = None
        self.viewport_mode = ViewportMode.SCROLL
        self.static_view_offset = None

    def update_position(self) -> float:
        """Update and return current playback position.

        Returns:
            Current playback position in seconds
        """
        if self.is_playing:
            actual_elapsed = time.time() - self.playback_start_time
            self.playback_position = actual_elapsed
        return self.playback_position

    def calculate_animation_phase(
        self, zoom_level: float, spec_frames: int, current_view_offset: float = 0.0
    ) -> Tuple[float, float, float]:
        """Calculate playback animation parameters.

        Args:
            zoom_level: Current zoom level
            spec_frames: Number of display frames
            current_view_offset: Current view offset (used as fallback)

        Returns:
            Tuple of (x_position, view_offset, visible_seconds)
        """
        if self.playback_duration <= 0:
            return 0.0, 0.0, UIConstants.SPECTROGRAM_DISPLAY_SECONDS

        # Calculate visible window based on recording duration and zoom
        visible_seconds = (
            self.recording_duration / zoom_level
            if self.recording_duration > 0
            else UIConstants.SPECTROGRAM_DISPLAY_SECONDS / zoom_level
        )

        # If viewport is locked (STATIC mode), use the fixed offset
        if (
            self.viewport_mode == ViewportMode.STATIC
            and self.static_view_offset is not None
        ):
            view_offset = self.static_view_offset
            time_from_view_start = self.playback_position - view_offset
            x_pos_ratio = time_from_view_start / visible_seconds
            x_pos = x_pos_ratio * (spec_frames - 1)
            x_pos = max(0, min(x_pos, spec_frames - 1))
            return x_pos, view_offset, visible_seconds

        # Determine effective playback range
        effective_end = (
            self.end_position
            if self.end_position is not None
            else self.playback_duration
        )
        effective_start = self.start_offset
        effective_duration = effective_end - effective_start

        # Special case: when zoomed out to show full recording (1x zoom)
        if visible_seconds >= self.playback_duration:
            view_offset = 0.0
            x_pos_ratio = self.playback_position / self.playback_duration
            x_pos = x_pos_ratio * (spec_frames - 1)

        # Special case: selection/range fits entirely within current visible window
        elif effective_duration <= visible_seconds:
            current_view_end = current_view_offset + visible_seconds
            selection_visible = (
                effective_start >= current_view_offset
                and effective_end <= current_view_end
            )

            if selection_visible:
                view_offset = current_view_offset
            else:
                view_offset = max(0.0, effective_start)
                if view_offset + visible_seconds > self.recording_duration:
                    view_offset = max(0.0, self.recording_duration - visible_seconds)

            time_from_view_start = self.playback_position - view_offset
            x_pos_ratio = time_from_view_start / visible_seconds
            x_pos = x_pos_ratio * (spec_frames - 1)
        else:
            # Three-phase animation for zoomed views
            half_visible = visible_seconds / 2

            # Determine phase 1 view offset (where view starts)
            current_view_end = current_view_offset + visible_seconds
            pos_visible_in_current = (
                self.playback_position >= current_view_offset
                and self.playback_position <= current_view_end
            )

            if pos_visible_in_current:
                phase1_view = current_view_offset
            else:
                phase1_view = max(0.0, effective_start)

            # Phase boundaries based on view position, not start position
            # Phase 1 ends when position reaches center of phase1_view
            phase1_center = phase1_view + half_visible

            # Phase 3 starts when view can no longer scroll (at end)
            max_view_offset = max(0.0, self.recording_duration - visible_seconds)
            phase3_start = max_view_offset + half_visible

            if self.playback_position < phase1_center:
                # Phase 1: View stays fixed, line moves toward center
                view_offset = phase1_view
                time_from_view_start = self.playback_position - view_offset
                x_pos_ratio = time_from_view_start / visible_seconds
                x_pos = x_pos_ratio * (spec_frames - 1)

            elif self.playback_position < phase3_start:
                # Phase 2: Line stays at center, view scrolls
                view_offset = self.playback_position - half_visible
                view_offset = max(0.0, min(view_offset, max_view_offset))
                x_pos = (spec_frames - 1) * 0.5

            else:
                # Phase 3: View at end, line moves from center to right
                view_offset = max_view_offset
                time_from_view_start = self.playback_position - view_offset
                x_pos_ratio = time_from_view_start / visible_seconds
                x_pos = x_pos_ratio * (spec_frames - 1)

        x_pos = max(0, min(x_pos, spec_frames - 1))

        return x_pos, view_offset, visible_seconds

    def is_finished(self) -> bool:
        """Check if playback has finished.

        Returns:
            True if playback position exceeds duration or end_position
        """
        if not self.is_playing:
            return False

        if self.end_position is not None:
            return self.playback_position >= self.end_position

        return self.playback_position >= self.playback_duration
