"""Synchronized playback handler using shared state.

This module provides hardware-synchronized position updates from the
struct shared state.
"""

import sys
import tkinter as tk
from typing import Optional, TYPE_CHECKING
import time
from matplotlib.lines import Line2D

from ...constants import UIConstants
from .controllers import PlaybackController, ZoomController

from ...audio.shared_state import (
    SHARED_STATUS_INVALID,
    PLAYBACK_STATUS_COMPLETED,
    PLAYBACK_STATUS_FINISHING,
    PLAYBACK_STATUS_IDLE,
)

if TYPE_CHECKING:
    from ...audio.shared_state import SharedState


class PlaybackHandler:
    """Handles playback visualization synchronized with hardware timing.

    Instead of using time.time() for animation, this handler reads
    the current playback position from the shared audio state, ensuring
    accurate synchronization with the actual audio playback.
    """

    def __init__(
        self,
        parent_widget: tk.Widget,
        ax,
        playback_controller: PlaybackController,
        zoom_controller: ZoomController,
        spec_frames: int,
        shared_audio_state: "SharedState",
    ):
        """Initialize synchronized playback handler.

        Args:
            parent_widget: Parent tkinter widget for scheduling
            ax: Matplotlib axes for drawing
            playback_controller: Playback state controller
            zoom_controller: Zoom state controller
            spec_frames: Number of display frames
            shared_audio_state: Struct-based shared state for position info
        """
        self.parent = parent_widget
        self.ax = ax
        self.playback_controller = playback_controller
        self.zoom_controller = zoom_controller
        self.spec_frames = spec_frames
        self.shared_audio_state = shared_audio_state

        # Playback visualization
        self.playback_line: Optional[Line2D] = None
        self.animation_id: Optional[str] = None
        self._fade_id: Optional[str] = None

        # Store total duration for calculations
        self.total_duration = 0.0
        self.sample_rate = 48000
        self._start_position = 0.0
        self._end_position: Optional[float] = None

        # Callbacks
        self.on_update_display = None
        self.on_update_time_axis = None
        self.on_draw_idle = None
        self.on_playback_finished = None

        # Progress watchdog
        self._last_sample_position: int = 0
        self._last_progress_ts: float = time.monotonic()
        self._end_visualized: bool = False

    def start_playback(
        self,
        duration: float,
        recording_duration: float,
        sample_rate: int,
        start_position: float = 0.0,
        end_position: Optional[float] = None,
    ) -> None:
        """Start playback animation.

        Args:
            duration: Playback duration in seconds (full recording duration)
            recording_duration: Total recording duration (for zoom calculations)
            sample_rate: Sample rate of the audio being played
            start_position: Start position in seconds (default 0.0)
            end_position: End position in seconds (default None = play to end)
        """

        # Clear any pending animation
        if self.animation_id:
            self.parent.after_cancel(self.animation_id)
            self.animation_id = None

        self.playback_controller.start(duration, start_position, end_position)
        self.playback_controller.recording_duration = recording_duration
        self.total_duration = duration
        self.sample_rate = sample_rate
        self._start_position = start_position
        self._end_position = end_position

        # Reset fade/progress state
        if self._fade_id:
            try:
                self.parent.after_cancel(self._fade_id)
            except Exception:
                pass
            self._fade_id = None
        self._last_sample_position = 0
        self._last_progress_ts = time.monotonic()
        self._end_visualized = False

        # Update time axis for current zoom
        if recording_duration > 0:
            visible_seconds = recording_duration / self.zoom_controller.zoom_level
        else:
            visible_seconds = (
                UIConstants.SPECTROGRAM_DISPLAY_SECONDS
                / self.zoom_controller.zoom_level
            )

        # Calculate initial view offset based on start/end position
        current_view_offset = self.zoom_controller.view_offset
        current_view_end = current_view_offset + visible_seconds

        # If a selection is active (end_position set), don't change the view at all
        # The user controls the viewport manually while playback runs
        has_selection = end_position is not None

        if has_selection:
            # Selection active: keep current view, user has full control
            initial_view_offset = current_view_offset
            self.playback_controller.set_static_viewport(current_view_offset)
        else:
            # No selection (marker or full playback): check if start is visible
            start_is_visible = (
                start_position >= current_view_offset
                and start_position <= current_view_end
            )

            if start_is_visible:
                # Start position is already visible - keep current view
                initial_view_offset = current_view_offset
            elif start_position > 0:
                # Start position not visible - adjust view to show it
                half_visible = visible_seconds / 2
                if start_position < half_visible:
                    initial_view_offset = 0.0
                elif start_position > recording_duration - half_visible:
                    initial_view_offset = max(0.0, recording_duration - visible_seconds)
                else:
                    initial_view_offset = start_position - half_visible
            else:
                initial_view_offset = 0.0

            view_changed = abs(initial_view_offset - current_view_offset) > 0.001

            if view_changed:
                self.zoom_controller.view_offset = initial_view_offset

                if self.on_update_time_axis:
                    self.on_update_time_axis(initial_view_offset, initial_view_offset + visible_seconds)

                if self.on_update_display:
                    self.on_update_display()

        # Calculate initial X position for playback line
        initial_x_pos = 0
        if start_position > 0 and visible_seconds > 0:
            relative_pos = (start_position - initial_view_offset) / visible_seconds
            initial_x_pos = relative_pos * (self.spec_frames - 1)

        # Create playback line if needed
        if self.playback_line is None:
            self.playback_line = self.ax.axvline(
                x=initial_x_pos,
                color=UIConstants.COLOR_PLAYBACK_LINE,
                linewidth=UIConstants.PLAYBACK_LINE_WIDTH,
            )
        else:
            self.playback_line.set_xdata([initial_x_pos])
            self.playback_line.set_visible(True)
        # Ensure fully opaque on start
        if self.playback_line is not None:
            self.playback_line.set_alpha(1.0)

        # Start animation with small delay to ensure player is ready
        self.parent.after(
            UIConstants.PLAYBACK_INITIAL_CHECK_MS, self._update_playback_position
        )

    def stop_playback(self) -> None:
        """Stop playback animation."""
        self.playback_controller.stop()
        if self.animation_id:
            try:
                self.parent.after_cancel(self.animation_id)
            except ValueError:
                pass
            self.animation_id = None

        # Hide playback line
        if self.playback_line:
            self.playback_line.set_visible(False)
            # Ensure line is fully opaque for next run
            self.playback_line.set_alpha(1.0)

        if self.on_draw_idle:
            self.on_draw_idle()

    def _handle_playback_status(self, status: int) -> bool:
        """Handle playback status and determine if we should continue.

        Args:
            status: Current playback status

        Returns:
            True if should continue processing, False otherwise
        """
        if status == SHARED_STATUS_INVALID:
            print("ERROR: Playback state not initialized", file=sys.stderr)
            self.stop_playback()
            return False

        if status == PLAYBACK_STATUS_COMPLETED:
            # Playback finished – finalize visualization immediately
            self._finish_playback_visual()
            return False

        if status == PLAYBACK_STATUS_IDLE:
            # Player not yet started or already stopped
            if self.playback_controller.is_playing:
                # Give the player a moment to switch to PLAYING
                self.parent.after(
                    UIConstants.PLAYBACK_IDLE_RETRY_MS, self._update_playback_position
                )
            else:
                # Not playing – ensure visualization is stopped
                self.stop_playback()
            return False

        return True

    def _check_playback_watchdog(self, current_sample: int, total_samples: int) -> bool:
        """Check if playback is stalled near the end.

        Args:
            current_sample: Current sample position (relative to playback range start)
            total_samples: Total number of samples in playback range

        Returns:
            True if stalled and should finish, False otherwise
        """
        if current_sample != self._last_sample_position:
            self._last_sample_position = current_sample
            self._last_progress_ts = time.monotonic()
            return False

        elapsed = time.monotonic() - self._last_progress_ts

        # Calculate sample ratio (already relative to playback range)
        sample_ratio = (current_sample / total_samples) if total_samples > 0 else 0.0

        # Calculate time ratio relative to the playback range, not the full recording
        time_ratio = 0.0
        effective_start = self.playback_controller.start_offset
        effective_end = (
            self.playback_controller.end_position
            if self.playback_controller.end_position is not None
            else self.playback_controller.playback_duration
        )
        effective_duration = effective_end - effective_start

        if effective_duration > 0:
            progress = self.playback_controller.playback_position - effective_start
            time_ratio = min(1.0, progress / effective_duration)

        near_end = (sample_ratio >= 0.95) or (time_ratio >= 0.95)

        if elapsed > 0.20 and near_end and not self._end_visualized:
            self._finish_playback_visual()
            return True

        return False

    def _update_playback_position(self) -> None:
        """Update playback position from shared audio state."""
        # Check if still playing
        if not self.shared_audio_state:
            return

        playback_state = self.shared_audio_state.get_playback_state()
        status = playback_state.get("status", 0)

        # Handle status transitions
        if not self._handle_playback_status(status):
            return

        # Get current position from shared state
        current_sample = playback_state.get("current_sample_position", 0)
        total_samples = playback_state.get("total_samples", 1)

        # Check watchdog for stalled playback
        if self._check_playback_watchdog(current_sample, total_samples):
            return

        # Calculate and update position
        position_seconds = self._calculate_position_seconds(
            current_sample, total_samples, status
        )
        self.playback_controller.playback_position = position_seconds

        # Update animation and display
        self._update_animation_display(
            status, position_seconds, current_sample, total_samples
        )

    def _calculate_position_seconds(
        self, current_sample: int, total_samples: int, status: int
    ) -> float:
        """Calculate playback position in seconds.

        Args:
            current_sample: Current sample position (relative to start_sample)
            total_samples: Total number of samples in playback range
            status: Current playback status

        Returns:
            Absolute position in seconds within the full recording
        """
        # current_sample is relative to start position, so add start_position
        position_seconds = self._start_position
        if total_samples > 0:
            position_seconds += current_sample / self.sample_rate

        # Handle FINISHING status - override position to animate to end
        if status == PLAYBACK_STATUS_FINISHING:
            if self._end_position is not None:
                position_seconds = self._end_position
            else:
                position_seconds = self.total_duration

        return position_seconds

    def _update_animation_display(
        self,
        status: int,
        position_seconds: float,
        current_sample: int,
        total_samples: int,
    ) -> None:
        """Update animation parameters and display.

        Args:
            status: Current playback status
            position_seconds: Current position in seconds
            current_sample: Current sample position
            total_samples: Total number of samples
        """
        if self.playback_controller.playback_duration <= 0:
            return

        previous_view_offset = self.zoom_controller.view_offset

        x_pos, view_offset, visible_seconds = (
            self.playback_controller.calculate_animation_phase(
                self.zoom_controller.zoom_level,
                self.spec_frames,
                self.zoom_controller.view_offset,
            )
        )

        # Check if viewport has changed
        view_changed = abs(view_offset - previous_view_offset) > 0.001

        self.zoom_controller.view_offset = view_offset
        self.playback_line.set_xdata([x_pos])

        # Only update time axis and spectrogram if view actually changed
        if view_changed:
            if self.on_update_time_axis:
                self.on_update_time_axis(view_offset, view_offset + visible_seconds)

            if self.on_update_display:
                self.on_update_display()

        if self.on_draw_idle:
            self.on_draw_idle()

        # Handle continuing animation for FINISHING status
        if (
            status == PLAYBACK_STATUS_FINISHING
            and position_seconds < self.total_duration
        ):
            # Continue scheduling updates until we reach the end
            self._schedule_next_frame()
            return

        # Check if we've reached end_position (for partial playback)
        if self._end_position is not None and position_seconds >= self._end_position:
            self._finish_playback_visual()
            return

        if current_sample >= total_samples - 1 or status == PLAYBACK_STATUS_COMPLETED:
            self._finish_playback_visual()
        else:
            # Schedule next update
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        """Schedule next animation frame."""
        if self.animation_id:
            try:
                self.parent.after_cancel(self.animation_id)
            except ValueError:
                pass

        # Update interval
        update_interval = UIConstants.PLAYBACK_UPDATE_MS

        self.animation_id = self.parent.after(
            update_interval, self._update_playback_position
        )

    def _finish_playback_visual(self) -> None:
        """Snap the playback line to the end and fade it out smoothly."""
        from .controllers import ViewportMode

        if self._end_visualized:
            # Avoid double-trigger
            return
        self._end_visualized = True

        # Cancel further position updates
        if self.animation_id:
            try:
                self.parent.after_cancel(self.animation_id)
            except Exception:
                pass
            self.animation_id = None

        # Ensure we have a visible line
        if self.playback_line is None:
            self.playback_line = self.ax.axvline(
                x=0,
                color=UIConstants.COLOR_PLAYBACK_LINE,
                linewidth=UIConstants.PLAYBACK_LINE_WIDTH,
            )

        # Calculate visible seconds
        if self.playback_controller.recording_duration > 0:
            visible_seconds = (
                self.playback_controller.recording_duration
                / self.zoom_controller.zoom_level
            )
        else:
            visible_seconds = (
                UIConstants.SPECTROGRAM_DISPLAY_SECONDS
                / self.zoom_controller.zoom_level
            )

        # In STATIC mode, keep the view unchanged and position line at selection end
        if self.playback_controller.viewport_mode == ViewportMode.STATIC:
            end_position = self._end_position if self._end_position else self.total_duration
            self.playback_controller.playback_position = end_position
            view_offset = self.zoom_controller.view_offset
            time_from_view_start = end_position - view_offset
            x_pos_ratio = time_from_view_start / visible_seconds
            x_pos = x_pos_ratio * (self.spec_frames - 1)
            x_pos = max(0, min(x_pos, self.spec_frames - 1))
        else:
            # SCROLL mode: snap to end of recording
            self.playback_controller.playback_position = self.total_duration
            x_pos = self.spec_frames - 1
            end_view_offset = max(
                0.0,
                (self.playback_controller.recording_duration or self.total_duration)
                - visible_seconds,
            )
            self.zoom_controller.view_offset = end_view_offset

        self.playback_line.set_xdata([x_pos])
        self.playback_line.set_visible(True)

        # Only update time axis and display in SCROLL mode (view changed)
        if self.playback_controller.viewport_mode != ViewportMode.STATIC:
            if self.on_update_time_axis:
                self.on_update_time_axis(end_view_offset, end_view_offset + visible_seconds)
            if end_view_offset > 0 and self.on_update_display:
                self.on_update_display()

        if self.on_draw_idle:
            self.on_draw_idle()

        # Start fade-out
        self._start_fade_out()

    def _start_fade_out(self, duration_ms: int = None, steps: int = None) -> None:
        """Fade out the playback line over duration_ms in given steps."""
        if not self.playback_line:
            return
        duration_ms = (
            duration_ms if duration_ms is not None else UIConstants.PLAYBACK_FADEOUT_MS
        )
        steps = steps if steps is not None else UIConstants.PLAYBACK_FADEOUT_STEPS

        # Cancel existing fade if any
        if self._fade_id:
            try:
                self.parent.after_cancel(self._fade_id)
            except Exception:
                pass
            self._fade_id = None

        def step(i: int) -> None:
            if not self.playback_line:
                return
            alpha = max(0.0, 1.0 - (i / steps))
            self.playback_line.set_alpha(alpha)
            if self.on_draw_idle:
                self.on_draw_idle()
            if i < steps:
                self._fade_id = self.parent.after(
                    max(1, duration_ms // steps), lambda: step(i + 1)
                )
            else:
                # Hide and restore alpha for next playback
                self.playback_line.set_visible(False)
                self.playback_line.set_alpha(1.0)
                if self.on_draw_idle:
                    self.on_draw_idle()
                self._fade_id = None
                # Notify that playback has finished
                if self.on_playback_finished:
                    self.on_playback_finished()

        step(0)
