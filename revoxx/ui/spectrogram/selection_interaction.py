"""Selection interaction handler for spectrogram widget.

This module handles mouse interactions for markers and selections.
"""

from typing import Optional, TYPE_CHECKING

from ...constants import UIConstants

if TYPE_CHECKING:
    from .widget import MelSpectrogramWidget


class SelectionInteractionHandler:
    """Handles mouse interactions for marker and selection operations.

    This class manages:
    - Left-click to set position markers
    - Click-and-drag to create selections
    - Dragging markers to resize selections or move position markers
    - Hover detection for resize cursor
    """

    def __init__(self, widget: "MelSpectrogramWidget"):
        """Initialize the selection interaction handler.

        Args:
            widget: Parent spectrogram widget
        """
        self.widget = widget

        # Drag state for new selections
        self._drag_start_x: Optional[float] = None
        self._drag_start_time: Optional[float] = None

        # Marker resize state
        self._resize_target: Optional[str] = None  # "start", "end", "position"
        self._resize_active: bool = False

    # --- Properties for cleaner access ---

    @property
    def _selection_state(self):
        """Access widget's selection state."""
        return self.widget.selection_state

    @property
    def _visualizer(self):
        """Access widget's selection visualizer."""
        return self.widget.selection_visualizer

    @property
    def _recording_duration(self) -> float:
        """Get current recording duration."""
        return self.widget.recording_display.recording_duration

    @property
    def _view_context(self):
        """Get current view context."""
        return self.widget.view_context

    # --- Mouse event handlers ---

    def on_mouse_motion(self, event) -> None:
        """Handle mouse motion for marker hover detection."""
        if self._resize_active:
            return

        marker = self._get_marker_at_position(event.x)
        if marker:
            self._resize_target = marker
            self._set_cursor_resize()
        else:
            self._resize_target = None
            self._set_cursor_default()

    def on_left_click(self, event) -> None:
        """Handle left mouse button press for marker/selection or resize."""
        if event.type == UIConstants.TK_EVENT_BUTTON_PRESS:
            mouse_rel_x = self.widget.get_mouse_position_in_axes(event)

            # Check if starting a resize operation
            if self._resize_target:
                self._resize_active = True
                self._drag_start_x = event.x
                return

            # Store drag start position for new selection
            self._drag_start_x = event.x
            self._drag_start_time = self._pixel_to_time(mouse_rel_x)

    def on_left_drag(self, event) -> None:
        """Handle left mouse button drag for selection preview or resize."""
        if self._resize_active and self._resize_target:
            self._handle_marker_drag(event)
            return

        if not self._is_drag_started():
            return

        if not self._drag_exceeds_threshold(event):
            return

        self._clear_marker_for_new_selection()
        self._update_selection_preview(event)

    def _is_drag_started(self) -> bool:
        """Check if a drag operation has been initiated."""
        return self._drag_start_x is not None and self._drag_start_time is not None

    def _drag_exceeds_threshold(self, event) -> bool:
        """Check if drag distance exceeds the minimum threshold."""
        dx = abs(event.x - self._drag_start_x)
        return dx >= UIConstants.SELECTION_DRAG_THRESHOLD

    def _clear_marker_for_new_selection(self) -> None:
        """Clear any existing marker when starting a new selection."""
        if self._selection_state.has_marker:
            self._selection_state.clear_marker()
            self._visualizer.update_marker(None, self._view_context)

    def _update_selection_preview(self, event) -> None:
        """Update the selection preview during drag."""
        mouse_rel_x = self.widget.get_mouse_position_in_axes(event)
        current_time = self._pixel_to_time(mouse_rel_x)
        if current_time is None:
            return

        ctx = self._view_context
        if not ctx.has_recording:
            return

        current_time = max(0.0, min(current_time, ctx.recording_duration))
        start_time = min(self._drag_start_time, current_time)
        end_time = max(self._drag_start_time, current_time)

        self._visualizer.update_selection(start_time, end_time, ctx)
        self.widget.draw_idle()

    def on_left_release(self, event) -> None:
        """Handle left mouse button release to finalize marker, selection, or resize."""
        if self._resize_active:
            self._finish_resize()
            return

        if self._drag_start_x is None:
            return

        if self._drag_exceeds_threshold(event):
            self._finalize_selection(event)
        else:
            self._finalize_marker(event)

        self._clear_drag_state()

    def _finish_resize(self) -> None:
        """Complete a resize operation and reset state."""
        self._resize_active = False
        self._resize_target = None
        self._set_cursor_default()
        self._drag_start_x = None

    def _finalize_marker(self, event) -> None:
        """Set marker at the click position."""
        mouse_rel_x = self.widget.get_mouse_position_in_axes(event)
        time_seconds = self._pixel_to_time(mouse_rel_x)
        if time_seconds is not None:
            self.set_marker(time_seconds)

    def _finalize_selection(self, event) -> None:
        """Finalize the selection range after drag."""
        mouse_rel_x = self.widget.get_mouse_position_in_axes(event)
        if self._drag_start_time is not None:
            end_time = self._pixel_to_time(mouse_rel_x)
            if end_time is not None:
                self.set_selection(self._drag_start_time, end_time)

    def _clear_drag_state(self) -> None:
        """Reset drag state after operation completes."""
        self._drag_start_x = None
        self._drag_start_time = None

    def _handle_marker_drag(self, event) -> None:
        """Handle dragging a marker (selection or position)."""
        new_time = self._get_time_from_event(event)
        if new_time is None or self._recording_duration <= 0:
            return

        if self._resize_target == "position":
            self._drag_position_marker(new_time)
        else:
            self._resize_selection(new_time)

    def _get_time_from_event(self, event) -> Optional[float]:
        """Convert mouse event to time position."""
        mouse_rel_x = self.widget.get_mouse_position_in_axes(event)
        return self._pixel_to_time(mouse_rel_x)

    def _drag_position_marker(self, new_time: float) -> None:
        """Move the position marker to a new time."""
        new_position = max(0.0, min(new_time, self._recording_duration))
        self._selection_state.set_marker(new_position)
        self._visualizer.update_marker(new_position, self._view_context)
        self.widget.draw_idle()

    def _resize_selection(self, new_time: float) -> None:
        """Resize the selection by moving start or end marker."""
        start = self._selection_state.selection_start
        end = self._selection_state.selection_end

        if start is None or end is None:
            return

        min_gap = 0.001  # Minimum 1ms selection

        if self._resize_target == "start":
            new_start = max(0.0, min(new_time, end - min_gap))
            self._selection_state.set_selection(new_start, end)
        elif self._resize_target == "end":
            new_end = max(start + min_gap, min(new_time, self._recording_duration))
            self._selection_state.set_selection(start, new_end)

        self._visualizer.update_selection(
            self._selection_state.selection_start,
            self._selection_state.selection_end,
            self._view_context,
        )
        self.widget.draw_idle()

    # --- Marker detection ---

    def _get_marker_at_position(self, pixel_x: float) -> Optional[str]:
        """Check if pixel position is near a marker (selection or position).

        Args:
            pixel_x: X position in pixels

        Returns:
            "start", "end", "position", or None
        """
        if self._recording_duration <= 0:
            return None

        threshold = UIConstants.MARKER_HOVER_THRESHOLD

        # Check selection markers first
        if self._selection_state.has_selection:
            start_pixel = self._time_to_drawn_pixel_x(
                self._selection_state.selection_start,
                UIConstants.SELECTION_LINE_OFFSET,
            )
            if start_pixel is not None and abs(pixel_x - start_pixel) <= threshold:
                return "start"

            end_pixel = self._time_to_drawn_pixel_x(
                self._selection_state.selection_end,
                UIConstants.SELECTION_LINE_OFFSET,
            )
            if end_pixel is not None and abs(pixel_x - end_pixel) <= threshold:
                return "end"

        # Check position marker
        if self._selection_state.has_marker:
            marker_pixel = self._time_to_drawn_pixel_x(
                self._selection_state.marker_position,
                UIConstants.POSITION_MARKER_OFFSET,
            )
            if marker_pixel is not None and abs(pixel_x - marker_pixel) <= threshold:
                return "position"

        return None

    # --- Coordinate conversion ---

    def _time_to_pixel_x(self, time_seconds: float) -> Optional[float]:
        """Convert time in seconds to pixel X position.

        Args:
            time_seconds: Time position in seconds

        Returns:
            Pixel X position, or None if outside visible range.
        """
        if self._recording_duration <= 0:
            return None

        view_start, view_end = self.widget.visible_time_range
        visible_seconds = view_end - view_start

        # Clamp time to visible range (with small epsilon for edge cases)
        epsilon = 0.001
        if time_seconds < view_start - epsilon or time_seconds > view_end + epsilon:
            return None

        # Calculate relative position, clamped to [0, 1]
        rel_x = (time_seconds - view_start) / visible_seconds
        rel_x = max(0.0, min(1.0, rel_x))

        ax_left, ax_width = self.widget.axes_pixel_bounds
        return ax_left + rel_x * ax_width

    def _time_to_drawn_pixel_x(
        self, time_seconds: float, line_offset: float
    ) -> Optional[float]:
        """Convert time to pixel X position where the line is actually drawn.

        Lines at axes edges are drawn with an offset to remain fully visible.

        Args:
            time_seconds: Time position in seconds
            line_offset: Offset in spec_frames used when drawing at edges

        Returns:
            Pixel X position of the drawn line, or None if outside visible range.
        """
        base_pixel = self._time_to_pixel_x(time_seconds)
        if base_pixel is None:
            return None

        # Calculate pixel offset from spec_frames offset
        _, ax_width = self.widget.axes_pixel_bounds
        pixel_offset = line_offset * (ax_width / self.widget.spec_frames)

        # Apply offset only at absolute edges (time 0.0 or recording end)
        at_absolute_start = time_seconds <= 0
        at_absolute_end = time_seconds >= self._recording_duration
        if at_absolute_start:
            return base_pixel + pixel_offset
        elif at_absolute_end:
            return base_pixel - pixel_offset
        return base_pixel

    def _pixel_to_time(self, rel_x: float) -> Optional[float]:
        """Convert relative X position (0-1) to time in seconds.

        Args:
            rel_x: Relative position in axes (0-1)

        Returns:
            Time in seconds, or None if no recording loaded
        """
        if self._recording_duration <= 0:
            return None

        view_start, view_end = self.widget.visible_time_range
        return view_start + rel_x * (view_end - view_start)

    # --- Selection/marker manipulation ---

    def set_marker(self, time_seconds: float) -> None:
        """Set marker at the specified time position.

        Args:
            time_seconds: Position in seconds
        """
        ctx = self._view_context
        if not ctx.has_recording:
            return

        # Clamp to recording bounds
        time_seconds = max(0.0, min(time_seconds, ctx.recording_duration))

        self._selection_state.set_marker(time_seconds)
        self._visualizer.update_marker(time_seconds, ctx)
        self._visualizer._hide_selection()
        self.widget.draw_idle()

    def set_selection(self, start_time: float, end_time: float) -> None:
        """Set selection range.

        Args:
            start_time: Selection start in seconds
            end_time: Selection end in seconds
        """
        ctx = self._view_context
        if not ctx.has_recording:
            return

        # Clamp to recording bounds
        start_time = max(0.0, min(start_time, ctx.recording_duration))
        end_time = max(0.0, min(end_time, ctx.recording_duration))

        self._selection_state.set_selection(start_time, end_time)
        self._visualizer.update_selection(
            self._selection_state.selection_start,
            self._selection_state.selection_end,
            ctx,
        )
        # Hide marker when setting selection
        if self._visualizer._marker_line:
            self._visualizer._marker_line.set_visible(False)
        self.widget.draw_idle()

    def clear_selection(self) -> None:
        """Clear marker and selection."""
        self._selection_state.clear_all()
        self._visualizer.clear()
        self.widget.draw_idle()

    # --- Cursor management ---

    def _set_cursor_resize(self) -> None:
        """Set cursor to horizontal resize cursor."""
        self.widget.canvas_widget.config(cursor="sb_h_double_arrow")

    def _set_cursor_default(self) -> None:
        """Reset cursor to default."""
        self.widget.canvas_widget.config(cursor="")
