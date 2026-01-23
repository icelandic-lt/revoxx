"""Selection visualization for spectrogram display.

This module provides visual elements for position markers and
range selections in the spectrogram view.
"""

from typing import Optional, TYPE_CHECKING
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from ....constants import UIConstants

if TYPE_CHECKING:
    from ..selection_state import SelectionState
    from ..view_context import ViewContext


def _format_time(seconds: float) -> str:
    """Format time in seconds to a readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1.234s" or "1:23.456")
    """
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


class SelectionVisualizer:
    """Manages visual display of markers and selections in the spectrogram.

    This visualizer handles:
    - Position marker: A cyan vertical line indicating playback start position
    - Selection range: A semi-transparent highlighted area with white borders
    """

    def __init__(self, ax: Axes):
        """Initialize selection visualizer.

        Args:
            ax: Matplotlib axes for drawing visual elements
        """
        self.ax = ax

        # Marker elements
        self._marker_line: Optional[Line2D] = None
        self._marker_time_text: Optional[Text] = None

        # Selection elements
        self._selection_patch: Optional[Rectangle] = None
        self._selection_start_line: Optional[Line2D] = None
        self._selection_end_line: Optional[Line2D] = None
        self._selection_duration_text: Optional[Text] = None
        self._selection_start_text: Optional[Text] = None
        self._selection_end_text: Optional[Text] = None

    def _ensure_marker_line(self) -> Line2D:
        """Create marker line and time text if they don't exist."""
        if self._marker_line is None:
            self._marker_line = self.ax.axvline(
                x=0,
                color=UIConstants.COLOR_POSITION_MARKER,
                linewidth=UIConstants.POSITION_MARKER_WIDTH,
                visible=False,
            )
        if self._marker_time_text is None:
            self._marker_time_text = self.ax.text(
                0,
                0,
                "",
                color=UIConstants.COLOR_POSITION_MARKER,
                fontsize=8,
                ha="center",
                va="top",
                visible=False,
            )
        return self._marker_line

    def _ensure_selection_elements(self, n_mels: int) -> None:
        """Create selection visual elements if they don't exist.

        Args:
            n_mels: Number of mel bins (height of the spectrogram)
        """
        if self._selection_patch is None:
            self._selection_patch = Rectangle(
                (0, 0),
                0,
                n_mels,
                facecolor=UIConstants.COLOR_SELECTION_FILL,
                edgecolor="none",
                visible=False,
            )
            self.ax.add_patch(self._selection_patch)

        if self._selection_start_line is None:
            self._selection_start_line = self.ax.axvline(
                x=0,
                color=UIConstants.COLOR_SELECTION_BORDER,
                linewidth=UIConstants.SELECTION_BORDER_WIDTH,
                visible=False,
            )

        if self._selection_end_line is None:
            # Use plot instead of axvline to allow shorter line (leave room for text)
            (self._selection_end_line,) = self.ax.plot(
                [0, 0],
                [0, n_mels - 8],
                color=UIConstants.COLOR_SELECTION_BORDER,
                linewidth=UIConstants.SELECTION_BORDER_WIDTH,
                visible=False,
            )

        if self._selection_duration_text is None:
            self._selection_duration_text = self.ax.text(
                0,
                n_mels - 10,
                "",
                color=UIConstants.COLOR_SELECTION_BORDER,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="top",
                visible=False,
            )

        if self._selection_start_text is None:
            self._selection_start_text = self.ax.text(
                0,
                0,
                "",
                color=UIConstants.COLOR_SELECTION_BORDER,
                fontsize=8,
                ha="center",
                va="top",
                visible=False,
            )

        if self._selection_end_text is None:
            self._selection_end_text = self.ax.text(
                0,
                n_mels - 2,
                "",
                color=UIConstants.COLOR_SELECTION_BORDER,
                fontsize=8,
                ha="center",
                va="top",
                visible=False,
            )

    def _time_to_display_x(
        self, time_seconds: float, ctx: "ViewContext"
    ) -> Optional[float]:
        """Convert time in seconds to display X coordinate.

        Args:
            time_seconds: Time position in seconds
            ctx: View context with display parameters

        Returns:
            X coordinate in display space, or None if outside visible range
        """
        visible_seconds = ctx.visible_seconds
        view_start = ctx.view_offset
        view_end = view_start + visible_seconds

        if time_seconds < view_start or time_seconds > view_end:
            return None

        relative_pos = (time_seconds - view_start) / visible_seconds
        return relative_pos * (ctx.spec_frames - 1)

    def update_marker(
        self, time_seconds: Optional[float], ctx: "ViewContext"
    ) -> None:
        """Update marker position display.

        Args:
            time_seconds: Marker position in seconds, or None to hide
            ctx: View context with display parameters
        """
        line = self._ensure_marker_line()

        if time_seconds is None or not ctx.has_recording:
            line.set_visible(False)
            if self._marker_time_text:
                self._marker_time_text.set_visible(False)
            return

        x_pos = self._time_to_display_x(time_seconds, ctx)

        if x_pos is None:
            line.set_visible(False)
            if self._marker_time_text:
                self._marker_time_text.set_visible(False)
            return

        line.set_xdata([x_pos])
        line.set_visible(True)

        # Update time text below the marker
        if self._marker_time_text:
            self._marker_time_text.set_position((x_pos, -2))
            self._marker_time_text.set_text(_format_time(time_seconds))
            self._marker_time_text.set_visible(True)

    def update_selection(
        self,
        start_seconds: Optional[float],
        end_seconds: Optional[float],
        ctx: "ViewContext",
    ) -> None:
        """Update selection range display.

        Args:
            start_seconds: Selection start in seconds, or None to hide
            end_seconds: Selection end in seconds, or None to hide
            ctx: View context with display parameters
        """
        self._ensure_selection_elements(ctx.n_mels)

        if start_seconds is None or end_seconds is None or not ctx.has_recording:
            self._hide_selection()
            return

        visible_start = ctx.view_offset
        visible_end = visible_start + ctx.visible_seconds

        # Check if selection is completely outside visible range
        if end_seconds < visible_start or start_seconds > visible_end:
            self._hide_selection()
            return

        # Convert times to display coordinates
        x_start = self._time_to_display_x(start_seconds, ctx)
        x_end = self._time_to_display_x(end_seconds, ctx)

        # If marker is outside view, clamp to edge
        x_start = x_start if x_start is not None else 0
        x_end = x_end if x_end is not None else ctx.spec_frames - 1

        x_start = max(0, min(x_start, ctx.spec_frames - 1))
        x_end = max(0, min(x_end, ctx.spec_frames - 1))

        # Show selection fill across visible area
        self._selection_patch.set_x(x_start)
        self._selection_patch.set_width(x_end - x_start)
        self._selection_patch.set_height(ctx.n_mels)
        self._selection_patch.set_visible(True)

        # Only show border lines if they are within visible range
        self._selection_start_line.set_xdata([x_start])
        start_visible = visible_start <= start_seconds <= visible_end
        self._selection_start_line.set_visible(start_visible)

        end_visible = visible_start <= end_seconds <= visible_end
        self._selection_end_line.set_data([x_end, x_end], [0, ctx.n_mels - 8])
        self._selection_end_line.set_visible(end_visible)

        # Show duration text in horizontal center, vertically near top
        if self._selection_duration_text:
            duration = end_seconds - start_seconds
            x_center = (x_start + x_end) / 2
            self._selection_duration_text.set_position((x_center, ctx.n_mels - 10))
            self._selection_duration_text.set_text(_format_time(duration))
            self._selection_duration_text.set_visible(True)

        # Show time labels at selection boundaries (start at bottom, end at top)
        if self._selection_start_text:
            self._selection_start_text.set_position((x_start, -2))
            self._selection_start_text.set_text(_format_time(start_seconds))
            self._selection_start_text.set_visible(start_visible)

        if self._selection_end_text:
            self._selection_end_text.set_position((x_end, ctx.n_mels - 2))
            self._selection_end_text.set_text(_format_time(end_seconds))
            self._selection_end_text.set_visible(end_visible)

    def _hide_selection(self) -> None:
        """Hide all selection visual elements."""
        if self._selection_patch:
            self._selection_patch.set_visible(False)
        if self._selection_start_line:
            self._selection_start_line.set_visible(False)
        if self._selection_end_line:
            self._selection_end_line.set_visible(False)
        if self._selection_duration_text:
            self._selection_duration_text.set_visible(False)
        if self._selection_start_text:
            self._selection_start_text.set_visible(False)
        if self._selection_end_text:
            self._selection_end_text.set_visible(False)

    def update_for_zoom(
        self, selection_state: "SelectionState", ctx: "ViewContext"
    ) -> None:
        """Update all visual elements after zoom change.

        Args:
            selection_state: Current selection state
            ctx: View context with display parameters
        """
        if selection_state.has_marker:
            self.update_marker(selection_state.marker_position, ctx)
        else:
            if self._marker_line:
                self._marker_line.set_visible(False)
            if self._marker_time_text:
                self._marker_time_text.set_visible(False)

        if selection_state.has_selection:
            self.update_selection(
                selection_state.selection_start,
                selection_state.selection_end,
                ctx,
            )
        else:
            self._hide_selection()

    def hide(self) -> None:
        """Hide all visual elements without clearing state."""
        if self._marker_line:
            self._marker_line.set_visible(False)
        if self._marker_time_text:
            self._marker_time_text.set_visible(False)
        self._hide_selection()

    def clear(self) -> None:
        """Clear all visual elements."""
        if self._marker_line:
            self._marker_line.set_visible(False)
        if self._marker_time_text:
            self._marker_time_text.set_visible(False)
        self._hide_selection()
