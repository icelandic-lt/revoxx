"""LED-style vertical level meter widget that reads from SharedState.

This module provides a LED-style level meter similar to
those found in audio production software. It displays both RMS and peak
levels with configurable target ranges and visual feedback.

Architecture Overview:
- LEDs are drawn from top (0 dB) to bottom (-60 dB)
- Non-linear scale with more resolution in critical -24 to 0 dB range
- Dynamic height calculation ensures all elements fit within available space
- Color-coded LEDs: red (danger), yellow (warning), green (optimal), blue (low)

Key Design Decisions:
1. Fixed canvas height: Works within allocated space rather than requesting more
2. Top-aligned LEDs: Ensures -60 dB marker is always visible at bottom
3. Adaptive sizing: LED height and spacing adjust based on available space
4. Padding constants: All spacing uses defined constants for consistency
"""

import tkinter as tk
from typing import Optional
import time

from .config import LevelMeterConfig, RecordingStandard, RECORDING_STANDARDS
from ...audio.shared_state import SharedState, SETTINGS_STATUS_VALID
from ..themes import theme_manager


class LEDLevelMeter(tk.Frame):
    """Vertical LED-style audio level meter widget that reads from SharedState.

    A audio level meter that displays RMS and peak levels using
    colored LED segments. Implements a non-linear dB scale for better resolution
    in the critical -24 to 0 dB range.

    Visual Layout (top to bottom):
    - Peak/RMS value labels
    - LED meter (0 dB at top, -60 dB at bottom)
    - Scale markings on the right

    Key Features:
    - Non-linear scale with pivot at -24 dB
    - Color-coded segments (red/yellow/green/blue)
    - Peak hold indicator
    - Target range visualization
    - Adaptive height calculation to prevent clipping

    Implementation Notes:
    - LEDs are indexed 0 (bottom) to LED_COUNT-1 (top) internally
    - Visual display is inverted: top shows 0 dB, bottom shows -60 dB
    - Height calculation ensures -60 dB marker is always visible
    - Uses top-aligned positioning rather than centering
    """

    # Visual constants
    METER_WIDTH = 80
    METER_MIN_HEIGHT = 220
    LED_COUNT = 30
    LED_MIN_HEIGHT = 4
    LED_MAX_HEIGHT = 10
    LED_MIN_SPACING = 2
    LED_MAX_SPACING = 4
    LED_X_INSET = 20
    SCALE_WIDTH = 50
    MARGIN = 8
    LED_SPACING_SCALE = 9  # Heuristic scale factor for spacing calculation

    # Padding constants
    LABEL_FRAME_PADY = (2, 5)  # Top and bottom padding for label frame
    CANVAS_PADX = (5, 0)  # Left and right padding for LED canvas
    CANVAS_PADY = (0, 10)  # Top and bottom padding for canvases
    SCALE_CANVAS_PADX = (2, 5)  # Left and right padding for scale canvas
    CONTAINER_PAD_X = 2  # Horizontal padding for containers
    EXTRA_HEIGHT_MARGIN = 10  # Extra margin for height calculation

    # Non-linear scale to give more resolution in the upper, critical range
    SCALE_PIVOT_DB = -30.0
    SCALE_BOTTOM_FRACTION = (
        0.4  # portion of the meter used for [-60..pivot], top gets the rest
    )

    # Colors will be loaded from theme
    @property
    def COLOR_BACKGROUND(self):
        return theme_manager.colors.LEVEL_COLOR_BACKGROUND

    @property
    def COLOR_LED_OFF(self):
        return theme_manager.colors.LEVEL_COLOR_LED_OFF

    @property
    def COLOR_OPTIMAL(self):
        return theme_manager.colors.LEVEL_COLOR_OPTIMAL

    @property
    def COLOR_WARNING(self):
        return theme_manager.colors.LEVEL_COLOR_WARNING

    @property
    def COLOR_DANGER(self):
        return theme_manager.colors.LEVEL_COLOR_DANGER

    @property
    def COLOR_LOW(self):
        return theme_manager.colors.LEVEL_COLOR_LOW

    @property
    def COLOR_RMS(self):
        return theme_manager.colors.LEVEL_COLOR_RMS

    @property
    def COLOR_PEAK(self):
        return theme_manager.colors.LEVEL_COLOR_PEAK

    @property
    def COLOR_TEXT(self):
        return theme_manager.colors.LEVEL_COLOR_TEXT

    @property
    def COLOR_GRID(self):
        return theme_manager.colors.LEVEL_COLOR_GRID

    @property
    def DIM_FACTOR(self):
        return theme_manager.colors.LEVEL_DIM_FACTOR

    def __init__(
        self,
        parent: tk.Widget,
        shared_state: SharedState,
        config: Optional[LevelMeterConfig] = None,
    ):
        """Initialize LED level meter widget.

        Args:
            parent: Parent tkinter widget
            shared_state: SharedState instance for reading level data
            config: Level meter configuration
        """
        # Use theme background color
        super().__init__(parent, bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND)

        self.shared_state = shared_state
        self.config = config or LevelMeterConfig()

        # Level tracking
        self.current_rms = -60.0
        self.current_peak = -60.0
        self.peak_hold_value = -60.0
        self._peak_hold_last_update = time.monotonic()
        self.PEAK_HOLD_MS = 1500  # hold time in milliseconds
        self.PEAK_DECAY_DB_PER_SEC = 10.0
        self.last_frame_count = 0

        # LED elements
        self.leds = []
        self.led_states = [False] * self.LED_COUNT

        # Geometry cache for fast updates
        self._geom_height = self.METER_MIN_HEIGHT
        self._geom_led_height = 8
        self._geom_spacing = 2
        self._geom_start_y = 0

        # Peak-hold line
        self._peak_hold_line_id: Optional[int] = None

        # Control flag to stop scheduling when widget is destroyed
        self._running: bool = True
        self.bind("<Destroy>", lambda e: setattr(self, "_running", False))

        self._create_ui()
        self._schedule_update()

    def refresh(self) -> None:
        """Force a redraw of geometry and display (useful after (re)show)."""
        try:
            self._rebuild_geometry()
            self._update_display()
        except Exception:
            pass

    def get_minimum_height(self) -> int:
        """Calculate the minimum height needed for proper display.

        Returns:
            Minimum height in pixels including all padding
        """
        # Estimate label frame height (typically around 40-50 pixels for 2 rows)
        estimated_label_height = 45

        # Extract padding values
        label_frame_pady = sum(self.LABEL_FRAME_PADY)
        canvas_pady = sum(self.CANVAS_PADY)

        # Minimum height for reasonable LED display
        min_canvas_height = self.METER_MIN_HEIGHT

        # Total minimum height
        return (
            estimated_label_height
            + label_frame_pady
            + min_canvas_height
            + canvas_pady
            + self.EXTRA_HEIGHT_MARGIN
        )

    def _create_ui(self) -> None:
        """Create the LED level meter UI components."""
        self._create_top_labels()
        self._create_meter_section()
        self._create_bottom_label()
        self._initialize_display()

    def _create_top_labels(self) -> None:
        """Create the top label section with PEAK and RMS displays."""
        # Main label frame
        self.label_frame = tk.Frame(
            self, bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND
        )
        self.label_frame.pack(
            fill=tk.X, padx=self.CONTAINER_PAD_X, pady=self.LABEL_FRAME_PADY
        )

        # Container aligned with LED meter
        self.label_container = tk.Frame(
            self.label_frame, bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND
        )
        self.label_container.pack(
            anchor="w", padx=(self.CANVAS_PADX[0] + self.LED_X_INSET, 0)
        )

        # Create PEAK and RMS rows
        self._create_peak_row()
        self._create_rms_row()

    def _create_peak_row(self) -> None:
        """Create the PEAK label row with value display."""
        peak_row = tk.Frame(
            self.label_container, bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND
        )
        peak_row.pack(fill=tk.X)

        # PEAK label
        self._create_styled_label(peak_row, "PEAK", self.COLOR_PEAK).pack(side=tk.LEFT)

        # Spacer for alignment
        self._create_spacer_label(peak_row, 4).pack(side=tk.LEFT)

        # PEAK value
        self.peak_value_label = self._create_styled_label(
            peak_row, "-- dB", self.COLOR_TEXT
        )
        self.peak_value_label.pack(side=tk.LEFT)

    def _create_rms_row(self) -> None:
        """Create the RMS label row with value display."""
        rms_row = tk.Frame(
            self.label_container, bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND
        )
        rms_row.pack(fill=tk.X)

        # RMS label
        self._create_styled_label(rms_row, "RMS", self.COLOR_RMS).pack(side=tk.LEFT)

        # Spacer for alignment (RMS is shorter than PEAK)
        self._create_spacer_label(rms_row, 5).pack(side=tk.LEFT)

        # RMS value
        self.rms_value_label = self._create_styled_label(
            rms_row, "-- dB", self.COLOR_TEXT
        )
        self.rms_value_label.pack(side=tk.LEFT)

    def _create_meter_section(self) -> None:
        """Create the main meter section with LED and scale canvases."""
        # Container for meter and scale
        self.meter_container = tk.Frame(
            self, bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND
        )
        self.meter_container.pack(fill=tk.BOTH, expand=True)

        # Create canvases
        self._create_led_canvas()
        self._create_scale_canvas()

    def _create_led_canvas(self) -> None:
        """Create the LED meter canvas."""
        self.canvas = tk.Canvas(
            self.meter_container,
            width=self.METER_WIDTH,
            height=self.METER_MIN_HEIGHT,
            bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND,
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=self.CANVAS_PADX, pady=self.CANVAS_PADY)

    def _create_scale_canvas(self) -> None:
        """Create the scale canvas for dB markings."""
        self.scale_canvas = tk.Canvas(
            self.meter_container,
            width=self.SCALE_WIDTH,
            height=self.METER_MIN_HEIGHT,
            bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND,
            highlightthickness=0,
        )
        self.scale_canvas.pack(
            side=tk.LEFT, padx=self.SCALE_CANVAS_PADX, pady=self.CANVAS_PADY
        )

    def _create_bottom_label(self) -> None:
        """Create the bottom level readout label."""
        self.level_label = tk.Label(
            self,
            text="-- dB",
            fg=self.COLOR_TEXT,
            bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND,
            font=("TkDefaultFont", 12),
        )
        self.level_label.pack(pady=(5, 2))

    def _initialize_display(self) -> None:
        """Initialize the display and bind resize events."""
        self._rebuild_geometry()
        self.bind("<Configure>", self._on_resize)

    def _create_styled_label(
        self, parent: tk.Widget, text: str, fg_color: str, font_size: int = 9
    ) -> tk.Label:
        """Create a label with consistent styling.

        Args:
            parent: Parent widget
            text: Label text
            fg_color: Foreground color
            font_size: Font size (default 9)

        Returns:
            Configured Label widget
        """
        return tk.Label(
            parent,
            text=text,
            fg=fg_color,
            bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND,
            font=("TkDefaultFont", font_size),
        )

    def _create_spacer_label(self, parent: tk.Widget, spaces: int) -> tk.Label:
        """Create a spacer label for alignment.

        Args:
            parent: Parent widget
            spaces: Number of spaces

        Returns:
            Spacer Label widget
        """
        return tk.Label(
            parent,
            text=" " * spaces,
            bg=theme_manager.colors.LEVEL_COLOR_BACKGROUND,
            font=("TkDefaultFont", 9),
        )

    def _rebuild_geometry(self) -> None:
        """Rebuild LED layout and scale based on current height.

        This is the core method that orchestrates the recalculation and redrawing
        of all meter components when the widget is resized or refreshed.
        """
        self.update_idletasks()  # Ensure geometry is updated

        height = self._calculate_canvas_height()
        self._update_all_backgrounds(height)
        led_height, spacing, start_y = self._calculate_led_layout(height)
        self._cache_geometry(height, led_height, spacing, start_y)
        self._redraw_all_elements(led_height, spacing, start_y)

    def _calculate_canvas_height(self) -> int:
        """Calculate the available height for the canvas.

        Returns:
            Available height in pixels for the canvas
        """
        total_widget_height = self.winfo_height()
        label_frame_height = (
            self.label_frame.winfo_reqheight() if self.label_frame.winfo_exists() else 0
        )

        # Extract and sum all padding values
        total_padding = (
            self.LABEL_FRAME_PADY[0]
            + self.LABEL_FRAME_PADY[1]
            + self.CANVAS_PADY[0]
            + self.CANVAS_PADY[1]
            + self.EXTRA_HEIGHT_MARGIN
        )

        return max(
            self.METER_MIN_HEIGHT,
            total_widget_height - label_frame_height - total_padding,
        )

    def _update_all_backgrounds(self, height: int) -> None:
        """Update all widget backgrounds with current theme color.

        Args:
            height: Canvas height in pixels
        """
        bg_color = theme_manager.colors.LEVEL_COLOR_BACKGROUND

        # Update canvas heights and backgrounds
        self.canvas.config(height=height, bg=bg_color)
        self.scale_canvas.config(height=height, bg=bg_color)

        # Update container backgrounds
        self.configure(bg=bg_color)
        self.meter_container.configure(bg=bg_color)
        self.label_frame.configure(bg=bg_color)

        # Update nested label backgrounds
        self._update_nested_backgrounds(self.label_frame, bg_color)

        # Schedule delayed refresh to ensure colors stick
        if hasattr(self, "_refresh_timer"):
            self.after_cancel(self._refresh_timer)
        self._refresh_timer = self.after(20, self._refresh_all_colors)

    def _update_nested_backgrounds(self, parent: tk.Widget, bg_color: str) -> None:
        """Recursively update backgrounds of nested widgets.

        Args:
            parent: Parent widget to traverse
            bg_color: Background color to apply
        """
        for widget in parent.winfo_children():
            widget.configure(bg=bg_color)
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    child.configure(bg=bg_color)
                    if isinstance(child, tk.Frame):
                        for grandchild in child.winfo_children():
                            grandchild.configure(bg=bg_color)

    def _calculate_led_layout(self, height: int) -> tuple[int, int, int]:
        """Calculate optimal LED dimensions and positioning.

        This method determines the best LED height and spacing to fit all LEDs
        within the available space without clipping.

        Args:
            height: Available canvas height in pixels

        Returns:
            Tuple of (led_height, spacing, start_y)
        """
        # Fixed margins for top/bottom
        TOP_MARGIN = 5
        BOTTOM_MARGIN = 10

        available = height - TOP_MARGIN - BOTTOM_MARGIN

        # Ensure minimum usable space
        if available < 50:
            available = 50

        # Calculate LED dimensions
        led_height, spacing = self._optimize_led_dimensions(available)

        return led_height, spacing, TOP_MARGIN

    def _optimize_led_dimensions(self, available_height: int) -> tuple[int, int]:
        """Calculate optimal LED height and spacing for available space.

        Args:
            available_height: Available vertical space in pixels

        Returns:
            Tuple of (led_height, spacing) in pixels
        """
        spacing = self.LED_MIN_SPACING

        # Calculate LED height to fit exactly in available space
        led_height = (
            available_height - (self.LED_COUNT - 1) * spacing
        ) / self.LED_COUNT
        led_height = int(led_height)
        led_height = max(self.LED_MIN_HEIGHT, min(self.LED_MAX_HEIGHT, led_height))

        # Verify and adjust if needed
        total_height = self.LED_COUNT * led_height + (self.LED_COUNT - 1) * spacing

        while total_height > available_height and led_height > self.LED_MIN_HEIGHT:
            led_height -= 1
            total_height = self.LED_COUNT * led_height + (self.LED_COUNT - 1) * spacing

        return led_height, spacing

    def _cache_geometry(
        self, height: int, led_height: int, spacing: int, start_y: int
    ) -> None:
        """Cache calculated geometry for efficient updates.

        Args:
            height: Canvas height
            led_height: Height of each LED
            spacing: Spacing between LEDs
            start_y: Starting Y position
        """
        self._geom_height = height
        self._geom_led_height = led_height
        self._geom_spacing = spacing
        self._geom_start_y = start_y

    def _redraw_all_elements(self, led_height: int, spacing: int, start_y: int) -> None:
        """Redraw all visual elements on the canvases.

        Args:
            led_height: Height of each LED
            spacing: Spacing between LEDs
            start_y: Starting Y position for LED layout
        """
        # Clear canvases
        self.canvas.delete("all")
        self.scale_canvas.delete("all")

        # Draw LEDs
        self._draw_leds(led_height, spacing, start_y)

        # Reset peak hold line
        self._reset_peak_hold_line()

        # Draw scale
        self._draw_scale_dynamic(self._geom_height, led_height, spacing, start_y)

    def _draw_leds(self, led_height: int, spacing: int, start_y: int) -> None:
        """Draw all LED rectangles on the canvas.

        Args:
            led_height: Height of each LED
            spacing: Spacing between LEDs
            start_y: Starting Y position
        """
        self.leds.clear()

        for i in range(self.LED_COUNT):
            # Calculate Y position: invert index so LED 0 is at bottom visually
            y = start_y + (self.LED_COUNT - 1 - i) * (led_height + spacing)
            db_value = self._led_index_to_db(i)
            color = self._get_led_color(db_value)

            led = self.canvas.create_rectangle(
                self.LED_X_INSET,
                y,
                self.METER_WIDTH - self.LED_X_INSET,
                y + led_height,
                fill=theme_manager.colors.LEVEL_COLOR_LED_OFF,
                outline=theme_manager.colors.LEVEL_COLOR_LED_OFF,
                width=1,
            )
            self.leds.append((led, color))

    def _reset_peak_hold_line(self) -> None:
        """Remove and reset the peak hold line."""
        if self._peak_hold_line_id is not None:
            try:
                self.canvas.delete(self._peak_hold_line_id)
            except Exception:
                pass
        self._peak_hold_line_id = None

    def _led_index_to_db(self, index: int) -> float:
        """Convert LED index to dB value using non-linear scale."""
        if index <= 0:
            return -60.0
        if index >= self.LED_COUNT - 1:
            return 0.0
        position = index / (self.LED_COUNT - 1)  # 0..1 from bottom to top
        bottom_frac = self.SCALE_BOTTOM_FRACTION
        top_frac = 1.0 - bottom_frac
        if position <= bottom_frac:
            # Map [0..bottom_frac] → [-60..pivot]
            rel = position / bottom_frac if bottom_frac > 0 else 0.0
            return -60.0 + rel * (self.SCALE_PIVOT_DB - (-60.0))
        else:
            # Map (bottom_frac..1] → [pivot..0]
            rel = (position - bottom_frac) / top_frac if top_frac > 0 else 0.0
            return self.SCALE_PIVOT_DB + rel * (0.0 - self.SCALE_PIVOT_DB)

    def _db_to_led_count(self, db: float) -> int:
        """Convert dB value to number of LEDs to light using non-linear scale."""
        if db <= -60.0:
            return 0
        if db >= 0.0:
            return self.LED_COUNT

        bottom_frac = self.SCALE_BOTTOM_FRACTION
        top_frac = 1.0 - bottom_frac

        if db <= self.SCALE_PIVOT_DB:
            # Map [-60..pivot] → [0..bottom_frac]
            rel = (
                (db - (-60.0)) / (self.SCALE_PIVOT_DB - (-60.0))
                if self.SCALE_PIVOT_DB > -60.0
                else 0.0
            )
            position = rel * bottom_frac
        else:
            # Map (pivot..0] → (bottom_frac..1]
            rel = (
                (db - self.SCALE_PIVOT_DB) / (0.0 - self.SCALE_PIVOT_DB)
                if self.SCALE_PIVOT_DB < 0.0
                else 0.0
            )
            position = bottom_frac + rel * top_frac

        # position 0..1 → LED count 0..LED_COUNT
        return int(round(position * self.LED_COUNT))

    def _get_led_color(self, db: float) -> str:
        """Get LED color based on dB value."""
        # Cache colors from theme manager to avoid property lookup issues
        colors = theme_manager.colors

        if db >= self.config.danger_level:
            color = colors.LEVEL_COLOR_DANGER
        elif db >= self.config.warning_level:
            color = colors.LEVEL_COLOR_WARNING
        elif db >= self.config.target_min:
            color = colors.LEVEL_COLOR_OPTIMAL
        else:
            color = colors.LEVEL_COLOR_LOW

        return color

    def _dim_color(self, color: str, factor: float = 0.25) -> str:
        """Return a dimmed version of a hex color by blending towards background.

        Args:
            color: Hex color string like '#RRGGBB'
            factor: 0..1 intensity (lower = dimmer)

        Returns:
            Hex color string
        """

        def hex_to_rgb(h: str) -> tuple[int, int, int]:
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

        def rgb_to_hex(r: int, g: int, b: int) -> str:
            return f"#{r:02x}{g:02x}{b:02x}"

        src_r, src_g, src_b = hex_to_rgb(color)
        bg_r, bg_g, bg_b = hex_to_rgb(self.COLOR_BACKGROUND)
        r = int(bg_r + (src_r - bg_r) * max(0.0, min(1.0, factor)))
        g = int(bg_g + (src_g - bg_g) * max(0.0, min(1.0, factor)))
        b = int(bg_b + (src_b - bg_b) * max(0.0, min(1.0, factor)))
        return rgb_to_hex(r, g, b)

    def _draw_scale_dynamic(
        self, height: int, led_height: int, spacing: int, start_y: int
    ) -> None:
        """Draw the dB scale aligned to LED centers with readable fonts."""
        # Adjusted ticks: 0, -6, -12, ... for clearer top range
        db_marks = [0, -6, -12, -18, -24, -30, -40, -50, -60]

        for db in db_marks:
            # place tick at the y of the LED that contains this value
            led_index = min(max(self._db_to_led_count(db) - 1, 0), self.LED_COUNT - 1)
            y = (
                start_y
                + (self.LED_COUNT - 1 - led_index) * (led_height + spacing)
                + led_height / 2
            )
            self.scale_canvas.create_line(0, y, 12, y, fill=self.COLOR_GRID, width=1)
            self.scale_canvas.create_text(
                16,
                y,
                text=str(db),
                fill=self.COLOR_TEXT,
                font=("TkDefaultFont", 10),
                anchor="w",
            )

        # Draw target range lines
        self._draw_target_range_lines(led_height, spacing, start_y)

    def _draw_target_range_lines(
        self, led_height: int, spacing: int, start_y: int
    ) -> None:
        """Draw horizontal lines indicating the target range.

        Args:
            led_height: Height of each LED
            spacing: Spacing between LEDs
            start_y: Starting Y position for LED layout
        """
        tmin_idx = self._db_to_led_count(self.config.target_min) - 1
        tmax_idx = self._db_to_led_count(self.config.target_max) - 1

        # Draw target range lines
        for target_idx in [tmin_idx, tmax_idx]:
            if 0 <= target_idx < self.LED_COUNT:
                y = (
                    start_y
                    + (self.LED_COUNT - 1 - target_idx) * (led_height + spacing)
                    + led_height / 2
                )
                # Draw through LED meter only
                self.canvas.create_line(
                    self.LED_X_INSET,
                    y,
                    self.METER_WIDTH - self.LED_X_INSET,
                    y,
                    fill=self.COLOR_OPTIMAL,
                    width=2,
                    dash=(3, 2),
                )

    def _schedule_update(self) -> None:
        """Schedule periodic updates from shared state."""
        try:
            self._update_from_shared_state()
        finally:
            # Schedule next update (30 Hz) only if widget is still alive and running
            if self._running and self.winfo_exists():
                self.after(33, self._schedule_update)

    def _update_from_shared_state(self) -> None:
        """Update level meter from shared state data."""
        if not self.shared_state or getattr(self.shared_state, "shm", None) is None:
            return

        # Get level meter state
        try:
            level_state = self.shared_state.get_level_meter_state()
        except Exception:
            # Shared memory likely already torn down during shutdown
            return

        # Check if valid data
        if level_state.get("status", 0) != SETTINGS_STATUS_VALID:
            return

        # Check if data has been updated
        frame_count = level_state.get("frame_count", 0)
        if frame_count == self.last_frame_count:
            return

        self.last_frame_count = frame_count

        # Update levels
        self.current_rms = level_state.get("rms_db", -60.0)
        instant_peak = level_state.get("peak_db", -60.0)
        self.current_peak = instant_peak

        self._calculate_peak_hold(instant_peak)

        # Update display
        self._update_display()

    def _calculate_peak_hold(self, instant_peak: float) -> None:
        """Update peak-hold value with hold and decay behavior.

        Peak values are held for PEAK_HOLD_MS milliseconds, then decay at
        PEAK_DECAY_DB_PER_SEC rate. The peak value never drops below the
        current instant peak or RMS level.

        Args:
            instant_peak: Current instantaneous peak value in dB
        """
        now = time.monotonic()
        if instant_peak > self.peak_hold_value:
            self.peak_hold_value = instant_peak
            self._peak_hold_last_update = now
        else:
            elapsed = now - self._peak_hold_last_update
            hold_seconds = self.PEAK_HOLD_MS / 1000.0
            if elapsed > hold_seconds:
                decay = self.PEAK_DECAY_DB_PER_SEC * (elapsed - hold_seconds)
                new_value = self.peak_hold_value - decay
                floor_value = max(instant_peak, self.current_rms)
                self.peak_hold_value = max(-60.0, max(new_value, floor_value))
                # keep last update to avoid compounding
                self._peak_hold_last_update = now

    def _get_color_for_db_value(self, db: float) -> str:
        """Get color for a given dB value based on thresholds.

        Args:
            db: Decibel value

        Returns:
            Color string
        """
        if db >= self.config.danger_level:
            return self.COLOR_DANGER
        if db >= self.config.warning_level:
            return self.COLOR_WARNING
        if self.config.target_min <= db <= self.config.target_max:
            return self.COLOR_OPTIMAL
        return self.COLOR_TEXT

    def _update_led_display(self, rms_led_count: int) -> None:
        """Update the LED bar display.

        Args:
            rms_led_count: Number of LEDs to light for RMS level
        """
        for i in range(self.LED_COUNT):
            led, _ = self.leds[i]

            # Get current color based on LED position and current config
            db_value = self._led_index_to_db(i)
            color = self._get_led_color(db_value)

            # Light up LEDs up to RMS level; others show dimmed zone color
            if i < rms_led_count:
                self.canvas.itemconfig(led, fill=color)
            else:
                self.canvas.itemconfig(
                    led, fill=self._dim_color(color, self.DIM_FACTOR)
                )

    def _update_numeric_labels(self) -> tuple:
        """Update numeric readout labels.

        Returns:
            Tuple of (rms_text, peak_text)
        """
        # Format text values
        rms_text = f"{self.current_rms:.1f}" if self.current_rms > -60 else "--"
        peak_text = (
            f"{self.peak_hold_value:.1f}" if self.peak_hold_value > -60 else "--"
        )

        # Update labels
        self.rms_value_label.config(text=f"{rms_text} dB")
        self.peak_value_label.config(text=f"{peak_text} dB")
        self.level_label.config(text=f"{rms_text} dB")

        # Update colors
        if self.current_rms > -60:
            self.rms_value_label.config(
                fg=self._get_color_for_db_value(self.current_rms)
            )
            self.level_label.config(fg=self._get_color_for_db_value(self.current_rms))
        else:
            self.rms_value_label.config(fg=self.COLOR_TEXT)
            self.level_label.config(fg=self.COLOR_TEXT)

        if self.peak_hold_value > -60:
            self.peak_value_label.config(
                fg=self._get_color_for_db_value(self.peak_hold_value)
            )
        else:
            self.peak_value_label.config(fg=self.COLOR_TEXT)

        return rms_text, peak_text

    def _update_peak_hold_line(self) -> None:
        """Update or create the peak hold line on the meter canvas."""
        if self.peak_hold_value > -60:
            y = self._value_to_y(self.peak_hold_value)
        else:
            y = None

        if y is not None:
            x0, x1 = self.LED_X_INSET, self.METER_WIDTH - self.LED_X_INSET
            # Choose color from the same zone mapping as LEDs
            ph_color = self._get_led_color(self.peak_hold_value)

            if self._peak_hold_line_id is None:
                self._peak_hold_line_id = self.canvas.create_line(
                    x0, y, x1, y, fill=ph_color, width=2
                )
            else:
                self.canvas.coords(self._peak_hold_line_id, x0, y, x1, y)
                self.canvas.itemconfig(self._peak_hold_line_id, fill=ph_color)
        else:
            # Hide line if below -60 dB
            if self._peak_hold_line_id is not None:
                try:
                    self.canvas.delete(self._peak_hold_line_id)
                except Exception:
                    pass
                self._peak_hold_line_id = None

    def _update_display(self) -> None:
        """Update the LED display."""
        # Calculate how many LEDs to light for RMS
        rms_led_count = self._db_to_led_count(self.current_rms)

        # Calculate peak LED position (unused but kept for compatibility)
        self._db_to_led_count(self.current_peak) - 1

        # Update LED bar
        self._update_led_display(rms_led_count)

        # Update numeric labels
        self._update_numeric_labels()

        # Draw/update peak-hold line
        self._update_peak_hold_line()

    def set_standard(self, standard: RecordingStandard) -> None:
        """Set the recording standard preset.

        Args:
            standard: Recording standard to use
        """
        if standard in RECORDING_STANDARDS:
            self.config = RECORDING_STANDARDS[standard]
            # Redraw LEDs with new color zones
            self._rebuild_geometry()

    def set_config(self, config: LevelMeterConfig) -> None:
        """Set custom configuration.

        Args:
            config: Level meter configuration
        """
        self.config = config

        # Rebuild full geometry (LEDs + scale) to reflect new thresholds
        self._rebuild_geometry()

        # Force immediate display update
        self._update_display()

    def get_current_levels(self) -> tuple[float, float]:
        """Get current RMS and peak levels.

        Returns:
            Tuple of (rms_db, peak_db)
        """
        return self.current_rms, self.current_peak

    def is_in_optimal_range(self) -> bool:
        """Check if current RMS level is in optimal range.

        Returns:
            True if in optimal range
        """
        return self.config.target_min <= self.current_rms <= self.config.target_max

    def reset(self) -> None:
        """Reset the level meter."""
        self.current_rms = -60.0
        self.current_peak = -60.0
        self.peak_hold_value = -60.0
        self.last_frame_count = 0
        self._update_display()

    # --- Resize handling ---
    def _on_resize(self, event) -> None:
        """Handle resizing to prevent clipping and keep layout readable."""
        self._rebuild_geometry()

    def _refresh_all_colors(self) -> None:
        """Refresh all colors from theme to ensure they stick after resize."""
        bg_color = theme_manager.colors.LEVEL_COLOR_BACKGROUND

        # Refresh canvas backgrounds
        self.canvas.config(bg=bg_color)
        self.scale_canvas.config(bg=bg_color)

        # Refresh container backgrounds
        self.configure(bg=bg_color)
        self.meter_container.configure(bg=bg_color)
        self.label_frame.configure(bg=bg_color)

        # Refresh all label backgrounds - handle nested structure
        for widget in self.label_frame.winfo_children():
            widget.configure(bg=bg_color)
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    child.configure(bg=bg_color)
                    if isinstance(child, tk.Frame):
                        for grandchild in child.winfo_children():
                            grandchild.configure(bg=bg_color)

        # Refresh LED colors
        for i, (led, _) in enumerate(self.leds):
            db_value = self._led_index_to_db(i)
            color = self._get_led_color(db_value)
            # Update the stored color for this LED
            self.leds[i] = (led, color)

        # Force an immediate display update to apply colors
        self._update_display()

    # --- Helpers ---
    def _value_to_y(self, db_value: float) -> Optional[float]:
        """Map a dB value to Y coordinate (center of the corresponding LED)."""
        if db_value <= -60:
            index = 0
        elif db_value >= 0:
            index = self.LED_COUNT - 1
        else:
            index = self._db_to_led_count(db_value) - 1
        if not (0 <= index < self.LED_COUNT):
            return None
        y = (
            self._geom_start_y
            + (self.LED_COUNT - 1 - index)
            * (self._geom_led_height + self._geom_spacing)
            + self._geom_led_height / 2
        )
        return y
