"""Main mel spectrogram widget that coordinates all components."""

from typing import Optional, List
import numpy as np
import tkinter as tk
import queue

from matplotlib.image import AxesImage

from ...constants import AudioConstants
from ...constants import UIConstants
from ...utils.adaptive_frame_rate import get_adaptive_frame_rate, DEBUG_FPS
from ..themes import theme_manager
from ...audio.processors import ClippingDetector
from ...audio.processors import MelSpectrogramProcessor, MEL_CONFIG
from ...utils.config import AudioConfig, DisplayConfig

from .display_base import SpectrogramDisplayBase
from .recording_handler import RecordingHandler
from .playback_handler import PlaybackHandler
from .recording_display import RecordingDisplay
from .controllers import (
    ZoomController,
    PlaybackController,
    ClippingVisualizer,
    EdgeIndicator,
    SelectionVisualizer,
)
from .selection_state import SelectionState
from .selection_interaction import SelectionInteractionHandler
from .view_context import ViewContext, SavedViewState


class MelSpectrogramWidget(SpectrogramDisplayBase):
    """Real-time mel spectrogram display widget with recording and playback.

    This widget provides a complete mel spectrogram visualization system
    combining live recording, playback animation, and zoom functionality.

    Constants:
        ZOOM_LEVELS: Available zoom levels
        BASE_FREQ_RANGE: Original frequency range for mel scaling
        MIN_ADAPTIVE_MELS: Minimum number of mel bins
        BASE_MEL_BINS: Base number of mel bins for scaling
        ZOOM_INDICATOR_HIDE_DELAY_MS: Auto-hide delay for zoom indicator
        ZOOM_INDICATOR_FONTSIZE: Font size for zoom indicator
        FIGURE_PADDING: Padding for figure size calculation
    """

    # Constants for frequently used calculations
    BASE_FREQ_RANGE = 24000 - 50
    MIN_ADAPTIVE_MELS = 80
    BASE_MEL_BINS = 96
    ZOOM_INDICATOR_HIDE_DELAY_MS = 2000
    ZOOM_INDICATOR_FONTSIZE = 10
    FIGURE_PADDING = 20
    MAX_CHUNKS_PER_UPDATE = 10  # Maximum audio chunks to process per display update

    def __init__(
        self,
        parent: tk.Widget,
        audio_config: AudioConfig,
        display_config: DisplayConfig,
        manager_dict: dict = None,
        shared_audio_state=None,
    ):
        """Initialize the mel spectrogram widget.

        Args:
            parent: Parent tkinter widget
            audio_config: Audio configuration
            display_config: Display configuration
            manager_dict: Shared application state
        """
        super().__init__(parent, audio_config, display_config, manager_dict)
        self.shared_audio_state = shared_audio_state

        # Initialize mel processor
        self.mel_processor, self.adaptive_n_mels = MelSpectrogramProcessor.create_for(
            audio_config.sample_rate, display_config.fmin
        )

        # Initialize recording-specific parameters
        self._recording_n_mels = self.adaptive_n_mels
        self._recording_sample_rate = audio_config.sample_rate
        self._recording_fmax = audio_config.sample_rate / 2
        self.max_detected_freq = 0.0

        # Initialize processors
        self.clipping_detector = ClippingDetector(sample_rate=audio_config.sample_rate)

        # Initialize controllers
        self.zoom_controller = ZoomController()
        self.playback_controller = PlaybackController()

        # Initialize display
        self._init_display()

        # Initialize visualizers (need axes)
        self.clipping_visualizer = ClippingVisualizer(self.ax)
        self.selection_visualizer = SelectionVisualizer(self.ax)

        # Initialize handlers
        self.recording_handler = RecordingHandler(
            self.mel_processor,
            self.clipping_detector,
            self.clipping_visualizer,
            self.spec_frames,
            self.adaptive_n_mels,
            audio_config.sample_rate,
        )

        self.playback_handler = PlaybackHandler(
            self.parent,
            self.ax,
            self.playback_controller,
            self.zoom_controller,
            self.spec_frames,
            self.shared_audio_state,
        )

        self.recording_display = RecordingDisplay(
            self.clipping_detector,
            self.clipping_visualizer,
            self.zoom_controller,
            self.spec_frames,
            display_config,
        )

        # Set up callbacks
        self.playback_handler.on_update_display = self._update_spectrogram_view
        self.playback_handler.on_update_time_axis = self._update_time_axis_labels
        self.playback_handler.on_draw_idle = self.draw_idle
        self.playback_handler.on_playback_finished = self._on_playback_finished

        # Initialize state
        self._init_state()

        # Initialize spectrogram display
        self._initialize_spectrogram_display()

        # Set up event bindings
        self._setup_event_bindings()

        # Audio queue for thread-safe updates
        self.audio_queue = queue.Queue(maxsize=100)

    # Properties for compatibility
    @property
    def all_spec_frames(self) -> List[np.ndarray]:
        """Get all recorded spec frames."""
        # Return frames from the appropriate source
        if self.recording_display.all_spec_frames:
            return self.recording_display.all_spec_frames
        else:
            return self.recording_handler.all_spec_frames

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording_handler.is_recording

    @property
    def frame_count(self) -> int:
        """Get current frame count."""
        return self.recording_handler.frame_count

    @property
    def recording_duration(self) -> float:
        """Get recording duration."""
        return self.recording_display.recording_duration

    @property
    def view_context(self) -> ViewContext:
        """Get current view context for visualization updates."""
        return ViewContext(
            spec_frames=self.spec_frames,
            n_mels=self._recording_n_mels,
            zoom_controller=self.zoom_controller,
            recording_duration=self.recording_display.recording_duration,
        )

    @property
    def axes_pixel_bounds(self) -> tuple:
        """Get axes boundaries in pixel coordinates.

        Returns:
            Tuple of (ax_left, ax_width) in pixels.
        """
        bbox = self.ax.get_position()
        fig_width = self.fig.get_figwidth() * self.fig.dpi
        return bbox.x0 * fig_width, bbox.width * fig_width

    @property
    def visible_time_range(self) -> tuple:
        """Get currently visible time range.

        Returns:
            Tuple of (view_start, view_end) in seconds.
        """
        visible_seconds = self.zoom_controller.get_visible_seconds()
        view_start = self.zoom_controller.view_offset
        return view_start, view_start + visible_seconds

    @property
    def _has_loaded_recording(self) -> bool:
        """Check if a recording is loaded (not in live mode)."""
        return self.recording_display.recording_duration > 0

    def _create_empty_spectrogram(self, n_mels: int = None) -> np.ndarray:
        """Create empty spectrogram data filled with minimum dB value.

        Args:
            n_mels: Number of mel bins, defaults to adaptive_n_mels.

        Returns:
            Empty spectrogram array.
        """
        if n_mels is None:
            n_mels = self.adaptive_n_mels
        return np.ones((n_mels, self.spec_frames)) * AudioConstants.DB_MIN

    def _display_empty_spectrogram(self) -> None:
        """Display an empty spectrogram with default mel bins."""
        self.update_display_data(self._create_empty_spectrogram(), self.adaptive_n_mels)

    def _set_recording_params(
        self, n_mels: int, sample_rate: int, fmax: float = None
    ) -> None:
        """Set recording-specific parameters for frequency display.

        Args:
            n_mels: Number of mel bins for the recording
            sample_rate: Sample rate of the recording
            fmax: Maximum frequency, defaults to sample_rate / 2
        """
        self._recording_n_mels = n_mels
        self._recording_sample_rate = sample_rate
        self._recording_fmax = fmax if fmax is not None else sample_rate / 2

    def _init_state(self) -> None:
        """Initialize widget state."""
        self.zoom_indicator = None
        self.no_data_text = None
        self.current_time = 0
        self.recording_update_id = None
        self._pan_active = False
        self._pan_last_x = 0
        self.edge_indicator: EdgeIndicator | None = None

        # Selection state and interaction handler
        self.selection_state = SelectionState()
        self.selection_handler = SelectionInteractionHandler(self)

        # View state for playback restoration
        self._saved_view_state = SavedViewState()

    @staticmethod
    def _clear_queue(queue_to_clear: queue.Queue) -> None:
        """Clear all items from a queue.

        Args:
            queue_to_clear: The queue to clear
        """
        while not queue_to_clear.empty():
            try:
                queue_to_clear.get_nowait()
            except queue.Empty:
                break

    def _initialize_spectrogram_display(self) -> None:
        """Initialize the spectrogram display with empty data and correct axis limits."""
        initial_data = self._create_empty_spectrogram()
        self.im = self._create_spectrogram_imshow(initial_data, self.adaptive_n_mels)

        # Set axis limits to match extent
        self.ax.set_xlim(0, self.spec_frames - 1)
        self.ax.set_ylim(0, self.adaptive_n_mels - 1)

        # Prepare edge indicators
        self._init_edge_indicator()
        self.edge_indicator.ensure_created(self.spec_frames)
        self.edge_indicator.update_positions(self.spec_frames)

    def _setup_event_bindings(self) -> None:
        """Set up mouse and keyboard event bindings."""
        # Mouse wheel for zoom
        self.canvas_widget.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas_widget.bind("<Button-4>", self._on_mouse_wheel)  # Linux
        self.canvas_widget.bind("<Button-5>", self._on_mouse_wheel)  # Linux

        # Double-click to reset zoom
        self.canvas_widget.bind("<Double-Button-1>", self._reset_zoom)

        # Middle mouse button drag for panning
        self.canvas_widget.bind("<ButtonPress-2>", self._on_middle_press)
        self.canvas_widget.bind("<B2-Motion>", self._on_middle_drag)
        self.canvas_widget.bind("<ButtonRelease-2>", self._on_middle_release)
        # Fallback: some platforms report middle as Button-3
        self.canvas_widget.bind("<ButtonPress-3>", self._on_middle_press)
        self.canvas_widget.bind("<B3-Motion>", self._on_middle_drag)
        self.canvas_widget.bind("<ButtonRelease-3>", self._on_middle_release)

        # Left mouse button for marker/selection
        self.canvas_widget.bind("<ButtonPress-1>", self.selection_handler.on_left_click)
        self.canvas_widget.bind("<B1-Motion>", self.selection_handler.on_left_drag)
        self.canvas_widget.bind(
            "<ButtonRelease-1>", self.selection_handler.on_left_release
        )

        # Mouse motion for hover detection (resize cursor)
        self.canvas_widget.bind("<Motion>", self.selection_handler.on_mouse_motion)

    def _update_mel_processor(self, sample_rate: int) -> None:
        """Update mel processor if sample rate has changed.

        Args:
            sample_rate: New sample rate
        """
        try:
            # Recreate mel processor with new sample rate
            new_mel_processor, new_adaptive_n_mels = MelSpectrogramProcessor.create_for(
                sample_rate, self.display_config.fmin
            )

            # Update audio config
            self.audio_config.sample_rate = sample_rate

            # Update mel processor
            self.mel_processor = new_mel_processor
            self.adaptive_n_mels = new_adaptive_n_mels

            # Update recording handler with new mel processor
            self.recording_handler.mel_processor = new_mel_processor
            self.recording_handler.n_mels = new_adaptive_n_mels

            self._set_recording_params(new_adaptive_n_mels, sample_rate)

            # Update the display y-axis limits for new mel bin count
            self.ax.set_ylim(0, new_adaptive_n_mels - 1)

            # Reinitialize the spectrogram display with new dimensions
            initial_data = self._create_empty_spectrogram(new_adaptive_n_mels)
            self._update_or_recreate_image(
                initial_data, new_adaptive_n_mels, force_recreate=True
            )

        except Exception as e:
            print(f"Error updating mel processor for sample rate {sample_rate}: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Recording methods
    def _update_recording_display(self) -> None:
        """Update the display for recording mode using playback approach."""
        # Calculate how many frames represent 3 seconds
        frames_for_3_seconds = int(
            UIConstants.SPECTROGRAM_DISPLAY_SECONDS * self.frames_per_second
        )
        all_frames = self.recording_handler.all_spec_frames
        if not all_frames:
            # No frames yet - show empty display
            self._display_empty_spectrogram()
            return

        # Calculate which frames to show (last 3 seconds or all if less)
        total_frames = len(all_frames)
        if total_frames > frames_for_3_seconds:
            # Show last 3 seconds
            start_frame = total_frames - frames_for_3_seconds
            end_frame = total_frames
        else:
            # Show all frames
            start_frame = 0
            end_frame = total_frames

        # Get visible frames
        visible_frames = all_frames[start_frame:end_frame]

        if visible_frames:
            # Use the same display method as playback - resample to window width
            # Pass min_duration_seconds=3 to ensure padding for recordings less than 3 seconds
            self._display_resampled_frames(
                visible_frames,
                start_frame,
                end_frame,
                min_duration_seconds=UIConstants.SPECTROGRAM_DISPLAY_SECONDS,
            )

    def start_recording(self, sample_rate: int) -> None:
        """Start recording animation.

        Args:
            sample_rate: Sample rate for the recording
        """
        self._hide_no_data_message()
        get_adaptive_frame_rate().reset()
        self._clear_queue(self.audio_queue)

        # Update mel processor if sample rate has changed
        self._update_mel_processor(sample_rate)

        self.recording_handler.configure_for_sample_rate(sample_rate)
        self.frames_per_second = self.recording_handler.frames_per_second
        self.time_per_frame = AudioConstants.HOP_LENGTH / sample_rate

        # Set recording-specific parameters for live recording
        params = MEL_CONFIG.calculate_params(sample_rate, self.display_config.fmin)
        self._set_recording_params(params["n_mels"], sample_rate, params["fmax"])

        self._update_frequency_axis(sample_rate)
        # Defensive: ensure no old clipping markers leak into live monitor/recording
        self.clipping_visualizer.clear()
        self.recording_handler.start_recording()

        # Reset time axis - always show 3 seconds for recording
        self._update_time_axis_labels(0, UIConstants.SPECTROGRAM_DISPLAY_SECONDS)

        # Clear display with empty data first
        self._display_empty_spectrogram()

        # Cache the background with empty spectrogram (needed for blitting)
        if self.use_blitting:
            self.invalidate_background()
            self.canvas.draw()
            self.cache_background()

        # Start displaying real data
        self._update_recording_display()

        # Start periodic updates for recording
        self._start_recording_updates()

    def stop_recording(self) -> None:
        """Stop recording animation."""
        self.recording_handler.stop_recording()
        # Stop periodic updates
        self._stop_recording_updates()

    def update_audio(self, audio_chunk: np.ndarray) -> None:
        """Update with new audio data during recording."""
        # Let recording_handler decide - allows updates when meters toggled
        try:
            self.audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            # Skip if queue is full
            pass

    def _update_display(self) -> None:
        """Update display from audio queue."""
        # Process all pending audio chunks for real-time display
        chunks_processed = 0
        display_needs_update = False

        # Process all available chunks to prevent queue buildup
        while not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get_nowait()
                should_update = self.recording_handler.update_audio(audio_chunk)

                if should_update:
                    display_needs_update = True
                    # Update time tracking
                    self.current_time = self.recording_handler.current_time
                    self.max_detected_freq = self.recording_handler.max_detected_freq

                chunks_processed += 1

            except queue.Empty:
                break

        # Update display once after processing all chunks
        if display_needs_update and self.recording_handler.is_recording:
            self._update_recording_display()
            self._update_clipping_markers_live()

        if self.recording_handler.is_recording or self.playback_controller.is_playing:
            self._update_frequency_display()

        if display_needs_update:
            self.draw_idle()

    # Playback methods
    def start_playback(
        self,
        duration: float,
        sample_rate: int,
        start_position: float = 0.0,
        end_position: Optional[float] = None,
    ) -> None:
        """Start playback animation.

        Args:
            duration: Playback duration in seconds
            sample_rate: Sample rate of the audio being played
            start_position: Start position in seconds (default 0.0)
            end_position: End position in seconds (default None = play to end)
        """
        self._hide_no_data_message()
        get_adaptive_frame_rate().reset()
        self._save_playback_view_state()

        # Hide position marker during playback (playback marker takes over)
        # Keep selection visible so user can see what range is being played
        if self.selection_state.has_marker and not self.selection_state.has_selection:
            self.selection_visualizer.hide()

        recording_duration = self.recording_display.recording_duration
        if recording_duration <= 0:
            raise ValueError(
                "Cannot start playback: No recording loaded (recording_duration <= 0)"
            )

        self.playback_handler.start_playback(
            duration, recording_duration, sample_rate, start_position, end_position
        )

    def stop_playback(self) -> None:
        """Stop playback animation."""
        self.playback_handler.stop_playback()
        # Restore selection visualization after playback stops
        self._restore_selection_visualization()

    def _on_playback_finished(self) -> None:
        """Called when playback finishes naturally."""
        self._restore_selection_visualization()

    def _save_playback_view_state(self) -> None:
        """Save the current view state before playback starts.

        Saves the marker/selection position and its relative position in the
        viewport so we can restore the view after playback finishes.
        """
        self._saved_view_state.clear()

        # Determine the target time (marker or selection start)
        target_time = None
        if self.selection_state.has_marker:
            target_time = self.selection_state.marker_position
        elif self.selection_state.has_selection:
            target_time = self.selection_state.selection_start

        if target_time is None:
            return

        # Calculate relative position of target in current viewport (0.0 to 1.0)
        visible_seconds = self.zoom_controller.get_visible_seconds()
        if visible_seconds <= 0:
            return

        view_start = self.zoom_controller.view_offset
        relative_pos = (target_time - view_start) / visible_seconds

        # Only save if target is currently visible
        if 0.0 <= relative_pos <= 1.0:
            self._saved_view_state.save(target_time, relative_pos)

    def _restore_playback_view_state(self) -> None:
        """Restore the view to show marker/selection at its original relative position."""
        saved = self._saved_view_state.get()
        if saved is None:
            return

        target_time, relative_pos = saved

        recording_duration = self.recording_display.recording_duration
        if recording_duration <= 0:
            return

        visible_seconds = self.zoom_controller.get_visible_seconds()

        # Calculate the view_offset needed to place target at relative_pos
        desired_offset = target_time - (relative_pos * visible_seconds)

        # Clamp to valid range (no negative offset, no empty space at end)
        max_offset = max(0.0, recording_duration - visible_seconds)
        new_offset = max(0.0, min(desired_offset, max_offset))

        # Only update if offset changed
        if abs(new_offset - self.zoom_controller.view_offset) > 0.001:
            self.zoom_controller.view_offset = new_offset
            self._update_after_zoom()

        self._saved_view_state.clear()

    def _restore_selection_visualization(self) -> None:
        """Restore marker visualization and view position after playback."""
        # Restore view position first
        self._restore_playback_view_state()

        # Only restore marker visuals if it was set (selection stays visible)
        if not self.selection_state.has_marker:
            return

        ctx = self.view_context
        if ctx.has_recording:
            self.selection_visualizer.update_marker(
                self.selection_state.marker_position, ctx
            )
            self.draw_idle()

    # Display methods
    def show_recording(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Display a complete recording."""
        # Hide NO DATA message if visible
        self._hide_no_data_message()

        # Process recording
        display_data, adaptive_n_mels, duration = (
            self.recording_display.process_recording(audio_data, sample_rate)
        )

        # Store recording-specific parameters for frequency display
        self._set_recording_params(adaptive_n_mels, sample_rate)
        self.max_detected_freq = self.recording_display.max_detected_freq

        self._set_zoom_indicator_visible(False)

        self._finalize_recording_display(
            display_data,
            adaptive_n_mels,
            self.recording_display.recording_duration,
            self._recording_sample_rate,
        )

    def _finalize_recording_display(
        self, display_data: np.ndarray, n_mels: int, duration: float, sample_rate: int
    ) -> None:
        """Finalize the recording display with all UI updates.

        This method handles all the display updates needed after loading or
        refreshing the spectrogram after resize.

        Args:
            display_data: The spectrogram data to display
            n_mels: Number of mel bins
            duration: Recording duration in seconds
            sample_rate: Sample rate of the recording
        """
        self._update_or_recreate_image(display_data, n_mels)

        # Update clipping markers
        self.clipping_visualizer.update_display(
            len(self.recording_display.all_spec_frames), self.spec_frames
        )

        self._update_frequency_axis(sample_rate)
        # IMPORTANT: Set y-axis limits AFTER frequency axis update
        self.ax.set_ylim(0, n_mels - 1)
        # Update frequency display (including max freq indicator) AFTER setting ylim
        self._update_frequency_display()

        # Update time axis
        self.ax.set_xlim(0, self.spec_frames - 1)
        self._apply_adaptive_layout()
        self._update_time_axis_labels(0, duration)

        # Invalidate background after major changes
        # self.invalidate_background()
        # self.canvas.draw()
        # Cache background after full draw
        # self.cache_background()

    def cleanup(self) -> None:
        """Clean up resources before widget destruction."""
        # Stop all periodic updates
        self._stop_recording_updates()

        # Stop playback if running
        if self.playback_handler and self.playback_controller.is_playing:
            self.playback_handler.stop_playback()

        # Clear the audio queue
        self._clear_queue(self.audio_queue)

    def clear(self) -> None:
        """Clear the spectrogram display."""
        self.recording_handler.clear()
        self.recording_display.clear()
        self.zoom_controller.set_recording_duration(0)
        self.clipping_visualizer.clear()
        self.selection_state.clear_all()
        self.selection_visualizer.clear()

        # Reset recording-specific parameters to defaults from current audio config
        self._set_recording_params(self.adaptive_n_mels, self.audio_config.sample_rate)

        # Reset frequency axis with default parameters
        self.freq_axis_manager.update_default_axis(
            self.adaptive_n_mels,
            self.mel_processor.fmin,
            self.mel_processor.actual_fmax,
        )

        # Reset display
        self._display_empty_spectrogram()

        # Show "NO DATA" text in the center
        self._show_no_data_message()

        # Reset y-axis to default range
        self.ax.set_ylim(0, self.adaptive_n_mels - 1)

        # Reset time axis to default 3 seconds
        self._update_time_axis_labels(0, UIConstants.SPECTROGRAM_DISPLAY_SECONDS)

        self.draw_idle()

    # Zoom methods
    def _on_mouse_wheel(self, event) -> None:
        """Handle mouse wheel zoom events."""
        mouse_rel_x = self.get_mouse_position_in_axes(event)
        current_time = self.recording_handler.current_time
        zoom_in = event.num == UIConstants.TK_EVENT_SCROLL_UP or event.delta > 0
        if self.zoom_controller.apply_zoom_at_position(
            mouse_rel_x, zoom_in, current_time
        ):
            self._update_after_zoom()

    def get_mouse_position_in_axes(self, event) -> float:
        """Get mouse position relative to axes (0-1), clamped to valid range.

        Positions outside axes are clamped to boundaries (0.0 or 1.0).
        This ensures selections can always reach the start/end of recordings.
        """
        ax_left, ax_width = self.axes_pixel_bounds
        mouse_rel_x = (event.x - ax_left) / ax_width
        return max(0.0, min(1.0, mouse_rel_x))

    def _reset_zoom(self, event=None) -> None:
        """Reset zoom to 1x."""
        self.zoom_controller.reset()

        self._update_time_axis_for_current_state()
        self._set_zoom_indicator_visible(False)
        if self.recording_display.all_spec_frames:
            self._update_spectrogram_view()

        self.draw_idle()

    # --- Middle-mouse panning ---
    def _on_middle_press(self, event) -> None:
        """Start panning with middle mouse button."""
        self._pan_active = True
        self._pan_last_x = event.x

    def _on_middle_drag(self, event) -> None:
        """Handle panning while middle mouse is held down."""
        if not getattr(self, "_pan_active", False):
            return

        dx_pixels = event.x - getattr(self, "_pan_last_x", event.x)
        self._pan_last_x = event.x

        if dx_pixels == 0:
            return

        # Convert pixel delta to time delta
        _, ax_width_px = self.axes_pixel_bounds
        if ax_width_px <= 0:
            return

        view_start, view_end = self.visible_time_range
        seconds_per_pixel = (view_end - view_start) / ax_width_px

        # Negative dx (drag left) should move view to earlier time (decrease offset)
        delta_seconds = -dx_pixels * seconds_per_pixel

        # For live mode, cap using current_time so the right edge won't exceed content
        is_live = not self._has_loaded_recording
        current_time = self.current_time if is_live else 0.0

        # Compute boundary before/after for indicator decision
        self.zoom_controller.view_offset
        new_offset = self.zoom_controller.pan_by_seconds(
            delta_seconds, current_time=current_time
        )

        # Decide if we hit a boundary and show an indicator briefly
        visible_seconds = self.zoom_controller.get_visible_seconds()
        if is_live:
            max_offset = max(0.0, max(0.0, current_time) - visible_seconds)
        else:
            max_offset = max(
                0.0,
                max(0.0, self.recording_display.recording_duration) - visible_seconds,
            )

        if self.edge_indicator is not None:
            if new_offset <= 0.0 and delta_seconds < 0:
                self.edge_indicator.show("left")
            elif new_offset >= max_offset and delta_seconds > 0:
                self.edge_indicator.show("right")

        # Update display in-place without changing zoom
        self._refresh_viewport()

    def _on_middle_release(self, event) -> None:
        """Finish panning with middle mouse button."""
        self._pan_active = False

    def _update_after_zoom(self) -> None:
        """Update display after zoom change."""
        self._update_zoom_indicator()
        self._refresh_viewport()

    def _refresh_viewport(self) -> None:
        """Refresh spectrogram view, time axis, and selection display.

        Common sequence called after pan or zoom adjustments.
        """
        view_start, view_end = self.visible_time_range
        self._update_time_axis_labels(view_start, view_end)
        self._update_spectrogram_view()
        self._update_selection_display()

        self.draw_idle()

    def _update_selection_for_resize(self) -> None:
        """Update selection/marker display after window resize."""
        self._update_selection_display()

    def _update_selection_display(self) -> None:
        """Update selection and marker visualization.

        Called after zoom changes or window resize to recalculate
        marker positions based on current spec_frames and view offset.
        """
        ctx = self.view_context
        if ctx.has_recording:
            self.selection_visualizer.update_for_zoom(self.selection_state, ctx)

    def _set_zoom_indicator_visible(self, visible: bool) -> None:
        """Set zoom indicator visibility.

        Args:
            visible: True to show, False to hide
        """
        if self.zoom_indicator is not None:
            self.zoom_indicator.set_visible(visible)

    def _update_zoom_indicator(self) -> None:
        """Update or create zoom indicator text."""
        if self._has_loaded_recording:
            visible_seconds = (
                self.recording_display.recording_duration
                / self.zoom_controller.zoom_level
            )
        else:
            visible_seconds = (
                UIConstants.SPECTROGRAM_DISPLAY_SECONDS
                / self.zoom_controller.zoom_level
            )

        indicator_text = (
            f"Zoom: {self.zoom_controller.zoom_level:.1f}x ({visible_seconds:.2f}s)"
        )

        if self.zoom_indicator is not None:
            self.zoom_indicator.set_text(indicator_text)
            self.zoom_indicator.set_visible(True)
        else:
            self.zoom_indicator = self.ax.text(
                0.98,
                0.95,
                indicator_text,
                transform=self.ax.transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=self.ZOOM_INDICATOR_FONTSIZE,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

        # Auto-hide after delay
        if self.zoom_controller.zoom_level == 1.0:
            self.canvas_widget.after(
                self.ZOOM_INDICATOR_HIDE_DELAY_MS, self._hide_zoom_indicator
            )

    def _hide_zoom_indicator(self) -> None:
        """Hide zoom indicator if still at 1x."""
        if self.zoom_controller.zoom_level == 1.0:
            self._set_zoom_indicator_visible(False)
            self.draw_idle()

    def _update_time_axis_for_current_state(self) -> None:
        """Update time axis based on current recording state."""
        if self._has_loaded_recording:
            self._update_time_axis_labels(0, self.recording_display.recording_duration)
        else:
            self._update_time_axis_labels(0, UIConstants.SPECTROGRAM_DISPLAY_SECONDS)

    def _create_spectrogram_imshow(self, data: np.ndarray, n_mels: int) -> AxesImage:
        """Create a new imshow with standard parameters.

        Args:
            data: The spectrogram data to display
            n_mels: Number of mel bins
        """
        im = self.ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            animated=True,
            cmap=theme_manager.colormap,
            interpolation="bilinear",
            vmin=AudioConstants.DB_MIN,
            vmax=AudioConstants.DB_MAX,
            extent=(0, self.spec_frames - 1, 0, n_mels - 1),
        )

        # Add to animated artists list for blitting
        if im not in self.animated_artists:
            self.animated_artists.append(im)

        return im

    def _update_or_recreate_image(
        self, data: np.ndarray, n_mels: int, force_recreate: bool = False
    ) -> None:
        """Update existing image or recreate if dimensions changed.

        Args:
            data: The spectrogram data to display
            n_mels: Number of mel bins
            force_recreate: Force recreation even if dimensions match
        """
        current_shape = self.im.get_array().shape if self.im else None
        needs_recreation = (
            force_recreate
            or not current_shape
            or current_shape[0] != n_mels
            or current_shape[1] != self.spec_frames
        )

        if needs_recreation:
            if self.im:
                self.im.remove()
            self.im = self._create_spectrogram_imshow(data, n_mels)
        else:
            self.update_display_data(data, n_mels)

    # Update methods
    def _update_spectrogram_view(self) -> None:
        """Update spectrogram display based on zoom/offset."""
        # Get the correct frame source
        if self.all_spec_frames:
            if self._has_loaded_recording:
                self._update_recording_view()
            else:
                self._update_live_view()

    def _update_recording_view(self) -> None:
        """Update view for loaded recordings."""
        start_frame, end_frame = self.recording_display.calculate_visible_frame_range()
        visible_frames = self.recording_display.get_visible_frames(
            start_frame, end_frame
        )

        if visible_frames:
            self._display_resampled_frames(visible_frames, start_frame, end_frame)

    def _update_live_view(self) -> None:
        """Update view for live recording."""
        frames = self.recording_handler.all_spec_frames
        start_frame, visible_frames = (
            self.zoom_controller.calculate_visible_frame_range(self.frames_per_second)
        )
        end_frame = min(start_frame + visible_frames, len(frames))

        if start_frame < len(frames):
            visible_data = frames[start_frame:end_frame]

            if visible_data:
                # Use the same resampling method as recording view
                self._display_resampled_frames(visible_data, start_frame, end_frame)

    def _display_resampled_frames(
        self,
        visible_frames: List[np.ndarray],
        start_frame: int,
        end_frame: int,
        min_duration_seconds: Optional[float] = None,
    ) -> None:
        """Display resampled frames with clipping markers.

        Args:
            visible_frames: List of frame arrays to display
            start_frame: Starting frame index
            end_frame: Ending frame index
            min_duration_seconds: Minimum duration to display (pads with zeros if needed)
        """
        visible_array = np.array(visible_frames).T
        n_mels = visible_array.shape[0]
        n_frames_visible = visible_array.shape[1]

        # Handle minimum duration padding
        if min_duration_seconds is not None:
            min_frames = int(min_duration_seconds * self.frames_per_second)
            if n_frames_visible < min_frames:
                # Prepend zeros to the left (oldest data left, newest right)
                padding_frames = min_frames - n_frames_visible
                padding = np.ones((n_mels, padding_frames)) * AudioConstants.DB_MIN
                visible_array = np.hstack([padding, visible_array])
                n_frames_visible = visible_array.shape[1]

        if n_frames_visible > 1:
            # Resample to fit display
            resampled = self._resample_frames_to_display(
                visible_array, n_mels, n_frames_visible
            )
            self.update_display_data(resampled, n_mels)
        else:
            self.update_display_data(visible_array, n_mels)

        # Update clipping markers only for playback (not live recording)
        # For live recording, markers are updated separately in _update_display()
        if not self.recording_handler.is_recording:
            self.clipping_visualizer.update_markers_for_zoom(
                start_frame, end_frame, self.spec_frames
            )
            self.clipping_visualizer.show_warning()

    def _update_clipping_markers_live(self) -> None:
        """Update clipping markers during live recording."""
        self.clipping_visualizer.update_markers_for_live(
            self.current_time,
            self.recording_handler.frame_count,
            self.spec_frames,
            self.frames_per_second,
            self.zoom_controller.zoom_level,
        )
        self.clipping_visualizer.show_warning()

    def _update_frequency_axis(self, sample_rate: int) -> None:
        """Update frequency axis for a specific recording."""
        # Update frequency axis using the recording_axis method
        self.freq_axis_manager.update_recording_axis(
            sample_rate, self.display_config.fmin
        )

    def _update_frequency_display(self) -> None:
        """Update highest frequency display."""
        if self.max_detected_freq > 0:
            # Use recording-specific parameters
            self.freq_axis_manager.highlight_max_frequency(
                self.max_detected_freq,
                self._recording_n_mels,
                self.mel_processor.fmin,
                self._recording_fmax,
            )

    def _on_spec_frames_changed(self, old_frames: int, new_frames: int) -> None:
        """Handle spec_frames change due to window resize.

        This is called from within _on_resize() event inside the base class.
        """
        # Update handlers with new spec_frames
        self.recording_handler.spec_frames = new_frames
        self.playback_handler.spec_frames = new_frames
        self.recording_display.spec_frames = new_frames

        # Update edge indicator positions to match new width
        if self.edge_indicator is not None:
            self.edge_indicator.update_positions(new_frames)

        # Recreate spectrogram display with new dimensions
        if self.im:
            self._update_spectrogram_view()

        # Update time axis to ensure full time range is shown
        self._update_time_axis_for_current_state()
        self._refresh_display()

        # Update selection/marker positions for new spec_frames
        self._update_selection_for_resize()

        # Also apply current zoom-level
        if self.zoom_controller.zoom_level > 1.0:
            self._update_after_zoom()

    def _refresh_display(self) -> None:
        """Refresh the display after spec_frames change."""
        if self._has_loaded_recording:
            # We have a loaded recording - resample it for new display width
            display_data = self.recording_display.resample_spectrogram_for_display(
                np.array(self.recording_display.all_spec_frames).T,
                len(self.recording_display.all_spec_frames),
                self._recording_n_mels,
            )
            n_mels = self._recording_n_mels
            duration = self.recording_display.recording_duration
            sample_rate = self._recording_sample_rate
        else:
            # No recording - create empty display data
            display_data = self._create_empty_spectrogram()
            n_mels = self.adaptive_n_mels
            duration = UIConstants.SPECTROGRAM_DISPLAY_SECONDS
            sample_rate = self.audio_config.sample_rate

        self._finalize_recording_display(display_data, n_mels, duration, sample_rate)

    def _on_figure_size_changed(self) -> None:
        """Called when figure size changes but spec_frames stays the same."""
        # When the figure size changes but spec_frames stays constant,
        # we still need to redraw to ensure the plot fills the canvas
        if self.recording_display.all_spec_frames:
            self._refresh_display()
        else:
            self.canvas.draw()

    # --- Edge indicator helpers ---
    def _init_edge_indicator(self) -> None:
        """Initialize edge indicator controller."""
        if self.edge_indicator is None:
            self.edge_indicator = EdgeIndicator(
                self.ax,
                color=UIConstants.COLOR_EDGE_INDICATOR,
                linewidth=UIConstants.EDGE_INDICATOR_WIDTH,
                alpha=UIConstants.EDGE_INDICATOR_ALPHA,
                timeout_ms=UIConstants.EDGE_INDICATOR_TIMEOUT_MS,
                after_call=self.parent.after,
            )

    def schedule_update(self) -> None:
        """Schedule a display update (called from main app)."""
        self._update_display()

    def _start_recording_updates(self) -> None:
        """Start periodic display updates during recording."""
        # Cancel any existing update
        if self.recording_update_id:
            self.parent.after_cancel(self.recording_update_id)

        # Schedule first update
        self._recording_update_loop()

    def _stop_recording_updates(self) -> None:
        """Stop periodic display updates."""
        if self.recording_update_id:
            try:
                self.parent.after_cancel(self.recording_update_id)
            except tk.TclError:
                # Widget might be destroyed already
                pass
            self.recording_update_id = None

    def _recording_update_loop(self) -> None:
        """Periodic update loop for recording display with adaptive timing."""
        if self.recording_handler.is_recording:
            afr = get_adaptive_frame_rate()
            afr.frame_start()
            self._update_display()
            update_interval = afr.frame_end()
            if DEBUG_FPS:
                print(
                    f"[REC] overshoot={afr.get_overshoot():.1f}ms "
                    f"interval={update_interval}ms fps={afr.get_current_fps():.1f}"
                )
            self.recording_update_id = self.parent.after(
                update_interval, self._recording_update_loop
            )

    def _show_no_data_message(self) -> None:
        """Show 'NO DATA' message in the center of spectrogram."""
        if self.no_data_text is not None:
            self.no_data_text.set_visible(True)
        else:
            self.no_data_text = self.ax.text(
                0.5,
                0.5,
                "NO DATA",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=UIConstants.COLOR_TEXT_SECONDARY,
                alpha=0.5,
            )

    def _hide_no_data_message(self) -> None:
        """Hide 'NO DATA' message."""
        if self.no_data_text is not None:
            self.no_data_text.set_visible(False)
            self.draw_idle()

    # --- Selection manipulation (delegated to handler) ---

    def _set_marker(self, time_seconds: float) -> None:
        """Set marker at the specified time position.

        Args:
            time_seconds: Position in seconds
        """
        self.selection_handler.set_marker(time_seconds)

    def _set_selection(self, start_time: float, end_time: float) -> None:
        """Set selection range.

        Args:
            start_time: Selection start in seconds
            end_time: Selection end in seconds
        """
        self.selection_handler.set_selection(start_time, end_time)

    def clear_selection(self) -> None:
        """Clear marker and selection."""
        self.selection_handler.clear_selection()
