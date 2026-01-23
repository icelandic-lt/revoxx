"""Main mel spectrogram widget that coordinates all components."""

from typing import Optional, List
import numpy as np
import tkinter as tk
import queue

from matplotlib.image import AxesImage

from ...constants import AudioConstants
from ...constants import UIConstants
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
    def _view_context(self) -> ViewContext:
        """Get current view context for visualization updates."""
        return ViewContext(
            spec_frames=self.spec_frames,
            n_mels=self._recording_n_mels,
            zoom_controller=self.zoom_controller,
            recording_duration=self.recording_display.recording_duration,
        )

    def _init_state(self) -> None:
        """Initialize widget state."""
        self.zoom_indicator = None
        self.current_time = 0
        self.recording_update_id = None
        self._pan_active = False
        self._pan_last_x = 0
        self.edge_indicator: EdgeIndicator | None = None

        # Selection state
        self.selection_state = SelectionState()
        self._drag_start_x: Optional[float] = None
        self._drag_start_time: Optional[float] = None

        # Marker drag state
        self._resize_target: Optional[str] = None  # "start", "end", "position", or None
        self._resize_active: bool = False
        self._default_cursor: str = ""

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
        initial_data = (
            np.ones((self.adaptive_n_mels, self.spec_frames)) * AudioConstants.DB_MIN
        )
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
        self.canvas_widget.bind("<ButtonPress-1>", self._on_left_click)
        self.canvas_widget.bind("<B1-Motion>", self._on_left_drag)

        # Mouse motion for hover detection (resize cursor)
        self.canvas_widget.bind("<Motion>", self._on_mouse_motion)
        self.canvas_widget.bind("<ButtonRelease-1>", self._on_left_release)

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

            # Update recording-specific parameters
            self._recording_n_mels = new_adaptive_n_mels
            self._recording_sample_rate = sample_rate
            self._recording_fmax = sample_rate / 2

            # Update the display y-axis limits for new mel bin count
            self.ax.set_ylim(0, new_adaptive_n_mels - 1)

            # Reinitialize the spectrogram display with new dimensions
            initial_data = (
                np.ones((new_adaptive_n_mels, self.spec_frames)) * AudioConstants.DB_MIN
            )
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
            empty_data = (
                np.ones((self.adaptive_n_mels, self.spec_frames))
                * AudioConstants.DB_MIN
            )
            self.update_display_data(empty_data, self.adaptive_n_mels)
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
        # Hide NO DATA message if visible
        self._hide_no_data_message()

        # Clear the audio queue first
        self._clear_queue(self.audio_queue)

        # Update mel processor if sample rate has changed
        self._update_mel_processor(sample_rate)

        self.recording_handler.configure_for_sample_rate(sample_rate)
        self.frames_per_second = self.recording_handler.frames_per_second
        self.time_per_frame = AudioConstants.HOP_LENGTH / sample_rate

        # Set recording-specific parameters for live recording
        # Calculate adaptive parameters based on the recording sample rate

        params = MEL_CONFIG.calculate_params(sample_rate, self.display_config.fmin)
        self._recording_sample_rate = sample_rate
        self._recording_n_mels = params["n_mels"]
        self._recording_fmax = params["fmax"]

        self._update_frequency_axis(sample_rate)
        # Defensive: ensure no old clipping markers leak into live monitor/recording
        self.clipping_visualizer.clear()
        self.recording_handler.start_recording()

        # Reset time axis - always show 3 seconds for recording
        self._update_time_axis_labels(0, UIConstants.SPECTROGRAM_DISPLAY_SECONDS)

        # Clear display with empty data first
        empty_data = (
            np.ones((self.adaptive_n_mels, self.spec_frames)) * AudioConstants.DB_MIN
        )
        self.update_display_data(empty_data, self.adaptive_n_mels)

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
        # Hide NO DATA message if visible
        self._hide_no_data_message()

        # Save marker position info to restore view after playback
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

        ctx = self._view_context
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
        self._recording_n_mels = adaptive_n_mels
        self._recording_sample_rate = sample_rate
        self._recording_fmax = sample_rate / 2
        self.max_detected_freq = self.recording_display.max_detected_freq

        if self.zoom_indicator:
            self.zoom_indicator.set_visible(False)

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
        if hasattr(self, "audio_queue"):
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
        self._recording_sample_rate = self.audio_config.sample_rate
        self._recording_n_mels = self.adaptive_n_mels
        self._recording_fmax = self.audio_config.sample_rate / 2

        # Reset frequency axis with default parameters
        self.freq_axis_manager.update_default_axis(
            self.adaptive_n_mels,
            self.mel_processor.fmin,
            self.mel_processor.actual_fmax,
        )

        # Reset display
        empty_data = (
            np.ones((self.adaptive_n_mels, self.spec_frames)) * AudioConstants.DB_MIN
        )
        self.update_display_data(empty_data, self.adaptive_n_mels)

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
        mouse_rel_x = self._get_mouse_position_in_axes(event)
        if mouse_rel_x is None:
            return

        current_time = self.recording_handler.current_time
        zoom_in = event.num == 4 or event.delta > 0
        if self.zoom_controller.apply_zoom_at_position(
            mouse_rel_x, zoom_in, current_time
        ):
            self._update_after_zoom()

    def _get_mouse_position_in_axes(self, event) -> Optional[float]:
        """Get mouse position relative to axes (0-1)."""
        bbox = self.ax.get_position()
        fig_width = self.fig.get_figwidth() * self.fig.dpi

        ax_left = bbox.x0 * fig_width
        ax_width = bbox.width * fig_width
        mouse_rel_x = (event.x - ax_left) / ax_width
        if mouse_rel_x < 0 or mouse_rel_x > 1:
            return None

        return mouse_rel_x

    def _reset_zoom(self, event=None) -> None:
        """Reset zoom to 1x."""
        self.zoom_controller.reset()

        self._update_time_axis_for_current_state()
        if self.zoom_indicator:
            self.zoom_indicator.set_visible(False)
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

        # Convert pixel delta to time delta based on current visible seconds and pixel width
        bbox = self.ax.get_position()
        fig_width_px = self.fig.get_figwidth() * self.fig.dpi
        ax_width_px = bbox.width * fig_width_px
        if ax_width_px <= 0:
            return

        visible_seconds = self.zoom_controller.get_visible_seconds()
        seconds_per_pixel = visible_seconds / ax_width_px

        # Negative dx (drag left) should move view to earlier time (decrease offset)
        delta_seconds = -dx_pixels * seconds_per_pixel

        # For live mode, cap using current_time so the right edge won't exceed content
        is_live = self.recording_display.recording_duration == 0
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
        self._update_time_axis_labels(
            self.zoom_controller.view_offset,
            self.zoom_controller.view_offset
            + self.zoom_controller.get_visible_seconds(),
        )
        self._update_spectrogram_view()

        # Update selection/marker display for pan
        ctx = self._view_context
        if ctx.has_recording:
            self.selection_visualizer.update_for_zoom(self.selection_state, ctx)

        self.draw_idle()

    def _on_middle_release(self, event) -> None:
        """Finish panning with middle mouse button."""
        self._pan_active = False

    def _update_after_zoom(self) -> None:
        """Update display after zoom change."""
        self._update_zoom_indicator()
        self._update_time_axis_labels(
            self.zoom_controller.view_offset,
            self.zoom_controller.view_offset
            + self.zoom_controller.get_visible_seconds(),
        )
        self._update_spectrogram_view()

        # Update selection/marker display for new zoom
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
        ctx = self._view_context
        if ctx.has_recording:
            self.selection_visualizer.update_for_zoom(self.selection_state, ctx)

    def _update_zoom_indicator(self) -> None:
        """Update or create zoom indicator text."""
        if self.recording_display.recording_duration > 0:
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

        if self.zoom_indicator:
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
        if self.zoom_controller.zoom_level == 1.0 and self.zoom_indicator:
            self.zoom_indicator.set_visible(False)
            self.draw_idle()

    def _update_time_axis_for_current_state(self) -> None:
        """Update time axis based on current recording state."""
        if self.recording_display.recording_duration > 0:
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
            if self.recording_display.recording_duration > 0:
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
        if self.recording_display.recording_duration > 0:
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
            display_data = (
                np.ones((self.adaptive_n_mels, self.spec_frames))
                * AudioConstants.DB_MIN
            )
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
        """Periodic update loop for recording display."""
        if self.recording_handler.is_recording:
            self._update_display()

            # Schedule next update
            self.recording_update_id = self.parent.after(
                UIConstants.ANIMATION_UPDATE_MS, self._recording_update_loop
            )

    def _show_no_data_message(self) -> None:
        """Show 'NO DATA' message in the center of spectrogram."""
        # Add text in the center of the axes
        if hasattr(self, "no_data_text"):
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
        if hasattr(self, "no_data_text"):
            self.no_data_text.set_visible(False)
            # Force redraw to ensure text is gone
            self.draw_idle()

    # --- Mouse motion for hover detection ---
    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for marker hover detection."""
        if self._resize_active:
            return

        mouse_rel_x = self._get_mouse_position_in_axes(event)
        if mouse_rel_x is None:
            self._set_cursor_default()
            self._resize_target = None
            return

        marker = self._get_marker_at_position(event.x)
        if marker:
            self._resize_target = marker
            self._set_cursor_resize()
        else:
            self._resize_target = None
            self._set_cursor_default()

    def _get_marker_at_position(self, pixel_x: float) -> Optional[str]:
        """Check if pixel position is near a marker (selection or position).

        Args:
            pixel_x: X position in pixels

        Returns:
            "start", "end", "position", or None
        """
        recording_duration = self.recording_display.recording_duration
        if recording_duration <= 0:
            return None

        threshold = UIConstants.MARKER_HOVER_THRESHOLD

        # Check selection markers first
        if self.selection_state.has_selection:
            start_pixel = self._time_to_pixel_x(self.selection_state.selection_start)
            if start_pixel is not None and abs(pixel_x - start_pixel) <= threshold:
                return "start"

            end_pixel = self._time_to_pixel_x(self.selection_state.selection_end)
            if end_pixel is not None and abs(pixel_x - end_pixel) <= threshold:
                return "end"

        # Check position marker
        if self.selection_state.has_marker:
            marker_pixel = self._time_to_pixel_x(self.selection_state.marker_position)
            if marker_pixel is not None and abs(pixel_x - marker_pixel) <= threshold:
                return "position"

        return None

    def _time_to_pixel_x(self, time_seconds: float) -> Optional[float]:
        """Convert time in seconds to pixel X position.

        Args:
            time_seconds: Time position in seconds

        Returns:
            Pixel X position, or None if outside visible range
        """
        recording_duration = self.recording_display.recording_duration
        if recording_duration <= 0:
            return None

        visible_seconds = self.zoom_controller.get_visible_seconds()
        view_start = self.zoom_controller.view_offset
        view_end = view_start + visible_seconds

        if time_seconds < view_start or time_seconds > view_end:
            return None

        rel_x = (time_seconds - view_start) / visible_seconds

        bbox = self.ax.get_position()
        fig_width = self.fig.get_figwidth() * self.fig.dpi
        ax_left = bbox.x0 * fig_width
        ax_width = bbox.width * fig_width

        return ax_left + rel_x * ax_width

    def _set_cursor_resize(self) -> None:
        """Set cursor to horizontal resize cursor."""
        self.canvas_widget.config(cursor="sb_h_double_arrow")

    def _set_cursor_default(self) -> None:
        """Reset cursor to default."""
        self.canvas_widget.config(cursor="")

    # --- Left mouse button marker/selection ---
    def _on_left_click(self, event) -> None:
        """Handle left mouse button press for marker/selection or resize."""
        if event.type == "4":  # ButtonPress
            mouse_rel_x = self._get_mouse_position_in_axes(event)
            if mouse_rel_x is None:
                self.clear_selection()
                return

            # Check if starting a resize operation
            if self._resize_target:
                self._resize_active = True
                self._drag_start_x = event.x
                return

            # Store drag start position for new selection
            self._drag_start_x = event.x
            self._drag_start_time = self._pixel_to_time(mouse_rel_x)

    def _on_left_drag(self, event) -> None:
        """Handle left mouse button drag for selection preview or resize."""
        # Handle marker drag (selection resize or position marker move)
        if self._resize_active and self._resize_target:
            self._handle_marker_drag(event)
            return

        if self._drag_start_x is None or self._drag_start_time is None:
            return

        # Check if drag exceeds threshold
        dx = abs(event.x - self._drag_start_x)
        if dx < UIConstants.SELECTION_DRAG_THRESHOLD:
            return

        # Clear any existing marker when starting a new selection
        ctx = self._view_context
        if self.selection_state.has_marker:
            self.selection_state.clear_marker()
            self.selection_visualizer.update_marker(None, ctx)

        # Get current mouse position
        mouse_rel_x = self._get_mouse_position_in_axes(event)
        if mouse_rel_x is None:
            return

        current_time = self._pixel_to_time(mouse_rel_x)
        if current_time is None:
            return

        # Update selection preview
        if ctx.has_recording:
            # Clamp current_time to recording bounds
            current_time = max(0.0, min(current_time, ctx.recording_duration))

            # Normalize start/end so start <= end
            start_time = min(self._drag_start_time, current_time)
            end_time = max(self._drag_start_time, current_time)

            self.selection_visualizer.update_selection(start_time, end_time, ctx)
            self.draw_idle()

    def _handle_marker_drag(self, event) -> None:
        """Handle dragging a marker (selection or position).

        Args:
            event: Mouse event
        """
        mouse_rel_x = self._get_mouse_position_in_axes(event)
        if mouse_rel_x is None:
            return

        new_time = self._pixel_to_time(mouse_rel_x)
        if new_time is None:
            return

        recording_duration = self.recording_display.recording_duration
        if recording_duration <= 0:
            return

        # Handle position marker drag
        ctx = self._view_context

        if self._resize_target == "position":
            # Clamp to recording bounds
            new_position = max(0.0, min(new_time, recording_duration))
            self.selection_state.set_marker(new_position)

            # Update visualization
            self.selection_visualizer.update_marker(new_position, ctx)
            self.draw_idle()
            return

        # Handle selection marker drag (resize)
        start = self.selection_state.selection_start
        end = self.selection_state.selection_end

        if start is None or end is None:
            return

        # Apply bounds based on which marker is being dragged
        min_selection_gap = 0.001  # Minimum 1ms selection

        if self._resize_target == "start":
            # Start marker: can't go past end marker or below 0
            new_start = max(0.0, min(new_time, end - min_selection_gap))
            self.selection_state.set_selection(new_start, end)
        elif self._resize_target == "end":
            # End marker: can't go before start marker or past recording end
            new_end = max(start + min_selection_gap, min(new_time, recording_duration))
            self.selection_state.set_selection(start, new_end)

        # Update visualization
        self.selection_visualizer.update_selection(
            self.selection_state.selection_start,
            self.selection_state.selection_end,
            ctx,
        )
        self.draw_idle()

    def _on_left_release(self, event) -> None:
        """Handle left mouse button release to finalize marker, selection, or resize."""
        # Handle resize completion
        if self._resize_active:
            self._resize_active = False
            self._resize_target = None
            self._set_cursor_default()
            self._drag_start_x = None
            return

        if self._drag_start_x is None:
            return

        # Check if this was a drag or a click
        dx = abs(event.x - self._drag_start_x)

        if dx < UIConstants.SELECTION_DRAG_THRESHOLD:
            # This was a click - set marker
            mouse_rel_x = self._get_mouse_position_in_axes(event)
            if mouse_rel_x is not None:
                time_seconds = self._pixel_to_time(mouse_rel_x)
                if time_seconds is not None:
                    self._set_marker(time_seconds)
        else:
            # This was a drag - finalize selection
            mouse_rel_x = self._get_mouse_position_in_axes(event)
            if mouse_rel_x is not None and self._drag_start_time is not None:
                end_time = self._pixel_to_time(mouse_rel_x)
                if end_time is not None:
                    self._set_selection(self._drag_start_time, end_time)

        # Clear drag state
        self._drag_start_x = None
        self._drag_start_time = None

    def _pixel_to_time(self, rel_x: float) -> Optional[float]:
        """Convert relative X position (0-1) to time in seconds.

        Args:
            rel_x: Relative position in axes (0-1)

        Returns:
            Time in seconds, or None if no recording loaded
        """
        recording_duration = self.recording_display.recording_duration
        if recording_duration <= 0:
            return None

        visible_seconds = self.zoom_controller.get_visible_seconds()
        view_start = self.zoom_controller.view_offset

        return view_start + rel_x * visible_seconds

    def _set_marker(self, time_seconds: float) -> None:
        """Set marker at the specified time position.

        Args:
            time_seconds: Position in seconds
        """
        ctx = self._view_context
        if not ctx.has_recording:
            return

        # Clamp to recording bounds
        time_seconds = max(0.0, min(time_seconds, ctx.recording_duration))

        self.selection_state.set_marker(time_seconds)
        self.selection_visualizer.update_marker(time_seconds, ctx)
        self.selection_visualizer._hide_selection()
        self.draw_idle()

    def _set_selection(self, start_time: float, end_time: float) -> None:
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

        self.selection_state.set_selection(start_time, end_time)
        self.selection_visualizer.update_selection(
            self.selection_state.selection_start,
            self.selection_state.selection_end,
            ctx,
        )
        # Hide marker when setting selection
        if self.selection_visualizer._marker_line:
            self.selection_visualizer._marker_line.set_visible(False)
        self.draw_idle()

    def clear_selection(self) -> None:
        """Clear marker and selection."""
        self.selection_state.clear_all()
        self.selection_visualizer.clear()
        self.draw_idle()
