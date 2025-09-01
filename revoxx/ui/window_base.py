"""Base window class for Revoxx UI windows.

This module provides the base functionality for all Revoxx windows,
containing common UI components and methods that can be reused
in both the main window and secondary windows.
"""

from typing import Optional
import tkinter as tk

from ..constants import UIConstants, MsgType, MsgConfig
from .font_manager import FontManager
from .emotion_indicator import EmotionLevelIndicator
from .widget_initializer import WidgetInitializer
from ..utils.text_utils import extract_emotion_level, get_max_emotion_level
from .level_meter.led_level_meter import LEDLevelMeter
from .level_meter.config import RECORDING_STANDARDS, RecordingStandard
from .spectrogram import MelSpectrogramWidget
from .themes import theme_manager
from ..utils.config import RecorderConfig
from ..utils.state import UIState, RecordingState
from ..utils.settings_manager import SettingsManager


class WindowBase:
    """Base class for Revoxx windows.

    This class provides common UI components and methods for all windows,
    including the main text display, info panels, meters, and status handling.

    Subclasses should implement window-specific features like menus and
    window management.
    """

    def __init__(
        self,
        parent: Optional[tk.Widget] = None,  # None for root window
        window_id: str = "window",
        features: Optional[dict] = None,
        config: Optional[RecorderConfig] = None,
        recording_state: Optional[RecordingState] = None,
        ui_state: Optional[UIState] = None,
        manager_dict: dict = None,
        app_callbacks: dict = None,
        settings_manager: Optional[SettingsManager] = None,
        shared_audio_state=None,
    ):
        """Initialize the base window.

        Args:
            parent: Parent widget (None for root Tk window)
            window_id: Unique identifier for this window
            features: Feature flags dictionary
            config: Application configuration with display and UI settings
            recording_state: Recording state manager tracking current utterance
            ui_state: UI state manager for window properties
            manager_dict: Shared state dictionary
            app_callbacks: Application callbacks
            settings_manager: Settings manager for persisting preferences
            shared_audio_state: Shared audio state for synchronization
        """
        if parent is None:
            self._window = tk.Tk()
            self._is_root = True
        else:
            self._window = tk.Toplevel(parent)
            self._is_root = False

        # Store window identity
        self.window_id = window_id
        self.features = features or {}
        self.is_active = True

        # Store configuration
        self.config = config
        self.recording_state = recording_state
        self.ui_state = ui_state
        self.manager_dict = manager_dict
        self.app_callbacks = app_callbacks
        self.settings_manager = settings_manager
        self.shared_audio_state = shared_audio_state
        self.meters_visible = False

        # Initialize font manager
        self.font_manager = FontManager(self.ui_state, self.config)

        # Initialize embedded level meter (will be created in _create_control_area)
        self.embedded_level_meter = None

    @property
    def window(self):
        """Get the underlying window object (Tk or Toplevel)."""
        return self._window

    def _create_info_bar(self) -> None:
        """Create the top information bar.

        Creates the status display, recording indicator, and progress
        counter. The bar has a fixed height.
        """
        height = 80  # Fixed height for top panel

        self.info_frame = tk.Frame(
            self.main_frame,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            height=height,
            highlightthickness=0,
        )
        self.info_frame.grid(
            row=0, column=0, sticky="ew", pady=(0, UIConstants.FRAME_SPACING)
        )
        self.info_frame.grid_propagate(False)

        # Status text (only shown during recording)
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            self.info_frame,
            textvariable=self.status_var,
            fg=UIConstants.COLOR_TEXT_SECONDARY,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(UIConstants.FONT_FAMILY_MONO[0], 14),
            width=40,  # Fixed width to prevent layout shifts
            anchor="w",  # Left-align text within the fixed width
        )
        self.status_label.pack(
            side=tk.LEFT,
            padx=UIConstants.FRAME_SPACING * 2,
            pady=UIConstants.FRAME_SPACING,
        )

        # Recording indicator
        self.rec_indicator = tk.Label(
            self.info_frame,
            text="● REC",
            fg=UIConstants.COLOR_TEXT_INACTIVE,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(UIConstants.FONT_FAMILY_MONO[0], 14, "bold"),
        )
        self.rec_indicator.pack(
            side=tk.RIGHT,
            padx=UIConstants.FRAME_SPACING * 2,
            pady=UIConstants.FRAME_SPACING,
        )

        # Progress info (right side)
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(
            self.info_frame,
            textvariable=self.progress_var,
            fg=UIConstants.COLOR_ACCENT,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(UIConstants.FONT_FAMILY_MONO[0], 15, "bold"),
        )
        self.progress_label.pack(
            side=tk.RIGHT,
            padx=UIConstants.MAIN_FRAME_PADDING,
            pady=UIConstants.FRAME_SPACING,
        )

        # Emotion level indicator (center - positioned absolutely)
        max_level = (
            get_max_emotion_level(self.recording_state.utterances)
            if self.recording_state.utterances
            else 5
        )
        self.emotion_indicator = EmotionLevelIndicator(
            self.info_frame,
            font_manager=self.font_manager,
            max_level=max_level,
            bg_color=UIConstants.COLOR_BACKGROUND_SECONDARY,
        )
        # Use place() for absolute centering, independent of other widgets
        self.emotion_indicator.place(relx=0.5, rely=0.5, anchor="center")

    def _create_utterance_display(self) -> None:
        """Create the middle area for utterance text display.

        This shows only the utterance text in large font, centered.
        The label/filename info is now in the info panel.
        """
        self.utterance_frame = tk.Frame(
            self.main_frame,
            bg=UIConstants.COLOR_BACKGROUND,
            highlightthickness=0,
        )
        # Grid in row 1 with weight=1 to expand
        self.utterance_frame.grid(
            row=1, column=0, sticky="nsew", pady=UIConstants.FRAME_SPACING
        )

        # Main text display only (no label)
        self.text_var = tk.StringVar()
        self.text_display = tk.Label(
            self.utterance_frame,
            textvariable=self.text_var,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            bg=UIConstants.COLOR_BACKGROUND,
            anchor="center",
            justify="center",
            font=(UIConstants.FONT_FAMILY_SANS[0], 32, "normal"),
            wraplength=800,  # Will be updated dynamically
        )
        # Configure utterance_frame grid
        self.utterance_frame.grid_rowconfigure(0, weight=1)
        self.utterance_frame.grid_columnconfigure(0, weight=1)
        # Place text_display in grid
        self.text_display.grid(
            row=0, column=0, sticky="nsew", padx=UIConstants.MAIN_FRAME_PADDING
        )

    def _create_spectrogram_area(self) -> None:
        """Create the bottom area for spectrogram and level meter.

        This is now a separate panel, independent from the info panel.
        """
        height = 270  # Fixed pixel height

        self.spec_container = tk.Frame(
            self.main_frame,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            height=height,
            highlightthickness=0,
        )
        # Configure grid layout for spec_container children
        self.spec_container.grid_columnconfigure(0, weight=1)  # spec_frame expands
        self.spec_container.grid_columnconfigure(1, weight=0)  # level_meter fixed width
        self.spec_container.grid_rowconfigure(0, weight=1)
        self.spec_container.grid_propagate(False)  # Critical: prevent size changes

        # Always create child widgets
        self._create_spectrogram_widget()
        self._create_embedded_level_meter()

        # Check if meters should be visible from settings
        self.meters_visible = getattr(
            self.settings_manager.settings, "show_meters", True
        )

        # Only grid the container if meters should be visible
        if self.meters_visible:
            self.spec_container.grid(
                row=2, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
            )

    def _create_spectrogram_widget(self) -> None:
        """Create the mel spectrogram widget.

        Creates a frame and initializes the MelSpectrogramWidget for
        real-time audio visualization.
        """
        self.spec_frame = tk.Frame(
            self.spec_container,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            highlightthickness=0,
        )
        self.spec_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(UIConstants.FRAME_SPACING, UIConstants.FRAME_SPACING // 2),
            pady=UIConstants.FRAME_SPACING,
        )

        # Create mel_spectrogram widget when window has real dimensions
        # This avoids the negative figure size error
        self.mel_spectrogram = None
        self._setup_mel_spectrogram_creation()

    def _setup_mel_spectrogram_creation(self) -> None:
        """Setup event-based mel spectrogram creation."""

        def create_and_notify():
            if self.mel_spectrogram is None:
                self._create_mel_spectrogram_widget()
                self.window.event_generate("<<SpectrogramReady>>")
                if self.embedded_level_meter:
                    self.window.event_generate("<<AllMetersReady>>")

        WidgetInitializer.when_ready(
            self.spec_frame, create_and_notify, min_width=10, min_height=10
        )

    def _create_mel_spectrogram_widget(self) -> None:
        """Create the mel spectrogram widget."""
        if not hasattr(self, "spec_frame") or self.mel_spectrogram is not None:
            return

        self.mel_spectrogram = MelSpectrogramWidget(
            self.spec_frame,
            self.config.audio,
            self.config.display,
            self.manager_dict,
            self.shared_audio_state,
        )

    def _create_combined_info_panel(self) -> None:
        """Create the combined info panel as a separate panel.

        This is now an independent panel between the utterance display and spectrogram.
        """
        # Check if info panel should be visible
        self.info_panel_visible = getattr(
            self.settings_manager.settings, "show_info_panel", True
        )

        # Create info panel frame directly in main_frame
        self.info_panel = tk.Frame(
            self.main_frame,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            height=60,  # Single line layout like top panel
            highlightthickness=0,
        )

        # Only grid if visible
        if self.info_panel_visible:
            self.info_panel.grid(
                row=3, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
            )

        self.info_panel.pack_propagate(False)

        # Get mono font from FontManager - ensures consistency with top panel
        mono_font = self.font_manager.get_mono_font()

        # Font size matching top panel style
        font_size = 24

        # Left side - File information (label/filename, size)
        self.file_info_var = tk.StringVar(value="")
        self.file_info_label = tk.Label(
            self.info_panel,
            textvariable=self.file_info_var,
            fg=UIConstants.COLOR_TEXT_SECONDARY,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(mono_font, font_size),
        )
        self.file_info_label.pack(
            side=tk.LEFT,
            padx=UIConstants.FRAME_SPACING * 2,
            pady=UIConstants.FRAME_SPACING,
        )

        # Center - Audio format (sample rate, bit depth, format)
        self.audio_format_var = tk.StringVar(value="48000 Hz, 24 bit FLAC mono")
        self.audio_format_label = tk.Label(
            self.info_panel,
            textvariable=self.audio_format_var,
            fg=UIConstants.COLOR_TEXT_SECONDARY,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(mono_font, font_size),
        )
        self.audio_format_label.pack(
            side=tk.LEFT,
            expand=True,
            padx=UIConstants.FRAME_SPACING * 2,
            pady=UIConstants.FRAME_SPACING,
        )

        # Right side - Duration in brackets
        self.duration_var = tk.StringVar(value="")
        self.duration_label = tk.Label(
            self.info_panel,
            textvariable=self.duration_var,
            fg=UIConstants.COLOR_TEXT_SECONDARY,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(mono_font, font_size),
        )
        self.duration_label.pack(
            side=tk.RIGHT,
            padx=UIConstants.MAIN_FRAME_PADDING,
            pady=UIConstants.FRAME_SPACING,
        )

        # Center message label (for "No recordings" etc.)
        self.info_center_message = tk.Label(
            self.info_panel,
            text="",
            fg=UIConstants.COLOR_TEXT_SECONDARY,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            font=(mono_font, font_size + 2, "bold"),
        )
        # Don't pack initially - will be shown/hidden as needed

    def _create_embedded_level_meter(self) -> None:
        """Create the embedded LED level meter."""
        # Create frame for level meter
        self.level_meter_frame = tk.Frame(
            self.spec_container,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,  # Match container background
            width=130,  # Increased width for level meter for better readability
            highlightthickness=0,
        )

        # Create LED level meter
        if self.shared_audio_state:
            self.embedded_level_meter = LEDLevelMeter(
                self.level_meter_frame, self.shared_audio_state
            )
            self.embedded_level_meter.pack(fill=tk.BOTH, expand=True)

            # Always grid the level meter frame within container
            # Container visibility controls whether it's shown
            self.level_meter_frame.grid(
                row=0,
                column=1,
                sticky="nsew",
                padx=(UIConstants.FRAME_SPACING // 2, UIConstants.FRAME_SPACING),
                pady=UIConstants.FRAME_SPACING,
            )
            self.level_meter_frame.grid_propagate(False)

            # Apply saved preset from settings after ensuring widget is ready
            preset_str = getattr(
                self.settings_manager.settings, "level_meter_preset", "broadcast_ebu"
            )
            self._apply_level_meter_preset_when_ready(preset_str)
        else:
            self.embedded_level_meter = None

    def _apply_level_meter_preset_when_ready(self, preset_str: str) -> None:
        """Apply level meter preset after ensuring widget is ready."""
        # Widget is now always in grid, just apply preset when ready
        WidgetInitializer.when_ready(
            self.level_meter_frame,
            lambda: self.set_level_meter_preset(preset_str),
            min_width=10,
            min_height=10,
        )

    def update_display(
        self, index: int, is_recording: bool, display_position: int
    ) -> None:
        """Update the display with current utterance.

        Updates the main text display, label, progress counter, and
        recording indicator based on current state.

        Args:
            index: Current utterance index
            is_recording: Whether currently recording
            display_position: Display position for progress counter (1-based)
        """
        if 0 <= index < len(self.recording_state.utterances):
            self._update_utterance_display(index)
            self._update_progress_display(display_position)

        self._update_recording_indicators(is_recording)

    def _update_utterance_display(self, index: int) -> None:
        """Update the utterance text display and emotion indicator.

        Extracts emotion level from utterance text and displays
        clean text without emotion label.

        Args:
            index: Index of current utterance
        """
        full_text = self.recording_state.utterances[index]

        # Extract emotion level and clean text
        emotion_level, clean_text = extract_emotion_level(full_text)

        self.update_utterance_text(clean_text)

        # Update emotion indicator
        if emotion_level is not None:
            self.emotion_indicator.set_level(emotion_level)
        else:
            self.emotion_indicator.set_level(0)  # No emotion level

    def _update_progress_display(self, display_position: int) -> None:
        """Update the progress counter display.

        Args:
            display_position: Current position (1-based) for user display
        """
        total = len(self.recording_state.utterances)
        self.progress_var.set(f"{display_position}/{total}")

    def _update_recording_indicators(self, is_recording: bool) -> None:
        """Update recording state indicators (colors and status).

        Args:
            is_recording: Whether currently recording
        """
        if is_recording:
            self.text_display.config(fg=UIConstants.COLOR_TEXT_RECORDING)
            self.rec_indicator.config(fg=UIConstants.COLOR_TEXT_RECORDING)
            self.set_status("RECORDING...", MsgType.ACTIVE)
        else:
            self.text_display.config(fg=UIConstants.COLOR_TEXT_NORMAL)
            self.rec_indicator.config(fg=UIConstants.COLOR_TEXT_INACTIVE)
            self.set_status("", MsgType.DEFAULT)

    def set_status(
        self,
        message: str,
        msg_type: MsgType = MsgType.TEMPORARY,
        duration_ms: int = None,
    ) -> None:
        """Set status message with automatic clearing behavior based on message type.

        Message types determine the lifecycle:

        DEFAULT (static info):
            - Shows current utterance label and take count
            - Example: "utterance_001 - Take 2/3"
            - This is what's always shown when nothing else is happening

        TEMPORARY (brief notifications):
            - Auto-clears after 3 seconds (or custom duration)
            - Then automatically returns to DEFAULT status
            - Example: "Recording saved", "Meters hidden"

        ACTIVE (ongoing operations):
            - Stays visible as long as the operation is running
            - Must be manually cleared when operation ends
            - Example: "RECORDING...", "Monitoring input levels..."

        ERROR (persistent problems):
            - Stays until manually cleared or successful action occurs
            - Example: "Error loading session: file not found"

        Args:
            message: Status text to display in the info bar
            msg_type: Type of message determining auto-clear behavior
            duration_ms: Optional custom duration for TEMPORARY messages (default: 3000ms)
        """
        # Cancel any existing timer
        if hasattr(self, "_status_timer") and self._status_timer is not None:
            try:
                self.window.after_cancel(self._status_timer)
            except ValueError:
                # Timer ID was invalid, ignore - might happen at init time
                pass
            self._status_timer = None

        # Store current message type
        self._current_msg_type = msg_type

        # Handle different message types
        if msg_type == MsgType.DEFAULT:
            # For DEFAULT type, update directly with utterance info (don't set message)
            self._update_default_status()
        else:
            # For all other types, set the message
            self.status_var.set(message)

        # Schedule auto-clear for temporary messages
        if msg_type == MsgType.TEMPORARY:
            duration = duration_ms or MsgConfig.DEFAULT_TEMPORARY_DURATION_MS
            self._status_timer = self.window.after(duration, self._clear_status)

    def _clear_status(self) -> None:
        """Clear status and return to default (utterance info)."""
        self._status_timer = None
        self._current_msg_type = MsgType.DEFAULT
        self._update_default_status()

    def _update_default_status(self) -> None:
        """Update status with default utterance/take information."""
        # Get current utterance label
        current_label = getattr(self.recording_state, "current_label", None)
        if not current_label:
            self.status_var.set("")
            return

        # Check if we have takes
        current_take = (
            self.recording_state.get_current_take(current_label)
            if hasattr(self.recording_state, "get_current_take")
            else 0
        )
        total_takes = (
            self.recording_state.get_take_count(current_label)
            if hasattr(self.recording_state, "get_take_count")
            else 0
        )

        if total_takes > 0 and current_take > 0:
            self.status_var.set(f"{current_label} - Take {current_take}/{total_takes}")
        else:
            # Just show label
            self.status_var.set(current_label)

    def update_label_with_filename(self, label: str, filename: str = None) -> None:
        """Update the label display with optional filename.

        Args:
            label: The utterance label
            filename: Optional filename to display (e.g., "take_001.flac")
        """
        # Store for use in update_info_panel
        self._current_label = label
        self._current_filename = filename

    def update_info_panel(self, recording_params: dict = None) -> None:
        """Update the combined info panel with recording information.

        Args:
            recording_params: Dict with recording parameters (sample_rate, bit_depth, channels,
                            format, duration, size, no_recordings)
        """
        if not recording_params:
            return

        # Always hide center message and show labels
        self.info_center_message.place_forget()

        # Pack labels if not visible
        if not self.file_info_label.winfo_viewable():
            self.file_info_label.pack(
                side=tk.LEFT,
                padx=UIConstants.FRAME_SPACING * 2,
                pady=UIConstants.FRAME_SPACING,
            )
            self.audio_format_label.pack(
                side=tk.LEFT,
                expand=True,
                padx=UIConstants.FRAME_SPACING * 2,
                pady=UIConstants.FRAME_SPACING,
            )
            self.duration_label.pack(
                side=tk.RIGHT,
                padx=UIConstants.MAIN_FRAME_PADDING,
                pady=UIConstants.FRAME_SPACING,
            )

        # Handle no recordings case - show audio settings instead
        if recording_params.get("no_recordings", False):
            self.file_info_var.set("")
            self.duration_var.set("")

            # Get audio settings from settings manager
            sample_rate = getattr(self.settings_manager.settings, "sample_rate", 48000)
            bit_depth = getattr(self.settings_manager.settings, "bit_depth", 24)

            # Display current audio settings
            self.audio_format_var.set(f"{sample_rate} Hz, {bit_depth} bit FLAC mono")
            return  # Exit early

        # Get label and filename from previous call to update_label_with_filename
        label = getattr(self, "_current_label", "")
        filename = getattr(self, "_current_filename", "")

        # Update left side - file info with size
        size_bytes = recording_params.get("size", 0)

        file_info_parts = []
        if label and filename:
            file_info_parts.append(f"{label}/{filename}")
        elif label:
            file_info_parts.append(label)

        # Add size
        if size_bytes > 0:
            if size_bytes < 1024 * 1024:
                size_text = f"{size_bytes / 1024:.1f} KB"
            else:
                size_text = f"{size_bytes / (1024 * 1024):.1f} MB"
            file_info_parts.append(size_text)

        self.file_info_var.set(", ".join(file_info_parts) if file_info_parts else "")

        # Update center - audio format
        sample_rate = recording_params.get("sample_rate", 48000)
        bit_depth = recording_params.get("bit_depth", 24)
        format_name = recording_params.get("format", "FLAC")
        channels = recording_params.get("channels", 1)
        channel_text = "mono" if channels == 1 else "stereo"
        self.audio_format_var.set(
            f"{sample_rate} Hz, {bit_depth} bit {format_name} {channel_text}"
        )

        # Update right side - duration
        duration = recording_params.get("duration", 0)
        if duration > 0:
            self.duration_var.set(f"{duration:.2f} seconds")
        else:
            self.duration_var.set("")

    def update_utterance_text(self, text: str) -> None:
        """Update utterance display with new text and optimal font size.

        This is the main public API for setting utterance text.
        Calculates optimal font size and applies both text and font in one operation.

        Args:
            text: The text to display
        """
        if not text:
            self.text_var.set("")
            return

        # Get current dimensions (cached or fresh)
        width, height = self._get_cached_dimensions()

        # Skip if frame hasn't been sized yet
        if width <= 0 or height <= 0:
            # Just set the text with default font
            self.text_var.set(text)
            return

        optimal_size, wrap_length = self._calculate_optimal_font_size(
            text, width, height
        )
        self._apply_text_font(optimal_size, wrap_length)
        self.text_var.set(text)

    def _invalidate_layout_cache(self) -> None:
        """Invalidate the layout dimensions cache.

        Call this when layout changes occur (meters toggle, info panel toggle, etc.)
        """
        if hasattr(self, "_cached_frame_dims"):
            delattr(self, "_cached_frame_dims")

    def _get_cached_dimensions(self) -> tuple[int, int]:
        """Get cached frame dimensions or calculate if needed.

        Returns:
            Tuple of (width, height) available for text display
        """
        # Check if recalculation needed
        should_recalculate = (
            not hasattr(self, "_cached_frame_dims")
            or self.meters_visible != getattr(self, "_last_meters_state", None)
            or self.info_panel_visible != getattr(self, "_last_info_panel_state", None)
        )

        if should_recalculate:
            self._update_layout_cache()

        return getattr(self, "_cached_width", 0), getattr(self, "_cached_height", 0)

    def _update_layout_cache(self) -> None:
        """Update cached frame dimensions."""
        # Only force update if dimensions changed
        if not hasattr(self, "_cached_frame_dims"):
            self.window.update_idletasks()

        self._cached_width = self.utterance_frame.winfo_width() - (
            2 * UIConstants.MAIN_FRAME_PADDING
        )
        self._cached_height = self.utterance_frame.winfo_height() - (
            2 * UIConstants.MAIN_FRAME_PADDING
        )
        self._cached_frame_dims = True
        self._last_meters_state = self.meters_visible
        self._last_info_panel_state = self.info_panel_visible

    def refresh_text_layout(self) -> None:
        """Refresh the text layout with recalculated font size.

        This is the main public API for updating font size of existing text.
        Use this after window resize, panel toggles, etc.
        """
        text = self.text_var.get()
        if not text:
            return

        # Get current dimensions
        width, height = self._get_cached_dimensions()

        # Skip if frame hasn't been sized yet
        if width <= 0 or height <= 0:
            return

        # Calculate and apply new font size
        optimal_size, wrap_length = self._calculate_optimal_font_size(
            text, width, height
        )
        self._apply_text_font(optimal_size, wrap_length)

    def _calculate_optimal_font_size(
        self, text: str, width: int, height: int
    ) -> tuple[int, int]:
        """Calculate optimal font size for given text and dimensions.

        Args:
            text: Text to size
            width: Available width
            height: Available height

        Returns:
            Tuple of (font_size, wrap_length)
        """
        return self.font_manager.calculate_adaptive_font_size(
            text,
            width,
            height,
            max_font_size=UIConstants.FONT_SIZE_LARGE,
            min_font_size=UIConstants.MIN_FONT_SIZE_SMALL,
            use_mono_font=False,
        )

    def _apply_text_font(self, font_size: int, wrap_length: int) -> None:
        """Apply font configuration to text display.

        Args:
            font_size: Font size in points
            wrap_length: Text wrap length in pixels
        """
        sans_font = self.font_manager.get_sans_font()
        self.text_display.config(
            font=(sans_font, font_size, "normal"), wraplength=wrap_length
        )

    def set_meters_visibility(self, visible: bool) -> None:
        """Set visibility for both mel spectrogram and level meter together.

        Args:
            visible: True to show both meters, False to hide both
        """
        # Hide/show the entire container
        if visible:
            # Show container
            self.spec_container.grid(
                row=2, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
            )
            # Keep fixed size
            self.spec_container.grid_propagate(False)
            # Now show the child frames
            self._set_spectrogram_visibility(visible)
            self._set_level_meter_visibility(visible)
        else:
            # Hide the entire container so the text area can expand
            self.spec_container.grid_forget()
            # Also hide child frames (in case they're referenced elsewhere)
            if hasattr(self, "spec_frame"):
                self.spec_frame.grid_forget()
            if hasattr(self, "level_meter_frame"):
                self.level_meter_frame.grid_forget()

        self.meters_visible = visible

        # Update menu checkbox if available (in RootWindow)
        if hasattr(self, "meters_var"):
            self.meters_var.set(visible)

    def _set_spectrogram_visibility(self, visible: bool) -> None:
        """Set spectrogram visibility.

        Args:
            visible: True to show, False to hide
        """
        if visible:
            self.spec_frame.grid(
                row=0,
                column=0,
                sticky="nsew",
                padx=(UIConstants.FRAME_SPACING, UIConstants.FRAME_SPACING // 2),
                pady=UIConstants.FRAME_SPACING,
            )
            # Create widget if not yet created (happens when meters were hidden at startup)
            if self.mel_spectrogram is None:
                self.window.update_idletasks()  # Update frame size after grid
                self._create_mel_spectrogram_widget()
                # Fire event after creation
                self.window.event_generate("<<SpectrogramReady>>")
            else:
                self.window.update_idletasks()
                self.mel_spectrogram.canvas.draw_idle()
        else:
            self.spec_frame.grid_forget()
        self.meters_visible = visible

    def _set_level_meter_visibility(self, visible: bool) -> None:
        """Set level meter visibility.

        Args:
            visible: True to show, False to hide
        """
        # Create level meter if showing for the first time
        if visible and self.embedded_level_meter is None:
            self._create_embedded_level_meter()

        if visible:
            # First configure frame with background
            self.level_meter_frame.configure(bg=UIConstants.COLOR_BACKGROUND_SECONDARY)
            self.level_meter_frame.update_idletasks()

            # Now grid it
            self.level_meter_frame.grid(
                row=0,
                column=1,
                sticky="ns",
                padx=(UIConstants.FRAME_SPACING // 2, UIConstants.FRAME_SPACING),
                pady=UIConstants.FRAME_SPACING,
            )
            self.level_meter_frame.grid_propagate(False)
            if self.embedded_level_meter:
                self.embedded_level_meter.refresh()
        else:
            self.level_meter_frame.grid_forget()

    def set_level_meter_preset(self, preset_name: str) -> None:
        """Set level meter configuration based on recording preset.

        Args:
            preset_name: Name of the recording preset (e.g., 'broadcast_ebu')
        """
        # Find the matching standard enum
        standard_enum = None
        for std in RecordingStandard:
            if std.value == preset_name:
                standard_enum = std
                break

        if not standard_enum or standard_enum not in RECORDING_STANDARDS:
            print(f"Unknown recording preset: {preset_name}")
            return

        # Get the configuration for this preset
        config = RECORDING_STANDARDS[standard_enum]

        # Apply to embedded level meter if it exists
        if hasattr(self, "embedded_level_meter") and self.embedded_level_meter:
            # Reset via shared state so producer/consumer are in Sync
            try:
                self.shared_audio_state.reset_level_meter()
            except Exception:
                pass
            self.embedded_level_meter.set_config(config)

        # Save the setting
        self.settings_manager.update_setting("level_meter_preset", preset_name)

        # Update the menu variable if available (in RootWindow)
        if hasattr(self, "level_meter_preset_var"):
            self.level_meter_preset_var.set(preset_name)

    def _update_fixed_ui_fonts(self) -> None:
        """Update fonts for fixed UI elements (status, indicators, etc.).

        This only updates non-content UI elements with fixed font sizes.
        Utterance text font sizing is handled separately.
        """
        # Get font families from manager
        mono_font = self.font_manager.get_mono_font()

        # Fixed font sizes for top panel
        self.status_label.config(font=(mono_font, 24))
        self.rec_indicator.config(font=(mono_font, 24, "bold"))
        self.progress_label.config(font=(mono_font, 24, "bold"))

        # Note: Text font sizing is handled by update_utterance_text() and refresh_text_layout()

    def _apply_theme_to_widgets(self) -> None:
        """Apply current theme colors to all widgets."""
        self.window.configure(bg=UIConstants.COLOR_BACKGROUND)

        if hasattr(self, "main_frame"):
            self.main_frame.configure(bg=UIConstants.COLOR_BACKGROUND)

        if hasattr(self, "info_frame"):
            self.info_frame.configure(bg=UIConstants.COLOR_BACKGROUND_SECONDARY)
            self.status_label.configure(
                fg=UIConstants.COLOR_TEXT_SECONDARY,
                bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            )
            self.rec_indicator.configure(
                fg=UIConstants.COLOR_TEXT_INACTIVE,
                bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            )
            self.progress_label.configure(
                fg=UIConstants.COLOR_ACCENT, bg=UIConstants.COLOR_BACKGROUND_SECONDARY
            )

        # Update utterance frame
        if hasattr(self, "utterance_frame"):
            self.utterance_frame.configure(bg=UIConstants.COLOR_BACKGROUND)
            self.text_display.configure(
                fg=UIConstants.COLOR_TEXT_NORMAL, bg=UIConstants.COLOR_BACKGROUND
            )

        # Update spec container
        if hasattr(self, "spec_container"):
            self.spec_container.configure(bg=UIConstants.COLOR_BACKGROUND_SECONDARY)

        # Update info panel
        if hasattr(self, "info_panel"):
            self.info_panel.configure(bg=UIConstants.COLOR_BACKGROUND_SECONDARY)
            self.file_info_label.configure(
                fg=UIConstants.COLOR_TEXT_SECONDARY,
                bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            )
            self.audio_format_label.configure(
                fg=UIConstants.COLOR_TEXT_SECONDARY,
                bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            )
            self.duration_label.configure(
                fg=UIConstants.COLOR_TEXT_SECONDARY,
                bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            )
            if hasattr(self, "info_center_message"):
                self.info_center_message.configure(
                    fg=UIConstants.COLOR_TEXT_SECONDARY,
                    bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
                )

        # Update level meter frame
        if hasattr(self, "level_meter_frame"):
            self.level_meter_frame.configure(bg=UIConstants.COLOR_BACKGROUND_SECONDARY)
            # Level meter needs refresh for new theme colors
            if hasattr(self, "embedded_level_meter") and self.embedded_level_meter:
                self.embedded_level_meter.refresh()

        # Update emotion indicator with new theme colors
        if hasattr(self, "emotion_indicator"):
            self.emotion_indicator.refresh_colors()

    def _force_redraw(self) -> None:
        """Force redraw of all components."""
        self.font_manager.clear_font_cache()

        # Update spectrogram if visible
        if hasattr(self, "mel_spectrogram") and self.mel_spectrogram:
            # Update matplotlib figure colors
            self.mel_spectrogram.fig.set_facecolor(UIConstants.COLOR_BACKGROUND)
            self.mel_spectrogram.ax.set_facecolor(UIConstants.COLOR_BACKGROUND)

            # Update axes colors
            self.mel_spectrogram.ax.tick_params(colors=UIConstants.COLOR_TEXT_SECONDARY)
            self.mel_spectrogram.ax.yaxis.label.set_color(
                UIConstants.COLOR_TEXT_SECONDARY
            )

            # Update spines
            for spine in self.mel_spectrogram.ax.spines.values():
                if spine.get_visible():
                    spine.set_color(UIConstants.COLOR_BORDER)

            # Update colormap
            if self.mel_spectrogram.im:
                self.mel_spectrogram.im.set_cmap(theme_manager.colormap)

            # Update NO DATA text if visible
            if hasattr(self.mel_spectrogram, "no_data_text"):
                self.mel_spectrogram.no_data_text.set_color(
                    UIConstants.COLOR_TEXT_SECONDARY
                )

            # Redraw canvas
            self.mel_spectrogram.canvas.draw()

        # Force Tkinter update
        self.window.update_idletasks()

    def toggle_meters(self) -> bool:
        """Toggle meters visibility.

        Returns:
            New meters visibility state
        """
        current = self.meters_visible
        self.set_meters_visibility(not current)

        self._invalidate_layout_cache()

        if self.text_var.get():
            self.window.after_idle(self.refresh_text_layout)

        return not current

    def toggle_info_panel(self) -> bool:
        """Toggle info panel visibility.

        Returns:
            New info panel visibility state
        """
        if hasattr(self, "info_panel"):
            if self.info_panel.winfo_viewable():
                self.info_panel.grid_forget()
                self.info_panel_visible = False
                new_state = False
            else:
                self.info_panel.grid(
                    row=3, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
                )
                self.info_panel_visible = True
                new_state = True

            # Invalidate cache to trigger recalculation on next update
            self._invalidate_layout_cache()

            # Schedule font recalculation after layout stabilizes
            if self.text_var.get():
                self.window.after_idle(self.refresh_text_layout)

            return new_state
        return False

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode for this window."""
        current = self.window.attributes("-fullscreen")
        self.window.attributes("-fullscreen", not current)

        # Save state if settings manager available
        if self.settings_manager and self.features.get("remember_session"):
            key = f"{self.window_id}_fullscreen"
            self.settings_manager.update_setting(key, not current)

    def update_configuration(
        self, show_meters: bool = None, show_info_panel: bool = None
    ) -> None:
        """Update window configuration.

        Args:
            show_meters: Whether to show meters (None = no change)
            show_info_panel: Whether to show info panel (None = no change)
        """
        if show_meters is not None:
            self.set_meters_visibility(show_meters)

        if show_info_panel is not None:
            if show_info_panel and hasattr(self, "info_panel"):
                from ..constants import UIConstants

                self.info_panel.grid(
                    row=3, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
                )
                self.info_panel_visible = True
            elif not show_info_panel and hasattr(self, "info_panel"):
                self.info_panel.grid_forget()
                self.info_panel_visible = False

    def focus_window(self) -> None:
        """Bring window to front and focus."""
        self.window.lift()
        self.window.focus_force()

    def set_theme(self, theme_preset: str) -> None:
        """Set the application theme.

        Args:
            theme_preset: Theme preset name (e.g., 'classic', 'modern')
        """
        from .themes import ThemePreset

        try:
            preset = ThemePreset(theme_preset)
            theme_manager.set_theme(preset)
            from ..constants import UIConstants

            UIConstants.refresh()

            # Save if settings manager available
            if self.settings_manager:
                self.settings_manager.update_setting("theme", theme_preset)

            # Apply theme to widgets
            self._apply_theme_to_widgets()
            self._force_redraw()
        except ValueError:
            print(f"Unknown theme preset: {theme_preset}")
