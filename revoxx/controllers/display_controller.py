"""Display controller for managing UI updates and visualization."""

from typing import Optional, Dict, Any, TYPE_CHECKING, Callable, List
from pathlib import Path
import soundfile as sf

from ..constants import MsgType
from ..ui.widget_initializer import WidgetInitializer

if TYPE_CHECKING:
    from ..app import Revoxx
    from ..ui.window_base import WindowBase


class DisplayController:
    """Handles display updates and UI state management.

    This controller manages:
    - Display content updates
    - UI element visibility
    - Status messages
    - Mel spectrogram visualization
    - Info overlay updates
    - Multi-window coordination
    """

    def __init__(self, app: "Revoxx", window_manager):
        """Initialize the display controller.

        Args:
            app: Reference to the main application instance
            window_manager: WindowManager instance for multi-window coordination
        """
        self.app = app
        self.window_manager = window_manager
        self._spectrogram_callbacks: List[Callable] = []

    def update_display(self) -> None:
        """Update the main display with current utterance information."""
        if not self.app.state.recording.utterances:
            # No utterances loaded - show empty state
            self._for_each_window(lambda w: w.update_display(0, False, 0))
            return

        current_index = self.app.state.recording.current_index
        is_recording = self.app.state.recording.is_recording
        display_pos = self.app.navigation_controller.get_display_position(current_index)

        self._for_each_window(
            lambda w: w.update_display(current_index, is_recording, display_pos)
        )

    def show_saved_recording_when_ready(self) -> None:
        """Show saved recording when spectrogram widget is ready."""
        # Check if spectrogram exists and is ready
        if (
            self.app.window
            and hasattr(self.app.window, "mel_spectrogram")
            and self.app.window.mel_spectrogram is not None
        ):
            # Already ready, show immediately
            self.show_saved_recording()
        elif self.app.window:
            # Wait for SpectrogramReady event
            self.app.window.window.bind(
                "<<SpectrogramReady>>", lambda e: self.show_saved_recording()
            )

    def show_saved_recording(self) -> None:
        """Load and display a saved recording if it exists."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            return

        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take == 0:
            # No recording exists, clear display
            self.clear_spectrograms()
            self.update_info_panel()
            return

        # Load the recording
        filepath = self.app.file_manager.get_recording_path(current_label, current_take)
        if filepath.exists():
            try:
                audio_data, sr = self.app.file_manager.load_audio(filepath)

                # Display in all spectrograms
                self.show_recording_in_spectrograms(audio_data, sr)
                self.update_info_panel()
            except (OSError, ValueError) as e:
                # OSError for file operations, ValueError for invalid audio data
                self.set_status(f"Error loading recording: {e}", MsgType.ERROR)

    def toggle_meters(self, window_id: Optional[str] = None) -> None:
        """Toggle both mel spectrogram and level meter visualization.

        Args:
            window_id: Optional specific window to toggle, or None for all windows
        """
        if window_id:
            # Toggle specific window
            window = self._get_window_by_id(window_id)
            if window:
                current = (
                    window.meters_visible if hasattr(window, "ui_state") else False
                )
                window.set_meters_visibility(not current)
        else:
            # Toggle all windows - each toggles its own state
            self._for_each_window(
                lambda w: w.set_meters_visibility(not w.meters_visible)
            )

        self.app.audio_controller.update_audio_queue_state()

        # Trigger font recalculation for affected window
        if window_id:
            self.recalculate_window_font(window_id)
        else:
            # Main window was toggled
            self.recalculate_window_font("main")

        # Show current recording if available - but only if not currently recording/monitoring
        # Check if any window has meters visible
        any_meters_visible = any(
            w.meters_visible
            for w in self._get_active_windows()
            if hasattr(w, "meters_visible")
        )

        if any_meters_visible:
            if (
                not self.app.state.recording.is_recording
                and not self.app.audio_controller.is_monitoring
            ):
                if self.app.window:
                    # Show saved recording when spectrogram is ready
                    self.show_saved_recording_when_ready()
            elif self.app.state.recording.is_recording:
                # If we're currently recording and meters were just turned on,
                # set mel spectrogram to recording mode
                sample_rate = self.app.config.audio.sample_rate
                self._for_each_spectrogram(
                    lambda spec: (
                        spec.start_recording(sample_rate)
                        if not spec.recording_handler.is_recording
                        else None
                    )
                )

        # Save the main window's meters state as the global preference
        if self.app.window and hasattr(self.app.window, "meters_visible"):
            self.app.settings_manager.update_setting(
                "show_meters", self.app.window.meters_visible
            )

    def update_info_panel(self) -> None:
        """Update the combined info panel with current recording information."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            # No current utterance - show default parameters
            recording_params = self._get_recording_parameters()
            self._for_each_window(lambda w: w.update_info_panel(recording_params))
            return

        recording_params = self._get_recording_parameters()
        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take > 0:
            # Recording exists, get file info
            filepath = self.app.file_manager.get_recording_path(
                current_label, current_take
            )
            if filepath.exists():
                try:
                    file_info = self._get_file_info(filepath)
                    recording_params.update(file_info)
                except (OSError, ValueError):
                    # Error reading file
                    pass
        else:
            # No recordings for this utterance
            recording_params["no_recordings"] = True

        self._for_each_window(lambda w: w.update_info_panel(recording_params))

    def update_recording_timer(self, elapsed_time: float) -> None:
        """Update the recording timer display.

        Args:
            elapsed_time: Elapsed recording time in seconds
        """
        self._for_each_window(
            lambda w: (
                w.recording_timer.update(elapsed_time)
                if hasattr(w, "recording_timer") and w.recording_timer
                else None
            )
        )

    def reset_recording_timer(self) -> None:
        """Reset the recording timer display."""
        self._for_each_window(
            lambda w: (
                w.recording_timer.reset()
                if hasattr(w, "recording_timer") and w.recording_timer
                else None
            )
        )

    def update_level_meter(self, level: float) -> None:
        """Update the level meter display.

        Args:
            level: Audio level value (0.0 to 1.0)
        """
        self._for_each_level_meter(lambda meter: meter.update_level(level))

    def reset_level_meter(self) -> None:
        """Reset the level meter display."""
        self.reset_level_meters()

    def set_status(self, status: str, msg_type: MsgType = MsgType.TEMPORARY) -> None:
        """Set the status bar text.

        Args:
            status: Status text to display
            msg_type: Type of status message
        """
        self._for_each_window(lambda w: w.set_status(status, msg_type))

    def update_window_title(self, title: Optional[str] = None) -> None:
        """Update the window title.

        Args:
            title: New title text, or None for default
        """
        if self.app.window:
            if title:
                self.app.window.window.title(title)
            else:
                # Default title
                session_name = ""
                if self.app.current_session:
                    session_name = f" - {self.app.current_session.name}"
                self.app.window.window.title(f"Revoxx{session_name}")

    def _get_recording_parameters(self) -> Dict[str, Any]:
        """Get current recording parameters.

        Returns:
            Dictionary of recording parameters
        """
        return {
            "sample_rate": self.app.config.audio.sample_rate,
            "bit_depth": self.app.config.audio.bit_depth,
            "channels": self.app.config.audio.channels,
        }

    @staticmethod
    def _get_file_info(filepath: Path) -> Dict[str, Any]:
        """Get information about an audio file.

        Args:
            filepath: Path to the audio file

        Returns:
            Dictionary of file information
        """
        info = {}
        with sf.SoundFile(filepath) as f:
            info["duration"] = len(f) / f.samplerate
            info["actual_sample_rate"] = f.samplerate
            info["actual_channels"] = f.channels
            info["size"] = filepath.stat().st_size  # Changed from file_size to size

        return info

    # ============= Window Management Methods =============

    def _get_active_windows(self) -> List["WindowBase"]:
        """Get list of all active windows.

        Returns:
            List of active window instances
        """
        return self.window_manager.get_active_windows()

    def _get_window_by_id(self, window_id: str) -> Optional["WindowBase"]:
        """Get a specific window by its ID.

        Args:
            window_id: ID of the window to retrieve

        Returns:
            Window instance or None if not found
        """
        return self.window_manager.get_window(window_id)

    def _for_each_window(self, action: Callable[["WindowBase"], None]) -> None:
        """Execute action on each active window.

        Args:
            action: Function to call with each window
        """
        for window in self._get_active_windows():
            try:
                action(window)
            except AttributeError:
                # Window might not have the expected attribute
                pass

    def _for_each_spectrogram(self, action: Callable[[Any], None]) -> None:
        """Execute action on each active spectrogram widget.

        Args:
            action: Function to call with each spectrogram
        """
        for window in self._get_active_windows():
            if window.mel_spectrogram:
                try:
                    action(window.mel_spectrogram)
                except AttributeError:
                    pass

    def _for_each_level_meter(self, action: Callable[[Any], None]) -> None:
        """Execute action on each active level meter widget.

        Args:
            action: Function to call with each level meter
        """
        for window in self._get_active_windows():
            if window.embedded_level_meter:
                try:
                    action(window.embedded_level_meter)
                except AttributeError:
                    pass

    def clear_spectrograms(self) -> None:
        """Clear all spectrogram displays."""
        self._for_each_spectrogram(lambda spec: spec.clear())

    def start_spectrogram_recording(self, sample_rate: int) -> None:
        """Start recording in all spectrograms.

        Args:
            sample_rate: Sample rate for recording
        """

        def start_if_ready(spec):
            spec.clear()
            spec.start_recording(sample_rate)

        self._for_each_spectrogram(start_if_ready)

    def stop_spectrogram_recording(self) -> None:
        """Stop recording in all spectrograms."""
        self._for_each_spectrogram(lambda spec: spec.stop_recording())

    def show_recording_in_spectrograms(self, audio_data, sample_rate: int) -> None:
        """Display recording in all spectrograms.

        Args:
            audio_data: Audio data to display
            sample_rate: Sample rate of the audio
        """
        self._for_each_spectrogram(
            lambda spec: spec.show_recording(audio_data, sample_rate)
        )

    def reset_level_meters(self) -> None:
        """Reset all level meter displays."""
        self._for_each_level_meter(lambda meter: meter.reset())

    def stop_spectrogram_playback(self) -> None:
        """Stop playback in all spectrograms."""
        self._for_each_spectrogram(lambda spec: spec.stop_playback())

    def start_spectrogram_playback(self, duration: float, sample_rate: int) -> None:
        """Start playback in all spectrograms.

        Args:
            duration: Duration of the playback in seconds
            sample_rate: Sample rate of the audio
        """
        self._for_each_spectrogram(
            lambda spec: spec.start_playback(duration, sample_rate)
        )

    def update_info_panels_with_params(self, recording_params: Dict[str, Any]) -> None:
        """Update info panels in all windows with given parameters.

        Args:
            recording_params: Recording parameters to display
        """
        self._for_each_window(
            lambda window: (
                window.update_info_panel(recording_params)
                if window.info_panel_visible
                else None
            )
        )

    def set_monitoring_var(self, value: bool) -> None:
        """Set monitoring variable in main window.

        Args:
            value: True if monitoring, False otherwise
        """
        if hasattr(self.app.window, "monitoring_var"):
            self.app.window.monitoring_var.set(value)

    def is_info_panel_visible(self) -> bool:
        """Check if info panel is visible in main window.

        Returns:
            True if info panel is visible, False otherwise
        """
        return getattr(self.app.window, "info_panel_visible", False)

    # ============= Generic Window Methods =============

    def toggle_window_meters(self, window_id: str) -> Optional[bool]:
        """Toggle meters visibility in specified window.

        Args:
            window_id: ID of the window to toggle

        Returns:
            New meters state or None if window not found
        """
        results = self.window_manager.execute_on_windows(window_id, "toggle_meters")
        if results:
            self.window_manager.focus_main_window()
            return results[0]
        return None

    def toggle_window_info_panel(self, window_id: str) -> Optional[bool]:
        """Toggle info panel visibility in specified window.

        Args:
            window_id: ID of the window to toggle

        Returns:
            New info panel state or None if window not found
        """
        results = self.window_manager.execute_on_windows(window_id, "toggle_info_panel")
        if results:
            self.window_manager.focus_main_window()
            return results[0]
        return None

    def toggle_window_fullscreen(self, window_id: str) -> None:
        """Toggle fullscreen mode for specified window.

        Args:
            window_id: ID of the window to toggle
        """
        window = self._get_window_by_id(window_id)
        if window and window.is_active:
            current = window.window.attributes("-fullscreen")
            window.toggle_fullscreen()
            if self.app.settings_manager:
                self.app.settings_manager.save_window_settings(
                    window_id, {"fullscreen": not current}
                )

    def get_window_config(self, window_id: str) -> Optional[Dict[str, bool]]:
        """Get current configuration of specified window.

        Args:
            window_id: ID of the window

        Returns:
            Dictionary with config values or None if window not found
        """
        window = self._get_window_by_id(window_id)
        if window and window.is_active:
            return {
                "show_meters": window.meters_visible,
                "show_info_panel": window.info_panel_visible,
            }
        return None

    def when_spectrograms_ready(self, callback: Callable[[], None]) -> None:
        """Execute callback when all spectrograms are ready.

        This handles the case where spectrograms might still be
        initializing when we try to use them.

        Args:
            callback: Function to call when spectrograms are ready
        """
        # Collect windows that need spectrograms
        windows_needing_spectrograms = [
            w for w in self._get_active_windows() if not w.mel_spectrogram
        ]

        if not windows_needing_spectrograms:
            # All ready, execute immediately
            callback()
        else:
            # wait for all spectrograms
            spec_frames = [
                w.spec_frame
                for w in windows_needing_spectrograms
                if hasattr(w, "spec_frame")
            ]

            if spec_frames:
                WidgetInitializer.fire_when_all_ready(
                    self.app.window.window,
                    "<<AllSpectrogramsReady>>",
                    *spec_frames,
                    min_dimensions=10,
                )

                # Bind to the event and execute callback
                self.app.window.window.bind(
                    "<<AllSpectrogramsReady>>", lambda e: callback()
                )

    def toggle_fullscreen(self) -> bool:
        """Toggle fullscreen mode and return new state.

        Returns:
            True if fullscreen is now enabled, False otherwise
        """
        self.app.window.toggle_fullscreen()
        new_state = self.app.window.window.attributes("-fullscreen")
        self.app.settings_manager.update_setting("fullscreen", new_state)
        return new_state

    def toggle_info_panel(self) -> bool:
        """Toggle info panel visibility and return new state.

        Returns:
            True if info panel is now visible, False otherwise
        """
        main_window = self._get_window_by_id("main")
        if not main_window:
            return False

        # Toggle main window info panel
        new_state = main_window.toggle_info_panel()

        self.app.settings_manager.update_setting("show_info_panel", new_state)

        # Broadcast to all other windows
        for window in self._get_active_windows():
            if window.window_id != "main":
                if new_state:
                    window.info_panel.grid(
                        row=3,
                        column=0,
                        sticky="ew",
                        pady=(10, 0),
                    )
                    window.info_panel_visible = True
                else:
                    window.info_panel.grid_forget()
                    window.info_panel_visible = False

        return new_state

    def set_theme(self, theme_preset: str) -> None:
        """Set application theme.

        Args:
            theme_preset: Theme preset name (e.g., 'classic', 'modern')
        """
        self.window_manager.broadcast("set_theme", theme_preset)

    def set_level_meter_preset(self, preset: str) -> None:
        """Set level meter preset.

        Args:
            preset: Preset name (e.g., 'broadcast_ebu')
        """
        self.window_manager.broadcast("set_level_meter_preset", preset)

    def open_window(self, window_id: str) -> None:
        """Open a window with specified ID.

        Args:
            window_id: ID of the window to open
        """
        window = self.window_manager.get_window(window_id)

        if window is None:
            window = self.window_manager.create_window(window_id=window_id)

            def close_with_menu_update():
                if self.app.menu:
                    self.app.menu.on_window_closed(window_id)
                self.window_manager.close_window(window_id)

            window.window.protocol("WM_DELETE_WINDOW", close_with_menu_update)

            # Sync content when window is ready
            WidgetInitializer.when_ready(
                window.window,
                lambda: self._sync_window_content(window_id),
                min_width=100,
                min_height=100,
            )
        elif window:
            # Just bring existing window to front
            window.window.lift()
            window.window.focus_force()

    def _sync_window_content(self, window_id: str) -> None:
        """Synchronize content from main window to specified window.

        Args:
            window_id: ID of the window to sync
        """
        window = self._get_window_by_id(window_id)
        if window and window.is_active:
            self.update_display()

            if window.info_panel_visible:
                self.update_info_panel()

            # Handle spectrogram content based on current state
            if window.meters_visible:

                def sync_spectrogram_state():
                    if (
                        self.app.state.recording.is_recording
                        or self.app.audio_controller.is_monitoring
                    ):
                        sample_rate = self.app.config.audio.sample_rate
                        if window.mel_spectrogram:
                            if (
                                not window.mel_spectrogram.recording_handler.is_recording
                            ):
                                window.mel_spectrogram.clear()
                                window.mel_spectrogram.start_recording(sample_rate)
                    else:
                        self.show_saved_recording()

                self.when_spectrograms_ready(sync_spectrogram_state)

    def close_window(self, window_id: str) -> None:
        """Close specified window.

        Args:
            window_id: ID of the window to close
        """
        self.window_manager.close_window(window_id)

    def update_window_config(self, window_id: str, config: Dict[str, bool]) -> None:
        """Update window configuration.

        Args:
            window_id: ID of the window to configure
            config: Configuration dictionary with 'show_meters' and 'show_info_panel'
        """
        window = self._get_window_by_id(window_id)
        if window and window.is_active:
            window.update_configuration(
                show_meters=config.get("show_meters"),
                show_info_panel=config.get("show_info_panel"),
            )

    def recalculate_window_font(self, window_id: str) -> None:
        """Recalculate font size for a specific window after layout change.

        Args:
            window_id: ID of the window to update font for
        """
        window = self._get_window_by_id(window_id)
        if not window:
            return

        window._invalidate_layout_cache()

        if hasattr(window, "text_var") and window.text_var.get():
            window.window.after_idle(window.refresh_text_layout)
