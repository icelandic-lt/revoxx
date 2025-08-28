"""Display controller for managing UI updates and visualization."""

from typing import Optional, Dict, Any, TYPE_CHECKING, Callable, List
from pathlib import Path
import soundfile as sf

from ..constants import MsgType

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

    def __init__(self, app: "Revoxx"):
        """Initialize the display controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self.window_manager = None  # Will be set by app after window_manager is created
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
                    window.ui_state.meters_visible
                    if hasattr(window, "ui_state")
                    else False
                )
                window.set_meters_visibility(not current)
        else:
            # Toggle all windows
            self.app.state.ui.meters_visible = not self.app.state.ui.meters_visible
            self._for_each_window(
                lambda w: w.set_meters_visibility(self.app.state.ui.meters_visible)
            )

        self.app.audio_controller.update_audio_queue_state()

        # Trigger font recalculation for affected window
        if window_id:
            self.recalculate_window_font(window_id)
        else:
            # Main window was toggled
            self.recalculate_window_font("main")

        # Show current recording if available - but only if not currently recording/monitoring
        if self.app.state.ui.meters_visible:
            if (
                not self.app.state.recording.is_recording
                and not self.app.audio_controller.is_monitoring
            ):
                if self.app.window:
                    self.app.window.window.after(50, self.show_saved_recording)
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

        self.app.settings_manager.update_setting(
            "show_meters", self.app.state.ui.meters_visible
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
        # Use WindowManager if available
        if self.window_manager:
            return self.window_manager.get_active_windows()

        # Fallback to direct window references
        windows = []
        if self.app.window:
            windows.append(self.app.window)
        if (
            hasattr(self.app, "second_window")
            and self.app.second_window
            and self.app.second_window.is_active
        ):
            windows.append(self.app.second_window)
        return windows

    def _get_window_by_id(self, window_id: str) -> Optional["WindowBase"]:
        """Get a specific window by its ID.

        Args:
            window_id: ID of the window to retrieve

        Returns:
            Window instance or None if not found
        """
        if self.window_manager:
            return self.window_manager.get_window(window_id)

        # Fallback to direct references
        if window_id == "main" and self.app.window:
            return self.app.window
        elif window_id == "second" and hasattr(self.app, "second_window"):
            return self.app.second_window
        return None

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

    # ============= Second Window Specific Methods =============

    def toggle_second_window_meters(self) -> Optional[bool]:
        """Toggle meters visibility in second window.

        Returns:
            New meters state or None if no second window
        """
        second = self._get_window_by_id("second")
        if second and second.is_active:
            result = second.toggle_meters()
            # Return focus to main window after action
            self.window_manager.focus_main_window()
            return result
        return None

    def toggle_second_window_info_panel(self) -> Optional[bool]:
        """Toggle info panel visibility in second window.

        Returns:
            New info panel state or None if no second window
        """
        second = self._get_window_by_id("second")
        if second and second.is_active:
            result = second.toggle_info_panel()
            # Return focus to main window after action
            self.window_manager.focus_main_window()
            return result
        return None

    def toggle_second_window_fullscreen(self) -> None:
        """Toggle fullscreen mode for second window."""
        second = self._get_window_by_id("second")
        if second and second.is_active:
            current = second.window.attributes("-fullscreen")
            # Toggle fullscreen
            second.toggle_fullscreen()
            # Save the new state
            if self.app.settings_manager:
                self.app.settings_manager.update_setting(
                    "second_fullscreen", not current
                )

    def get_second_window_config(self) -> Optional[Dict[str, bool]]:
        """Get current configuration of second window.

        Returns:
            Dictionary with config values or None if no second window
        """
        second = self._get_window_by_id("second")
        if second and second.is_active:
            return {
                "show_meters": second.ui_state.meters_visible,
                "show_info_panel": second.info_panel_visible,
            }
        return None

    def execute_on_second_window(self, action: str, *args, **kwargs) -> Any:
        """Execute an action on the second window if active.

        Generic method for less common operations.

        Args:
            action: Method name to call on second window
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method call or None if no second window
        """
        second = self._get_window_by_id("second")
        if second and second.is_active:
            method = getattr(second, action, None)
            if method and callable(method):
                return method(*args, **kwargs)
        return None

    def when_spectrograms_ready(self, callback: Callable[[], None]) -> None:
        """Execute callback when all spectrograms are ready.

        This handles the case where spectrograms might still be
        initializing when we try to use them.

        Args:
            callback: Function to call when spectrograms are ready
        """
        all_ready = True
        for window in self._get_active_windows():
            if not window.mel_spectrogram:
                all_ready = False
                break

        if all_ready:
            # Execute immediately
            callback()
        else:
            # Store callback and wait
            self._spectrogram_callbacks.append(callback)
            self.app.window.window.after(50, self._check_spectrograms_ready)

    def _check_spectrograms_ready(self) -> None:
        """Check if spectrograms are ready and execute callbacks."""
        all_ready = True
        for window in self._get_active_windows():
            if not window.mel_spectrogram:
                all_ready = False
                break

        if all_ready:
            # Execute all pending callbacks
            callbacks = self._spectrogram_callbacks.copy()
            self._spectrogram_callbacks.clear()
            for callback in callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in spectrogram ready callback: {e}")
        else:
            self.app.window.window.after(50, self._check_spectrograms_ready)

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

        second = self._get_window_by_id("second")
        if second and second.is_active:
            if new_state:
                second.info_panel.grid(
                    row=3,
                    column=0,
                    sticky="ew",
                    pady=(10, 0),
                )
                second.info_panel_visible = True
            else:
                second.info_panel.grid_forget()
                second.info_panel_visible = False

        return new_state

    def set_theme(self, theme_preset: str) -> None:
        """Set application theme.

        Args:
            theme_preset: Theme preset name (e.g., 'classic', 'modern')
        """
        self.app.window.set_theme(theme_preset)

        second = self._get_window_by_id("second")
        if second and second.is_active:
            second.set_theme(theme_preset)

    def set_level_meter_preset(self, preset: str) -> None:
        """Set level meter preset.

        Args:
            preset: Preset name (e.g., 'broadcast_ebu')
        """
        self.app.window.set_level_meter_preset(preset)

        second = self._get_window_by_id("second")
        if second and second.is_active:
            second.set_level_meter_preset(preset)

    def open_second_window(self) -> None:
        """Open the second window."""
        second_window = (
            self.window_manager.get_window("second") if self.window_manager else None
        )

        if second_window is None and self.window_manager:
            second_window = self.window_manager.create_window(
                window_id="second",
                parent=self.app.window.window,  # Main window's tk root
                window_type="secondary",
            )

            # Apply saved settings for second window panels
            if self.app.settings_manager:
                show_meters = getattr(
                    self.app.settings_manager.settings,
                    "second_window_show_meters",
                    False,
                )
                show_info = getattr(
                    self.app.settings_manager.settings,
                    "second_window_show_info_panel",
                    True,
                )

                second_window.set_meters_visibility(show_meters)
                if not show_info and hasattr(second_window, "info_panel"):
                    second_window.info_panel.grid_forget()
                    second_window.info_panel_visible = False

            def close_with_menu_update():
                if self.app.menu:
                    self.app.menu.on_second_window_closed()
                self.window_manager.close_window("second")

            second_window.window.protocol("WM_DELETE_WINDOW", close_with_menu_update)

            # Sync content with main window
            self.app.window.window.after(50, self._sync_second_window_content)
        elif second_window:
            # Just bring existing window to front
            second_window.window.lift()
            second_window.window.focus_force()

    def _sync_second_window_content(self) -> None:
        """Synchronize content from main window to second window."""
        second = self._get_window_by_id("second")
        if second and second.is_active:
            self.update_display()

            if second.info_panel_visible:
                self.update_info_panel()

            # Handle spectrogram content based on current state
            if second.ui_state.meters_visible:
                # Use when_spectrograms_ready to wait for initialization
                def sync_spectrogram_state():
                    if (
                        self.app.state.recording.is_recording
                        or self.app.audio_controller.is_monitoring
                    ):
                        # If recording or monitoring, start recording mode in spectrogram
                        sample_rate = self.app.config.audio.sample_rate
                        if second.mel_spectrogram:
                            if (
                                not second.mel_spectrogram.recording_handler.is_recording
                            ):
                                second.mel_spectrogram.clear()
                                second.mel_spectrogram.start_recording(sample_rate)
                    else:
                        # Otherwise show saved recording if available
                        self.show_saved_recording()

                self.when_spectrograms_ready(sync_spectrogram_state)

    def close_second_window(self) -> None:
        """Close the second window."""
        if self.window_manager:
            self.window_manager.close_window("second")

    def update_second_window_config(self, config: Dict[str, bool]) -> None:
        """Update second window configuration.

        Args:
            config: Configuration dictionary with 'show_meters' and 'show_info_panel'
        """
        second = self._get_window_by_id("second")
        if second and second.is_active:
            second.update_configuration(
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

        # Force layout update first
        window.window.update_idletasks()

        # Recalculate font size immediately after layout update
        if hasattr(window, "text_var") and window.text_var.get():
            window.adjust_text_font_size(window.text_var.get())
