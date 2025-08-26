"""Display controller for managing UI updates and visualization."""

from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import soundfile as sf

from ..constants import MsgType

if TYPE_CHECKING:
    from ..app import Revoxx


class DisplayController:
    """Handles display updates and UI state management.

    This controller manages:
    - Display content updates
    - UI element visibility
    - Status messages
    - Mel spectrogram visualization
    - Info overlay updates
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the display controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app

    def update_display(self) -> None:
        """Update the main display with current utterance information."""
        if not self.app.state.recording.utterances:
            # No utterances loaded - show empty state
            self.app.window.update_display(0, False, 0)
            return

        # Get current index and display position
        current_index = self.app.state.recording.current_index
        is_recording = self.app.state.recording.is_recording

        # Get display position (user-friendly numbering)
        display_pos = self.app.navigation_controller.get_display_position(current_index)

        # Update the display with the correct parameters
        self.app.window.update_display(current_index, is_recording, display_pos)

    def show_saved_recording(self) -> None:
        """Load and display a saved recording if it exists."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            return

        current_take = self.app.state.recording.get_current_take(current_label)
        if current_take == 0:
            # No recording exists, clear display
            if self.app.window.mel_spectrogram:
                self.app.window.mel_spectrogram.clear()
            # Update info panel even when no recording
            self.update_info_panel()
            return

        # Load the recording
        filepath = self.app.file_manager.get_recording_path(current_label, current_take)
        if filepath.exists():
            try:
                audio_data, sr = self.app.file_manager.load_audio(filepath)

                # Display in mel spectrogram
                if self.app.window.mel_spectrogram:
                    self.app.window.mel_spectrogram.show_recording(audio_data, sr)

                # Update info panel with recording info
                self.update_info_panel()
            except (OSError, ValueError) as e:
                # OSError for file operations, ValueError for invalid audio data
                self.app.window.set_status(
                    f"Error loading recording: {e}", MsgType.ERROR
                )

    def toggle_meters(self) -> None:
        """Toggle both mel spectrogram and level meter visualization."""
        # Toggle visibility state
        self.app.state.ui.meters_visible = not self.app.state.ui.meters_visible

        # Update both meters in window
        self.app.window.set_meters_visibility(self.app.state.ui.meters_visible)

        # Update audio queue state - needed if either spectrogram or level meter is visible
        self.app.audio_controller.update_audio_queue_state()

        # Show current recording if available - but only if not currently recording/monitoring
        if self.app.state.ui.meters_visible:
            if (
                not self.app.state.recording.is_recording
                and not self.app.audio_controller.is_monitoring
            ):
                self.app.root.after(50, self.show_saved_recording)
            elif self.app.state.recording.is_recording:
                # If we're currently recording and meters were just turned on,
                # ensure the mel spectrogram knows it's in recording mode
                if self.app.window.mel_spectrogram:
                    if (
                        not self.app.window.mel_spectrogram.recording_handler.is_recording
                    ):
                        # Start recording mode with current sample rate
                        sample_rate = self.app.config.audio.sample_rate
                        self.app.window.mel_spectrogram.start_recording(sample_rate)

        # Update settings
        self.app.settings_manager.update_setting(
            "show_meters", self.app.state.ui.meters_visible
        )

    def update_info_panel(self) -> None:
        """Update the combined info panel with current recording information."""
        current_label = self.app.state.recording.current_label
        if not current_label:
            # No current utterance - show default parameters
            recording_params = self._get_recording_parameters()
            self.app.window.update_info_panel(recording_params)
            return

        # Get recording parameters
        recording_params = self._get_recording_parameters()

        # Check if recording exists
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

        # Update the info panel
        self.app.window.update_info_panel(recording_params)

    def update_recording_timer(self, elapsed_time: float) -> None:
        """Update the recording timer display.

        Args:
            elapsed_time: Elapsed recording time in seconds
        """
        # recording_timer should exist if used
        self.app.window.recording_timer.update(elapsed_time)

    def reset_recording_timer(self) -> None:
        """Reset the recording timer display."""
        # recording_timer should exist if used
        self.app.window.recording_timer.reset()

    def update_level_meter(self, level: float) -> None:
        """Update the level meter display.

        Args:
            level: Audio level value (0.0 to 1.0)
        """
        # Widget must exist when keyboard bindings are active
        self.app.window.embedded_level_meter.update_level(level)

    def reset_level_meter(self) -> None:
        """Reset the level meter display."""
        # Widget must exist when keyboard bindings are active
        self.app.window.embedded_level_meter.reset()

    def set_status(self, status: str) -> None:
        """Set the status bar text.

        Args:
            status: Status text to display
        """
        self.app.window.set_status(status)

    def update_window_title(self, title: Optional[str] = None) -> None:
        """Update the window title.

        Args:
            title: New title text, or None for default
        """
        if title:
            self.app.root.title(title)
        else:
            # Default title
            session_name = ""
            if self.app.current_session:
                session_name = f" - {self.app.current_session.name}"
            self.app.root.title(f"Revoxx{session_name}")

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
