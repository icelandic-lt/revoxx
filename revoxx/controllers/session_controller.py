"""Session controller for managing recording sessions."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..session import SessionManager, Session, SessionConfig
from ..utils.file_manager import RecordingFileManager
from ..utils.active_recordings import ActiveRecordings
from ..constants import FileConstants, UIConstants
from ..ui.dialogs import NewSessionDialog

if TYPE_CHECKING:
    from ..app import Revoxx


class SessionController:
    """Handles session management operations.

    This controller manages:
    - Creating new sessions
    - Opening existing sessions
    - Loading sessions
    - Session persistence
    - Script loading and reloading
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the session controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self.session_manager = SessionManager()

    def new_session(self):
        """Handle new session creation."""
        # Determine default base directory
        default_base_dir = None
        if self.app.current_session:
            default_base_dir = self.app.current_session.session_dir.parent
        else:
            # Try to get from settings
            default_base_dir = self.session_manager.get_default_base_dir()

        if not default_base_dir:
            default_base_dir = Path.cwd()  # Fallback to current working directory

        dialog = NewSessionDialog(
            self.app.root,
            default_base_dir,
            self.app.config.audio.sample_rate,
            self.app.config.audio.bit_depth,
            self.app.config.audio.input_device,
        )
        result = dialog.show()

        if result:
            try:
                # Create new session
                new_session = self.session_manager.create_session(
                    base_dir=result.base_dir,
                    speaker_name=result.speaker_name,
                    gender=result.gender,
                    emotion=result.emotion,
                    audio_config=SessionConfig(
                        sample_rate=result.sample_rate,
                        bit_depth=result.bit_depth,
                        channels=1,
                        format=result.recording_format.upper(),
                        input_device=result.input_device,
                    ),
                    script_source=result.script_path,
                    custom_dir_name=result.custom_dir_name,
                )

                # Load the new session
                self.load_session(new_session)

                # Update window title
                self.app.window.update_session_title(new_session.session_dir.name)

                # Update status
                self.app.window.set_status(
                    f"Created new session: {new_session.session_dir.name}"
                )

            except (OSError, ValueError, KeyError) as e:
                # OSError for file operations, ValueError for invalid params, KeyError for missing config
                self.app.window.set_status(f"Error creating session: {e}")

    def open_session(self, session_path: Path):
        """Open a session from the given path.

        Args:
            session_path: Path to the session directory
        """
        try:
            # Load session
            session = self.session_manager.load_session(session_path)

            # Load the session
            self.load_session(session)

            # Update window title
            self.app.window.update_session_title(session.session_dir.name)

            # Update recent sessions menu
            self.app.window._update_recent_sessions_menu()

            # Update status
            self.app.window.set_status(f"Loaded session: {session.session_dir.name}")

        except (OSError, json.JSONDecodeError, KeyError) as e:
            # OSError for file issues, JSONDecodeError for invalid JSON, KeyError for missing fields
            self.app.window.set_status(f"Error loading session: {e}")

    def get_recent_sessions(self):
        """Get list of recent sessions.

        Returns:
            List of recent session paths
        """
        return self.session_manager.get_recent_sessions()

    def get_current_session(self):
        """Get the current session.

        Returns:
            Current session or None
        """
        return self.app.current_session

    def load_session(self, session: Session):
        """Load a session and update the application state.

        Args:
            session: Session to load
        """
        self.app.current_session = session

        # Update paths
        self.app.script_file = session.session_dir / SessionManager.SCRIPT_FILE
        self.app.recording_dir = session.session_dir / "recordings"

        # Initialize or reinitialize file manager with new recording dir
        self.app.file_manager = RecordingFileManager(self.app.recording_dir)
        self.app.active_recordings = ActiveRecordings(self.app.file_manager)

        # Load script and scan recordings first
        self.reload_script_and_recordings()

        # Then apply saved sort settings from session (after data is loaded)
        if session:
            self.app.active_recordings.set_sort(
                session.sort_column, session.sort_reverse
            )

        # Update window title with session name
        self.app.root.title(f"Revoxx - {session.name}")

        # Update recent sessions menu
        self.app.window._update_recent_sessions_menu()

        # Resume at last position (like when starting the app)
        self.app.navigation_controller.resume_at_last_recording()

        # Show saved recording with delay to ensure mel spectrogram is ready
        self.app.root.after(
            UIConstants.INITIAL_DISPLAY_DELAY_MS,
            self.app.display_controller.show_saved_recording,
        )

        # Apply session audio config to runtime config
        if session.audio_config:
            self.app.config.audio.sample_rate = session.audio_config.sample_rate
            self.app.config.audio.bit_depth = session.audio_config.bit_depth
            self.app.config.audio.__post_init__()  # Update dtype and subtype

            # Update settings manager with new audio settings
            self.app.settings_manager.update_setting(
                "sample_rate", self.app.config.audio.sample_rate
            )
            self.app.settings_manager.update_setting(
                "bit_depth", self.app.config.audio.bit_depth
            )

            # Update shared state with new audio settings
            format_type = 1 if FileConstants.AUDIO_FILE_EXTENSION == ".flac" else 0
            self.app.shared_state.update_audio_settings(
                sample_rate=self.app.config.audio.sample_rate,
                bit_depth=self.app.config.audio.bit_depth,
                channels=self.app.config.audio.channels,
                format_type=format_type,
            )

            # Update info overlay if it's visible
            if self.app.window.info_overlay.visible:
                self.app.display_controller.update_info_overlay()

            # Reinitialize audio if needed
            if hasattr(self.app, "recorder"):
                self.app.device_controller.update_audio_settings()

            # Apply session device and channel settings
            self.app.device_controller.apply_session_audio_settings(
                session.audio_config
            )

    def load_script(self) -> None:
        """Load and parse the script file from current session."""
        if not self.app.current_session:
            print("Warning: No session loaded, cannot load script")
            self.app.state.recording.labels = []
            self.app.state.recording.utterances = []
            self.app.state.recording.takes = {}
            return

        # Script file must exist in valid session
        if not self.app.script_file or not self.app.script_file.exists():
            raise FileNotFoundError(
                f"Required script file not found in session: {self.app.script_file}"
            )

        self.reload_script_and_recordings()

        # Set initial index to 0 (will be overridden by resume if needed)
        self.app.state.recording.current_index = 0

    def reload_script_and_recordings(self) -> None:
        """Reload script content and scan for existing recordings.

        This method is used both during initial load and when switching sessions.
        It parses the script file, loads utterances, and scans for existing takes.
        """
        if not self.app.script_file or not self.app.script_file.exists():
            print(f"Warning: Script file not found: {self.app.script_file}")
            self.app.state.recording.labels = []
            self.app.state.recording.utterances = []
            self.app.state.recording.takes = {}
            return

        try:
            # Parse script file
            labels, utterances = self.app.script_manager.load_script(
                self.app.script_file
            )
            self.app.state.recording.labels = labels
            self.app.state.recording.utterances = utterances

            # Update active recordings with new data
            self.app.active_recordings.set_data(labels, utterances)
            self.app.state.recording.takes = self.app.active_recordings.get_all_takes()

        except (OSError, ValueError, KeyError) as e:
            # OSError for file issues, ValueError for parse errors, KeyError for missing fields
            print(f"Error loading script: {e}")
            # Set empty state on error
            self.app.state.recording.labels = []
            self.app.state.recording.utterances = []
            self.app.state.recording.takes = {}
