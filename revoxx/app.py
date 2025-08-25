"""Main application for Revoxx"""

# Set matplotlib backend before any matplotlib imports
import matplotlib

matplotlib.use("TkAgg")

import argparse
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional
import traceback

from .constants import KeyBindings, FileConstants
from .utils.config import RecorderConfig, load_config
from .utils.state import AppState
from .utils.file_manager import RecordingFileManager, ScriptFileManager
from .utils.active_recordings import ActiveRecordings
from .utils.settings_manager import SettingsManager
from .ui.main_window import MainWindow
from .utils.device_manager import get_device_manager
from .audio.buffer_manager import BufferManager
from .audio.shared_state import SharedState
from .session import SessionManager, Session

# Import all controllers
from .controllers import (
    AudioController,
    NavigationController,
    SessionController,
    DeviceController,
    DisplayController,
    FileOperationsController,
    DialogController,
    ProcessManager,
)


class Revoxx:
    """Main application class for Revoxx - Refactored version.

    This class manages the entire recording application using specialized controllers
    for better separation of concerns and maintainability.

    Attributes:
        config: Application configuration including audio, display, and UI settings
        session_controller: Manages session operations
        audio_controller: Handles audio recording and playback
        navigation_controller: Manages navigation through utterances
        device_controller: Handles audio device management
        display_controller: Manages UI updates
        file_operations_controller: Handles file operations
        dialog_controller: Manages dialog interactions
        process_manager: Manages background processes
    """

    def __init__(
        self,
        config: RecorderConfig,
        session: Optional[Session] = None,
        debug: bool = False,
    ):
        """Initialize the application.

        Args:
            config: Application configuration object
            session: Optional pre-loaded session
            debug: Enable debug output
        """
        self.config = config
        self.debug = debug

        # Initialize core components
        self.session_manager = SessionManager()
        self.current_session = session
        self.settings_manager = SettingsManager()

        # Initialize state
        self.state = AppState()

        # Initialize file managers
        self.script_manager = ScriptFileManager()

        if self.current_session:
            self.script_file = self.current_session.get_script_path()
            self.recording_dir = self.current_session.get_recordings_dir()
            self.file_manager = RecordingFileManager(self.recording_dir)
            self.active_recordings = ActiveRecordings(self.file_manager)
        else:
            self.script_file = None
            self.recording_dir = None
            self.file_manager = None
            self.active_recordings = None

        # Initialize shared state for audio
        self.shared_state = SharedState(create=True)

        # Initialize audio settings in shared state
        format_type = 1 if FileConstants.AUDIO_FILE_EXTENSION == ".flac" else 0
        self.shared_state.update_audio_settings(
            sample_rate=self.config.audio.sample_rate,
            bit_depth=self.config.audio.bit_depth,
            channels=self.config.audio.channels,
            format_type=format_type,
        )

        # Initialize recording state to STOPPED
        self.shared_state.stop_recording()

        # Initialize playback state to IDLE
        self.shared_state.stop_playback()

        # Initialize buffer manager
        self.buffer_manager = BufferManager()

        # Initialize device state flags
        self._default_input_in_effect = False
        self._default_output_in_effect = False
        self._notified_default_input = False
        self._notified_default_output = False
        self.last_output_error = False

        # Initialize process manager first (creates queues and shared resources)
        self.process_manager = ProcessManager(self)

        # Copy references for compatibility
        self.manager_dict = self.process_manager.manager_dict
        self.shutdown_event = self.process_manager.shutdown_event
        self.queue_manager = self.process_manager.queue_manager

        # Initialize manager_dict state
        self.manager_dict["recording"] = False
        self.manager_dict["playing"] = False
        self.manager_dict["audio_queue_active"] = self.config.display.show_spectrogram
        self.manager_dict["save_path"] = None
        self.manager_dict["debug"] = self.debug

        # Start background processes BEFORE UI initialization (like in original)
        self.process_manager.start_processes()

        # Initialize UI
        self.root = tk.Tk()
        self.root.withdraw()  # Hide until ready

        # Set window title with session name
        if self.current_session:
            self.root.title(f"Revoxx - {self.current_session.name}")
        else:
            self.root.title("Revoxx")

        # Prepare callbacks for MainWindow (will be populated after controllers are initialized)
        self.app_callbacks = {}

        # Create main window with proper parameters
        self.window = MainWindow(
            self.root,
            self.config,
            self.state.recording,
            self.state.ui,
            self.process_manager.manager_dict,
            self.app_callbacks,  # Will be populated after controllers are initialized
            self.settings_manager,
            self.shared_state,
        )

        # Initialize controllers after window is created
        self._init_controllers()

        # Populate app_callbacks after controllers are initialized
        self._populate_app_callbacks()

        # Apply saved settings
        self._apply_saved_settings()

        # Load session data if available
        if self.current_session:
            self.session_controller.load_session(self.current_session)

        # Bind keyboard shortcuts
        self._bind_keys()

        # Show window
        self.root.deiconify()

        # Initial display update
        self.display_controller.update_display()

        # Resume at last position if available
        if self.current_session:
            self.navigation_controller.resume_at_last_recording()
            # Show saved recording with delay to ensure mel spectrogram is ready
            # (same as in original implementation)
            from .constants import UIConstants

            self.root.after(
                UIConstants.INITIAL_DISPLAY_DELAY_MS,
                self.display_controller.show_saved_recording,
            )

        # Start audio queue processing transfer thread
        # This thread runs continuously and transfers audio data from the recording process
        # to the UI widgets when they are available. It polls every 100ms.
        # The thread will discard data if no widget is available to display it.
        self.audio_controller.start_audio_queue_processing()

    def _init_controllers(self):
        """Initialize all controllers."""
        self.audio_controller = AudioController(self)
        self.navigation_controller = NavigationController(self)
        self.session_controller = SessionController(self)
        self.device_controller = DeviceController(self)
        self.display_controller = DisplayController(self)
        self.file_operations_controller = FileOperationsController(self)
        self.dialog_controller = DialogController(self)

    def _populate_app_callbacks(self):
        """Populate app_callbacks dictionary with controller methods."""
        # Session callbacks
        self.app_callbacks["quit"] = self._quit
        self.app_callbacks["new_session"] = self._new_session
        self.app_callbacks["open_session"] = self._open_session
        self.app_callbacks["get_current_session"] = lambda: self.current_session
        self.app_callbacks["get_recent_sessions"] = (
            lambda: self.session_manager.get_recent_sessions()
        )
        self.app_callbacks["open_recent_session"] = (
            lambda path: self.session_controller.open_session(Path(path))
        )

        # Display callbacks
        self.app_callbacks["toggle_mel_spectrogram"] = (
            self.display_controller.toggle_mel_spectrogram
        )
        self.app_callbacks["update_info_panel"] = (
            self.display_controller.update_info_panel
        )

        # Audio callbacks
        self.app_callbacks["toggle_monitoring"] = (
            self.audio_controller.toggle_monitoring
        )

        # Device callbacks (if needed by menu)
        self.app_callbacks["set_input_device"] = self.device_controller.set_input_device
        self.app_callbacks["set_output_device"] = (
            self.device_controller.set_output_device
        )

        # Edit menu callbacks
        self.app_callbacks["delete_recording"] = (
            self.file_operations_controller.delete_current_recording
        )
        self.app_callbacks["show_utterance_order"] = (
            self.dialog_controller.show_utterance_order_dialog
        )
        self.app_callbacks["show_find_dialog"] = self.dialog_controller.show_find_dialog

    def _apply_saved_settings(self):
        """Apply saved settings to configuration."""
        settings = self.settings_manager.settings

        # Audio settings
        self.config.audio.sample_rate = settings.sample_rate
        self.config.audio.bit_depth = settings.bit_depth
        self.config.audio.sync_response_time_ms = settings.audio_sync_response_time_ms
        self.config.audio.__post_init__()  # Update dtype and subtype

        # Apply device settings through controller
        self.device_controller.apply_saved_settings()

        # Display settings
        self.config.display.show_spectrogram = settings.show_spectrogram
        self.config.ui.fullscreen = settings.fullscreen

        # Store window geometry for later use
        self._saved_window_geometry = settings.window_geometry

    def _bind_keys(self):
        """Bind keyboard shortcuts."""
        # Recording controls
        self.root.bind(
            f"<{KeyBindings.RECORD}>",
            lambda e: self.audio_controller.toggle_recording(),
        )
        self.root.bind(
            f"<{KeyBindings.PLAY}>", lambda e: self.audio_controller.play_current()
        )
        self.root.bind(
            "<Control-d>",
            lambda e: self.file_operations_controller.delete_current_recording(),
        )
        self.root.bind(
            "<Control-D>",
            lambda e: self.file_operations_controller.delete_current_recording(),
        )

        # Navigation keys
        self.root.bind(
            f"<{KeyBindings.NAVIGATE_UP}>",
            lambda e: self.navigation_controller.navigate(-1),
        )
        self.root.bind(
            f"<{KeyBindings.NAVIGATE_DOWN}>",
            lambda e: self.navigation_controller.navigate(1),
        )

        # Browse takes
        self.root.bind(
            f"<{KeyBindings.BROWSE_TAKES_LEFT}>",
            lambda e: self.navigation_controller.browse_takes(-1),
        )
        self.root.bind(
            f"<{KeyBindings.BROWSE_TAKES_RIGHT}>",
            lambda e: self.navigation_controller.browse_takes(1),
        )

        # Toggle displays
        for key in KeyBindings.TOGGLE_SPECTROGRAM:
            self.root.bind(
                f"<{key}>", lambda e: self.display_controller.toggle_mel_spectrogram()
            )
        for key in KeyBindings.TOGGLE_LEVEL_METER:
            self.root.bind(
                f"<{key}>", lambda e: self.display_controller.toggle_level_meter()
            )

        # Dialog keys
        self.root.bind(
            "<Control-f>", lambda e: self.dialog_controller.show_find_dialog()
        )
        self.root.bind(
            "<Control-F>", lambda e: self.dialog_controller.show_find_dialog()
        )
        self.root.bind(
            "<Control-u>",
            lambda e: self.dialog_controller.show_utterance_order_dialog(),
        )
        self.root.bind(
            "<Control-U>",
            lambda e: self.dialog_controller.show_utterance_order_dialog(),
        )
        self.root.bind("<Control-q>", lambda e: self._quit())
        self.root.bind("<Control-Q>", lambda e: self._quit())

        # Additional key bindings
        self.root.bind(
            f"<{KeyBindings.TOGGLE_MONITORING}>",
            lambda e: self.audio_controller.toggle_monitoring(),
        )
        self.root.bind(
            f"<{KeyBindings.TOGGLE_FULLSCREEN}>", lambda e: self._toggle_fullscreen()
        )

        # Delete recording (both Control and no modifier variants)
        import platform

        if platform.system() == "Darwin":  # macOS
            modifier = "Command"
        else:
            modifier = "Control"
        self.root.bind(
            f"<{modifier}-{KeyBindings.DELETE_RECORDING}>",
            lambda e: self.file_operations_controller.delete_current_recording(),
        )

        # Help and info
        self.root.bind(
            f"<{KeyBindings.SHOW_HELP}>", lambda e: self.dialog_controller.show_help()
        )
        self.root.bind(
            f"<{KeyBindings.SHOW_INFO}>",
            lambda e: self.window._toggle_info_panel_callback(),
        )

        # Session management with platform-specific modifiers
        if platform.system() == "Darwin":  # macOS uses Command
            self.root.bind("<Command-n>", lambda e: self._new_session())
            self.root.bind("<Command-N>", lambda e: self._new_session())
            self.root.bind("<Command-o>", lambda e: self._open_session())
            self.root.bind("<Command-O>", lambda e: self._open_session())
            self.root.bind(
                "<Command-i>", lambda e: self.dialog_controller.show_settings_dialog()
            )
            self.root.bind(
                "<Command-I>", lambda e: self.dialog_controller.show_settings_dialog()
            )
            self.root.bind(
                "<Command-f>", lambda e: self.dialog_controller.show_find_dialog()
            )
            self.root.bind(
                "<Command-F>", lambda e: self.dialog_controller.show_find_dialog()
            )
            self.root.bind(
                "<Command-u>",
                lambda e: self.dialog_controller.show_utterance_order_dialog(),
            )
            self.root.bind(
                "<Command-U>",
                lambda e: self.dialog_controller.show_utterance_order_dialog(),
            )
            self.root.bind("<Command-q>", lambda e: self._quit())
            self.root.bind("<Command-Q>", lambda e: self._quit())

        # Session keys
        self.root.bind("<Control-n>", lambda e: self._new_session())
        self.root.bind("<Control-N>", lambda e: self._new_session())
        self.root.bind("<Control-o>", lambda e: self._open_session())
        self.root.bind("<Control-O>", lambda e: self._open_session())

        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _new_session(self):
        """Create a new session."""
        from .session import SessionConfig

        result = self.dialog_controller.show_new_session_dialog()

        if result:
            try:
                # Create new session with all required parameters
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
                self.current_session = new_session
                self.session_controller.load_session(new_session)

                # Update window title
                self.window.update_session_title(new_session.session_dir.name)

                # Update status
                self.window.set_status(
                    f"Created new session: {new_session.session_dir.name}"
                )

            except Exception as e:
                self.window.set_status(f"Error creating session: {e}")

    def _open_session(self):
        """Open an existing session."""
        session_path = self.dialog_controller.show_open_session_dialog()

        if session_path:
            try:
                session = self.session_manager.load_session(session_path)
                self.current_session = session
                self.session_controller.load_session(session)
                self.window.update_session_title(session.session_dir.name)

                if hasattr(self.window, "_update_recent_sessions_menu"):
                    self.window._update_recent_sessions_menu()

                self.window.set_status(f"Loaded session: {session.session_dir.name}")

            except Exception as e:
                self.window.set_status(f"Error loading session: {e}")

    def _quit(self):
        """Quit the application."""
        if not self.dialog_controller.confirm_quit():
            return

        # Stop any ongoing recording
        if self.state.recording.is_recording:
            self.audio_controller.stop_recording()

        # Save current session state and remember it for next start
        if self.current_session:
            # Save session state
            if hasattr(self.current_session, "save"):
                self.current_session.save()
            self.settings_manager.update_setting(
                "last_session_path", str(self.current_session.session_dir)
            )

        # Cleanup
        self.process_manager.shutdown()
        self.dialog_controller.cleanup()

        # Close UI
        try:
            self.root.destroy()
        except (AttributeError, RuntimeError):
            pass

        # Exit
        sys.exit(0)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode.
        This setting is saved to the user's settings.
        """
        current_state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current_state)

        # Update config
        self.config.ui.fullscreen = not current_state

        # Update settings
        self.settings_manager.update_setting("fullscreen", not current_state)

    def notify_if_default_device(self, device_type: str = "output") -> None:
        """Notify user once if using default device due to missing selection.

        Args:
            device_type: Either 'output' or 'input'
        """
        if device_type == "output":
            # If default output device is in effect and not yet notified, inform user once
            if self._default_output_in_effect and not self._notified_default_output:
                try:
                    self.window.show_message(
                        "Using system default output device (no saved/available selection)"
                    )
                except AttributeError:
                    pass
                self._notified_default_output = True

            # Additionally, warn once if last stream open failed
            if hasattr(self, "last_output_error") and self.last_output_error:
                self.window.set_status(
                    "Output device unavailable. Using system default if possible."
                )
                self.last_output_error = False

        elif device_type == "input":
            # If default input device is in effect and not yet notified, inform user once
            if self._default_input_in_effect and not self._notified_default_input:
                try:
                    self.window.show_message(
                        "Using system default input device (no saved/available selection)"
                    )
                except AttributeError:
                    pass
                self._notified_default_input = True

    def run(self):
        """Run the application."""
        self.window.focus_window()
        self.root.mainloop()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Revoxx - Speech recording application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Session management
    session = parser.add_argument_group("session management")
    session.add_argument(
        "--session", type=str, help="path to session directory (.revoxx)"
    )

    # Audio configuration
    audio = parser.add_argument_group("audio configuration")
    audio.add_argument(
        "--show-devices",
        action="store_true",
        help="show available audio devices and exit",
    )
    audio.add_argument(
        "--audio-device",
        type=str,
        help="audio device name (sets both input and output)",
    )
    audio.add_argument(
        "--audio-in", type=str, default=None, help="input device index or name"
    )
    audio.add_argument(
        "--audio-out", type=str, default=None, help="output device index or name"
    )
    audio.add_argument(
        "--start-idx", type=int, default=0, help="starting index (not id) of UI"
    )

    # Debug options
    parser.add_argument("--debug", action="store_true", help="enable debug output")

    return parser.parse_args()


def _handle_show_devices() -> None:
    """Display available audio devices and exit."""
    device_manager = get_device_manager()
    print("\nInput Devices:")
    for device in device_manager.get_input_devices():
        print(
            f"  [{device['index']}] {device['name']} ({device['max_input_channels']} channels)"
        )
    print("\nOutput Devices:")
    for device in device_manager.get_output_devices():
        print(
            f"  [{device['index']}] {device['name']} ({device['max_output_channels']} channels)"
        )
    sys.exit(0)


def _apply_command_line_overrides(args, config) -> None:
    """Apply command line arguments to configuration.

    Args:
        args: Parsed command line arguments
        config: Application configuration to modify
    """
    if args.audio_device:
        # Set both input and output to the same device
        config.audio.input_device = args.audio_device
        config.audio.output_device = args.audio_device
    if args.audio_in is not None:
        config.audio.input_device = args.audio_in
    if args.audio_out is not None:
        config.audio.output_device = args.audio_out

    # Set display defaults to True (configurable in app)
    config.display.show_spectrogram = True
    config.display.show_info_overlay = True
    config.display.show_level_meter = True


def _load_session_from_args(args, session_manager):
    """Load session from command line arguments or last session.

    Args:
        args: Parsed command line arguments
        session_manager: Session manager instance

    Returns:
        Session object or None
    """
    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            print(f"Error: Session directory not found: {session_path}")
            sys.exit(1)

        session = session_manager.load_session(session_path)
        if not session:
            print(f"Error: Failed to load session from {session_path}")
            sys.exit(1)
        return session

    # Try to load last session
    last_session_path = session_manager.get_last_session()
    if last_session_path:
        try:
            session = session_manager.load_session(last_session_path)
            if session:
                print(f"Loaded last session: {session.name}")
                return session
        except Exception:
            # Last session not available, will need to create/select one
            pass

    return None


def main():
    """Main entry point for the application."""
    args = parse_arguments()

    if args.show_devices:
        _handle_show_devices()

    config = load_config()

    _apply_command_line_overrides(args, config)
    session_manager = SessionManager()
    session = _load_session_from_args(args, session_manager)

    # Create and run application
    try:
        app = Revoxx(config, session, debug=args.debug)
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
