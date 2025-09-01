"""Main application for Revoxx"""

# Set matplotlib backend before any matplotlib imports
import matplotlib

matplotlib.use("TkAgg")

import argparse
import sys
from pathlib import Path
from typing import Optional
import traceback

from .constants import KeyBindings, FileConstants, MsgType
from .utils.config import RecorderConfig, load_config
from .utils.state import AppState
from .utils.file_manager import RecordingFileManager, ScriptFileManager
from .utils.active_recordings import ActiveRecordings
from .utils.settings_manager import SettingsManager
from .utils.process_cleanup import ProcessCleanupManager
from .ui.window_manager import WindowManager
from .ui.menus.application_menu import ApplicationMenu
from .ui.themes import theme_manager, ThemePreset
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

        # Initialize settings manager first (needed for process initialization)
        self.settings_manager = SettingsManager()

        # Initialize core components
        self.session_manager = SessionManager()
        self.current_session = session

        # Initialize state
        self.state = AppState()

        # Initialize file managers
        self.script_manager = ScriptFileManager()

        # Setup process cleanup manager
        self.cleanup_manager = ProcessCleanupManager(
            cleanup_callback=self._perform_cleanup, debug=self.debug
        )
        self.cleanup_manager.setup_signal_handlers()

        if self.current_session and self.current_session.session_dir:
            # Use session_dir directly to avoid method calls that check for None
            self.script_file = self.current_session.session_dir / (
                self.current_session.script_path or "script.txt"
            )
            self.recording_dir = self.current_session.session_dir / "recordings"
            self.file_manager = RecordingFileManager(self.recording_dir)
            self.active_recordings = ActiveRecordings(self.file_manager)
        else:
            self.script_file = None
            self.recording_dir = None
            self.file_manager = None
            self.active_recordings = None
            # Clear invalid session
            if self.current_session and not self.current_session.session_dir:
                self.current_session = None

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
        self.process_manager.set_audio_queue_active(
            self.settings_manager.settings.show_meters
        )
        self.process_manager.set_save_path(None)
        self.manager_dict["debug"] = self.debug

        # Start background processes BEFORE UI initialization (like in original)
        self.process_manager.start_processes()

        # Initialize theme from settings
        if self.settings_manager:
            saved_theme = getattr(
                self.settings_manager.settings, "theme", ThemePreset.CYAN.value
            )
            try:
                theme_manager.set_theme(ThemePreset(saved_theme))
            except ValueError:
                theme_manager.set_theme(ThemePreset.CYAN)

        # Refresh UI constants with theme colors
        from .constants import UIConstants

        UIConstants.refresh()

        # Initialize WindowManager
        self.window_manager = WindowManager(self)

        # Prepare callbacks for windows (will be populated after controllers are initialized)
        self.app_callbacks = {}

        # Create main window using WindowManager
        self.window = self.window_manager.create_window(
            window_id="main", parent=None, window_type="main"
        )

        # Hide window until ready
        self.window.window.withdraw()

        # Restore any saved windows (e.g., second window if it was enabled)
        self.window_manager.restore_saved_windows()

        # Initialize controllers after window is created
        self._init_controllers()

        # Populate app_callbacks after controllers are initialized
        self._populate_app_callbacks()

        # Apply saved settings BEFORE creating menu so device settings are loaded
        self._apply_saved_settings()

        # Create application menu after settings are loaded
        self.menu = ApplicationMenu(self)
        self.menu.create_menu()

        # Load session data if available
        if self.current_session:
            try:
                self.session_controller.load_session(self.current_session)
            except Exception as e:
                print(f"Error loading session: {e}")
                traceback.print_exc()

        # Show window
        self.window.window.deiconify()

        # Initial display update
        self.display_controller.update_display()

        # Resume at last position if available
        if self.current_session:
            self.navigation_controller.resume_at_last_recording()
            self.display_controller.show_saved_recording_when_ready()

        # Start audio queue processing transfer thread
        # This thread runs continuously and transfers audio data from the recording process
        # to the UI widgets when they are available. It polls every 100ms.
        # The thread will discard data if no widget is available to display it.
        self.audio_controller.start_audio_queue_processing()

        # Bind keyboard shortcuts
        self._bind_keys()

    def _init_controllers(self):
        """Initialize all controllers."""
        self.audio_controller = AudioController(self)
        self.navigation_controller = NavigationController(self)
        self.session_controller = SessionController(self)
        self.device_controller = DeviceController(self)
        self.display_controller = DisplayController(self, self.window_manager)
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
        self.app_callbacks["toggle_meters"] = self.display_controller.toggle_meters
        self.app_callbacks["update_info_panel"] = (
            self.display_controller.update_info_panel
        )
        self.app_callbacks["update_second_window"] = self._update_second_window_content

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
        self.config.display.show_spectrogram = settings.show_meters
        self.config.ui.fullscreen = settings.fullscreen

    def _bind_keys(self):
        """Bind keyboard shortcuts."""
        # Recording controls
        self.window.window.bind(
            f"<{KeyBindings.RECORD}>",
            lambda e: self.audio_controller.toggle_recording(),
        )
        self.window.window.bind(
            f"<{KeyBindings.PLAY}>", lambda e: self.audio_controller.play_current()
        )
        self.window.window.bind(
            "<Control-d>",
            lambda e: self.file_operations_controller.delete_current_recording(),
        )
        self.window.window.bind(
            "<Control-D>",
            lambda e: self.file_operations_controller.delete_current_recording(),
        )

        # Navigation keys
        self.window.window.bind(
            f"<{KeyBindings.NAVIGATE_UP}>",
            lambda e: self.navigation_controller.navigate(-1),
        )
        self.window.window.bind(
            f"<{KeyBindings.NAVIGATE_DOWN}>",
            lambda e: self.navigation_controller.navigate(1),
        )

        # Browse takes
        self.window.window.bind(
            f"<{KeyBindings.BROWSE_TAKES_LEFT}>",
            lambda e: self.navigation_controller.browse_takes(-1),
        )
        self.window.window.bind(
            f"<{KeyBindings.BROWSE_TAKES_RIGHT}>",
            lambda e: self.navigation_controller.browse_takes(1),
        )

        # Toggle displays
        for key in KeyBindings.TOGGLE_SPECTROGRAM:
            self.window.window.bind(
                f"<{key}>", lambda e: self.display_controller.toggle_meters("main")
            )

        # Dialog keys
        self.window.window.bind(
            "<Control-f>", lambda e: self.dialog_controller.show_find_dialog()
        )
        self.window.window.bind(
            "<Control-F>", lambda e: self.dialog_controller.show_find_dialog()
        )
        self.window.window.bind(
            "<Control-u>",
            lambda e: self.dialog_controller.show_utterance_order_dialog(),
        )
        self.window.window.bind(
            "<Control-U>",
            lambda e: self.dialog_controller.show_utterance_order_dialog(),
        )
        self.window.window.bind("<Control-q>", lambda e: self._quit())
        self.window.window.bind("<Control-Q>", lambda e: self._quit())

        # Additional key bindings
        self.window.window.bind(
            f"<{KeyBindings.TOGGLE_MONITORING}>",
            lambda e: self.audio_controller.toggle_monitoring(),
        )
        self.window.window.bind(
            f"<{KeyBindings.TOGGLE_FULLSCREEN}>", lambda e: self._toggle_fullscreen()
        )

        # Delete recording (both Control and no modifier variants)
        import platform

        if platform.system() == "Darwin":  # macOS
            modifier = "Command"
        else:
            modifier = "Control"
        self.window.window.bind(
            f"<{modifier}-{KeyBindings.DELETE_RECORDING}>",
            lambda e: self.file_operations_controller.delete_current_recording(),
        )

        # Help and info
        self.window.window.bind(
            f"<{KeyBindings.SHOW_HELP}>", lambda e: self.dialog_controller.show_help()
        )
        self.window.window.bind(
            f"<{KeyBindings.SHOW_INFO}>",
            lambda e: self.display_controller.toggle_info_panel(),
        )

        # Second window shortcuts (Shift + key)
        self.window.window.bind(
            "<Shift-M>",
            lambda e: self._toggle_second_window_meters(),
        )
        self.window.window.bind(
            "<Shift-I>",
            lambda e: self._toggle_second_window_info_panel(),
        )
        self.window.window.bind(
            "<Shift-F10>",
            lambda e: self._toggle_second_window_fullscreen(),
        )

        # Session management with platform-specific modifiers
        if platform.system() == "Darwin":  # macOS uses Command
            self.window.window.bind("<Command-n>", lambda e: self._new_session())
            self.window.window.bind("<Command-N>", lambda e: self._new_session())
            self.window.window.bind("<Command-o>", lambda e: self._open_session())
            self.window.window.bind("<Command-O>", lambda e: self._open_session())
            self.window.window.bind(
                "<Command-i>", lambda e: self.dialog_controller.show_settings_dialog()
            )
            self.window.window.bind(
                "<Command-I>", lambda e: self.dialog_controller.show_settings_dialog()
            )
            self.window.window.bind(
                "<Command-f>", lambda e: self.dialog_controller.show_find_dialog()
            )
            self.window.window.bind(
                "<Command-F>", lambda e: self.dialog_controller.show_find_dialog()
            )
            self.window.window.bind(
                "<Command-u>",
                lambda e: self.dialog_controller.show_utterance_order_dialog(),
            )
            self.window.window.bind(
                "<Command-U>",
                lambda e: self.dialog_controller.show_utterance_order_dialog(),
            )
            # Override macOS Cmd+Q behavior
            self.window.window.bind("<Command-q>", lambda e: self._handle_cmd_q())
            self.window.window.bind("<Command-Q>", lambda e: self._handle_cmd_q())
            # Also try to catch it with createcommand
            self.window.window.createcommand("::tk::mac::Quit", self._handle_cmd_q)

        # Session keys
        self.window.window.bind("<Control-n>", lambda e: self._new_session())
        self.window.window.bind("<Control-N>", lambda e: self._new_session())
        self.window.window.bind("<Control-o>", lambda e: self._open_session())
        self.window.window.bind("<Control-O>", lambda e: self._open_session())

        # Window close event
        self.window.window.protocol("WM_DELETE_WINDOW", self._quit)

    def _new_session(self, default_script=None):
        """Create a new session.

        Args:
            default_script: Optional path to a script file to use for the new session
        """
        from .session import SessionConfig

        result = self.dialog_controller.show_new_session_dialog(
            default_script=default_script
        )

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
                self.display_controller.update_window_title()

                # Update status
                self.display_controller.set_status(
                    f"Created new session: {new_session.session_dir.name}"
                )

            except Exception as e:
                self.display_controller.set_status(
                    f"Error creating session: {e}", MsgType.ERROR
                )

    def _open_session(self):
        """Open an existing session."""
        session_path = self.dialog_controller.show_open_session_dialog()

        if session_path:
            try:
                session = self.session_manager.load_session(session_path)
                self.current_session = session
                self.session_controller.load_session(session)
                self.display_controller.update_window_title()

                # Update recent sessions menu
                if hasattr(self, "menu"):
                    self.menu.update_recent_sessions()

                self.display_controller.set_status(
                    f"Loaded session: {session.session_dir.name}"
                )

            except Exception as e:
                self.display_controller.set_status(
                    f"Error loading session: {e}", MsgType.ERROR
                )

    def _handle_cmd_q(self):
        """Handle Cmd+Q on macOS specifically."""
        if self.debug:
            print("[App] Cmd+Q intercepted - calling _quit()")
        self._quit()
        return "break"  # Prevent default handling

    def _perform_cleanup(self):
        """Perform cleanup when signals are received or on emergency exit."""
        # Only do critical cleanup - no UI interactions
        if hasattr(self, "process_manager"):
            if self.debug:
                print("[App] Shutting down process manager...")
            self.process_manager.shutdown()

        # Clean up buffer manager
        if hasattr(self, "buffer_manager"):
            if self.debug:
                print("[App] Cleaning up buffer manager...")
            # No wait needed - processes already terminated
            self.buffer_manager.cleanup_all(wait_time=0)

        # Clean up shared state
        if hasattr(self, "shared_state"):
            if self.debug:
                print("[App] Cleaning up shared state...")
            self.shared_state.close()
            self.shared_state.unlink()

    def _quit(self):
        """Quit the application."""
        if self.debug:
            print("[App] _quit() called")

        if not self.dialog_controller.confirm_quit():
            if self.debug:
                print("[App] Quit cancelled by user")
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

        # Save current window positions before saving settings
        self.window_manager.save_all_positions()

        # Save settings to disk before cleanup
        self.settings_manager.save_settings()

        # Dialog cleanup
        self.dialog_controller.cleanup()

        # Perform all non-UI cleanup through the centralized cleanup method
        self._perform_cleanup()

        # Mark cleanup as done in cleanup manager
        self.cleanup_manager.cleanup_complete()

        # Close UI
        try:
            self.window.window.destroy()
        except (AttributeError, RuntimeError):
            pass

        # Exit
        sys.exit(0)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode.
        This setting is saved to the user's settings.
        """
        current_state = self.window.window.attributes("-fullscreen")
        self.window.window.attributes("-fullscreen", not current_state)

        # Update config
        self.config.ui.fullscreen = not current_state

        # Update settings
        self.settings_manager.update_setting("fullscreen", not current_state)

    def _toggle_second_window_fullscreen(self):
        """Toggle fullscreen mode for the second window."""
        self.display_controller.toggle_window_fullscreen("monitor1")

    def _update_second_window_content(self) -> None:
        """Update second window with current content from main window."""
        if self.has_active_second_window:
            self.display_controller.update_display()
            self.display_controller.update_info_panel()
            self.display_controller.show_saved_recording()

    def _toggle_second_window_meters(self) -> None:
        """Toggle meters visibility in second window via keyboard shortcut."""
        new_state = self.display_controller.toggle_window_meters("monitor1")
        if new_state is not None:
            # Update menu checkbox
            self.menu.menu_vars["second_window_meters"].set(new_state)
            # Save setting
            self.settings_manager.save_window_settings(
                "monitor1", {"meters_visible": new_state}
            )

    def _toggle_second_window_info_panel(self) -> None:
        """Toggle info panel visibility in second window via keyboard shortcut."""
        new_state = self.display_controller.toggle_window_info_panel("monitor1")
        if new_state is not None:
            # Update menu checkbox
            self.menu.menu_vars["second_window_info"].set(new_state)
            # Save setting
            self.settings_manager.save_window_settings(
                "monitor1", {"info_panel_visible": new_state}
            )

    def notify_if_default_device(self, device_type: str = "output") -> None:
        """Notify user once if using default device due to missing selection.

        Args:
            device_type: Either 'output' or 'input'
        """
        if device_type == "output":
            # If default output device is in effect and not yet notified, inform user once
            if self._default_output_in_effect and not self._notified_default_output:
                self.display_controller.set_status(
                    "Using system default output device (no saved/available selection)",
                    MsgType.TEMPORARY,
                )
                self._notified_default_output = True

            # Additionally, warn once if last stream open failed
            if hasattr(self, "last_output_error") and self.last_output_error:
                self.display_controller.set_status(
                    "Output device unavailable. Using system default if possible.",
                    MsgType.TEMPORARY,
                )
                self.last_output_error = False

        elif device_type == "input":
            # If default input device is in effect and not yet notified, inform user once
            if self._default_input_in_effect and not self._notified_default_input:
                self.display_controller.set_status(
                    "Using system default input device (no saved/available selection)",
                    MsgType.TEMPORARY,
                )
                self._notified_default_input = True

    @property
    def has_active_second_window(self) -> bool:
        """Check if second window exists and is active.

        Returns:
            True if second window exists and is active, False otherwise
        """
        monitor1 = self.window_manager.get_window("monitor1")
        return monitor1 and monitor1.is_active if monitor1 else False

    def run(self):
        """Run the application."""
        # Re-register SIGINT handler right before mainloop
        # Tkinter might have changed it during setup
        self.cleanup_manager.refresh_sigint_handler()

        self.window.focus_window()
        self.window.window.mainloop()


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
                # Check session_dir exists
                if not session.session_dir:
                    print(
                        f"Warning: Session {session.name} has no session_dir, skipping"
                    )
                    return None
                return session
        except Exception as e:
            # Last session not available, will need to create/select one
            print(f"Warning: Could not load last session from {last_session_path}: {e}")

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
