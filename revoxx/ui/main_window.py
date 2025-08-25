"""Main window UI for the Revoxx Recorder."""

from typing import Optional, Callable
import tkinter as tk
import tkinter.font as tkfont
import platform

from ..constants import KeyBindings
from .dialogs import SessionSettingsDialog
from .dialogs.dataset_dialog import DatasetDialog
from .icon import AppIcon
from .level_meter.led_level_meter import LEDLevelMeter
from .level_meter.config import RECORDING_STANDARDS
from .level_meter.config import RecordingStandard, get_standard_description
from .menus.audio_devices import AudioDevicesMenuBuilder
from pathlib import Path
from .spectrogram import MelSpectrogramWidget
from ..constants import UIConstants
from .themes import theme_manager, ThemePreset
from ..utils.config import RecorderConfig
from ..utils.state import UIState, RecordingState
from ..utils.settings_manager import SettingsManager


class MainWindow:
    """Main application window for the Revoxx Recorder.

    This class manages the main UI window, including layout, widget creation,
    font scaling, and window resizing. It provides a three-panel layout:
    - Top info bar: Status, recording indicator, and progress
    - Center content: Current utterance display
    - Bottom control: Mel spectrogram and keyboard shortcuts

    The window supports dynamic font scaling based on window size and
    can toggle between fullscreen and windowed modes.

    Attributes:
        root: Tkinter root window
        config: Application configuration
        recording_state: State manager for recording data
        ui_state: State manager for UI elements
        mel_spectrogram: Optional mel spectrogram visualization widget
        main_frame: Main container frame
        info_frame: Top information bar
        utterance_frame: Middle area for utterance text display
        spec_container: Container for spectrogram and level meter
        info_panel: Bottom info panel with recording details
    """

    def __init__(
        self,
        root: tk.Tk,
        config: RecorderConfig,
        recording_state: RecordingState,
        ui_state: UIState,
        manager_dict: dict = None,
        app_callbacks: dict = None,
        settings_manager: Optional[SettingsManager] = None,
        shared_audio_state=None,
    ):
        """Initialize the main window.

        Args:
            root: Tkinter root window
            config: Application configuration with display and UI settings
            recording_state: Recording state manager tracking current utterance
            ui_state: UI state manager for window properties
            manager_dict: Shared state dictionary
            app_callbacks: Application callbacks
            settings_manager: Settings manager for persisting preferences
            shared_audio_state: Shared audio state for synchronization
        """
        self.root = root
        self.config = config
        self.recording_state = recording_state
        self.ui_state = ui_state
        self.manager_dict = manager_dict
        self.app_callbacks = app_callbacks
        self.settings_manager = settings_manager
        self.shared_audio_state = shared_audio_state

        # Initialize theme from settings
        saved_theme = getattr(
            self.settings_manager.settings, "theme", ThemePreset.CYAN.value
        )
        try:
            theme_manager.set_theme(ThemePreset(saved_theme))
        except ValueError:
            theme_manager.set_theme(ThemePreset.CYAN)

        # Refresh UI constants with theme colors
        UIConstants.refresh()

        # Get screen information
        self._setup_screen_geometry()

        # Configure window
        self._setup_window()

        # Create menu bar
        self._create_menu()

        # Create UI elements
        self._create_ui()

        # Initialize embedded level meter (will be created in _create_control_area)
        self.embedded_level_meter = None

        if getattr(self.settings_manager.settings, "show_level_meter", False):
            # Show embedded level meter
            self.show_level_meter()
            # Update checkbox
            if hasattr(self, "level_meter_var"):
                self.level_meter_var.set(True)
            # Force refresh shortly after showing to prevent blank canvas
            if hasattr(self, "embedded_level_meter") and self.embedded_level_meter:
                self.root.after(20, self.embedded_level_meter.refresh)

        # Bind resize events
        self.root.bind("<Configure>", self._on_window_resize)

        # Force initial resize event after window is mapped
        self.root.after(50, self._trigger_initial_resize)

    def show_level_meter(self):
        """Show level meter"""
        # Create embedded level meter if it doesn't exist yet
        if (
            not hasattr(self, "embedded_level_meter")
            or self.embedded_level_meter is None
        ):
            self._create_embedded_level_meter()

        if hasattr(self, "level_meter_frame") and self.level_meter_frame:
            self.level_meter_frame.grid_forget()
            self.level_meter_frame.grid(
                row=0,
                column=1,
                sticky="ns",
                padx=(UIConstants.FRAME_SPACING // 2, UIConstants.FRAME_SPACING),
                pady=UIConstants.FRAME_SPACING,
            )
            self.level_meter_frame.grid_propagate(False)

    def _setup_screen_geometry(self) -> None:
        """Get screen dimensions and calculate window size.

        Determines screen dimensions and calculates appropriate window
        size based on configuration. Supports both percentage-based
        and absolute pixel dimensions.
        """
        self.root.update_idletasks()

        # Get screen dimensions
        self.ui_state.screen_width = self.root.winfo_screenwidth()
        self.ui_state.screen_height = self.root.winfo_screenheight()

        # Calculate window dimensions
        width_pct, height_pct = self.config.ui.is_window_size_percentage

        # Helper function to calculate dimension
        def calc_dimension(config_val, screen_val, is_pct):
            if config_val:
                return int(screen_val * config_val / 100) if is_pct else config_val
            return int(screen_val * UIConstants.DEFAULT_WINDOW_SIZE_RATIO)

        self.ui_state.window_width = calc_dimension(
            self.config.ui.window_width, self.ui_state.screen_width, width_pct
        )
        self.ui_state.window_height = calc_dimension(
            self.config.ui.window_height, self.ui_state.screen_height, height_pct
        )

    def _setup_window(self) -> None:
        """Configure the main window.

        Sets window size, position, title, and appearance properties.
        Handles both fullscreen and windowed modes, centering the
        window on screen when not fullscreen.
        """
        if self.config.ui.fullscreen:
            self.root.attributes("-fullscreen", True)
            self.ui_state.window_width = self.ui_state.screen_width
            self.ui_state.window_height = self.ui_state.screen_height
        else:
            # Try to restore saved geometry
            saved_geometry = self.settings_manager.settings.window_geometry
            if saved_geometry:
                self.root.geometry(saved_geometry)
                # Update state from saved geometry
                parts = saved_geometry.split("+")[0].split("x")
                self.ui_state.window_width = int(parts[0])
                self.ui_state.window_height = int(parts[1])
            else:
                # Set window size
                self.root.geometry(
                    f"{self.ui_state.window_width}x{self.ui_state.window_height}"
                )

                # Center window
                x = (self.ui_state.screen_width - self.ui_state.window_width) // 2
                y = (self.ui_state.screen_height - self.ui_state.window_height) // 2
                self.root.geometry(
                    f"{self.ui_state.window_width}x{self.ui_state.window_height}+{x}+{y}"
                )

        # Set minimum window size
        self.root.minsize(800, 600)

        # Window properties
        self.root.title("Revoxx")
        self.root.resizable(True, True)
        self.root.configure(bg=UIConstants.COLOR_BACKGROUND)

        # Set window icon
        self._set_window_icon()

    def _create_menu(self) -> None:
        """Create the application menu bar.

        Creates a menu bar with File, View, and Help menus.
        """
        self.menubar = tk.Menu(
            self.root,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            borderwidth=0,
        )
        self.root.config(menu=self.menubar)

        # Helper function to create menu commands that look up callbacks at runtime
        def make_menu_cb(callback_name: str, fallback=None):
            """Create a lambda that looks up menu callback at runtime.

            Args:
                callback_name: Name of callback in app_callbacks dict
                fallback: Optional fallback function if callback not found

            Returns:
                Lambda function that performs runtime lookup
            """
            if fallback is None:
                return lambda: (
                    self.app_callbacks[callback_name]()
                    if callback_name in self.app_callbacks
                    else None
                )
            else:
                return lambda: (
                    self.app_callbacks[callback_name]()
                    if callback_name in self.app_callbacks
                    else fallback()
                )

        # File menu
        file_menu = tk.Menu(
            self.menubar,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        self.menubar.add_cascade(label="File", menu=file_menu)

        # Session management
        # Platform-specific accelerator display
        accel_mod = "Cmd" if platform.system() == "Darwin" else "Ctrl"

        file_menu.add_command(
            label="New Session...",
            command=self._new_session_callback,
            accelerator=f"{accel_mod}+N",
        )
        file_menu.add_command(
            label="Open Session...",
            command=self._open_session_callback,
            accelerator=f"{accel_mod}+O",
        )

        # Recent Sessions submenu
        self.recent_menu = tk.Menu(
            file_menu,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        file_menu.add_cascade(label="Recent Sessions", menu=self.recent_menu)
        self._update_recent_sessions_menu()

        file_menu.add_separator()

        # Create Dataset
        file_menu.add_command(
            label="Create Dataset...",
            command=self._create_dataset_callback,
        )

        file_menu.add_separator()

        # Route Quit via app callback if provided to ensure clean shutdown
        file_menu.add_command(
            label="Quit", command=make_menu_cb("quit", self.root.quit), accelerator="Q"
        )

        # Edit menu
        edit_menu = tk.Menu(
            self.menubar,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        self.menubar.add_cascade(label="Edit", menu=edit_menu)

        # Find utterance
        find_accel = "Cmd+F" if platform.system() == "Darwin" else "Ctrl+F"
        edit_menu.add_command(
            label="Find Utterance...",
            command=make_menu_cb("show_find_dialog"),
            accelerator=find_accel,
        )

        edit_menu.add_separator()

        # Delete recording
        delete_accel = "Cmd+D" if platform.system() == "Darwin" else "Ctrl+D"
        edit_menu.add_command(
            label="Delete Recording",
            command=make_menu_cb("delete_recording"),
            accelerator=delete_accel,
        )

        edit_menu.add_separator()

        # Utterance Order
        order_accel = "Cmd+U" if platform.system() == "Darwin" else "Ctrl+U"
        edit_menu.add_command(
            label="Utterance Order...",
            command=make_menu_cb("show_utterance_order"),
            accelerator=order_accel,
        )

        # View menu
        view_menu = tk.Menu(
            self.menubar,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        self.menubar.add_cascade(label="View", menu=view_menu)

        # Session Settings (at the top)
        # Use Cmd-I on macOS, Ctrl-I on Windows/Linux
        accel = "Cmd+I" if tk.sys.platform == "darwin" else "Ctrl+I"
        view_menu.add_command(
            label="Session Settings...",
            command=self._show_session_settings,
            accelerator=accel,
        )

        view_menu.add_separator()

        # Mel Spectrogram checkbutton
        self.mel_spectrogram_var = tk.BooleanVar(
            value=self.config.display.show_spectrogram
        )
        view_menu.add_checkbutton(
            label="Show Mel Spectrogram",
            variable=self.mel_spectrogram_var,
            command=self._toggle_mel_spectrogram_callback,
            accelerator="M",
        )

        # Level Meter checkbutton
        self.level_meter_var = tk.BooleanVar(
            value=getattr(self.settings_manager.settings, "show_level_meter", False)
        )
        view_menu.add_checkbutton(
            label="Show Level Meter",
            variable=self.level_meter_var,
            command=self.toggle_level_meter_callback,
            accelerator="L",
        )

        # Info Panel checkbutton
        self.info_panel_var = tk.BooleanVar(
            value=getattr(self.settings_manager.settings, "show_info_panel", True)
        )
        view_menu.add_checkbutton(
            label="Show Info Panel",
            variable=self.info_panel_var,
            command=self._toggle_info_panel_callback,
            accelerator="I",
        )

        view_menu.add_separator()

        # Monitoring mode checkbutton
        self.monitoring_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(
            label="Monitor Input Levels",
            variable=self.monitoring_var,
            command=self._toggle_monitoring_callback,
            accelerator="O",
        )

        view_menu.add_separator()

        # Fullscreen checkbutton
        self.fullscreen_var = tk.BooleanVar(value=self.config.ui.fullscreen)
        view_menu.add_checkbutton(
            label="Fullscreen",
            variable=self.fullscreen_var,
            command=self._toggle_fullscreen_callback,
            accelerator="F10",
        )

        # Settings menu
        settings_menu = tk.Menu(
            self.menubar,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        self.menubar.add_cascade(label="Settings", menu=settings_menu)

        # Device submenu
        def _call_app(name: str, idx: int) -> None:
            try:
                cb = self.app_callbacks.get(name)
                if cb:
                    cb(idx)
            except Exception:
                pass

        self.audio_devices_menu = AudioDevicesMenuBuilder(
            settings_menu,
            on_select_input=lambda idx: _call_app("set_input_device", idx),
            on_select_output=lambda idx: _call_app("set_output_device", idx),
            on_select_input_channels=lambda m: _call_app(
                "set_input_channel_mapping", m
            ),
            on_select_output_channels=lambda m: _call_app(
                "set_output_channel_mapping", m
            ),
            initial_input_index=self.config.audio.input_device,
            initial_output_index=self.config.audio.output_device,
            initial_input_mapping=getattr(
                self.settings_manager.settings, "input_channel_mapping", None
            ),
            initial_output_mapping=getattr(
                self.settings_manager.settings, "output_channel_mapping", None
            ),
            debug=bool(self.manager_dict.get("debug", False)),
        )

        # Theme submenu
        theme_menu = tk.Menu(
            settings_menu,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        settings_menu.add_cascade(label="Theme", menu=theme_menu)

        # Create radio buttons for each theme
        self.theme_var = tk.StringVar(
            value=getattr(
                self.settings_manager.settings, "theme", ThemePreset.CYAN.value
            )
        )

        for preset_value, theme_name in theme_manager.get_available_themes().items():
            theme_menu.add_radiobutton(
                label=theme_name,
                variable=self.theme_var,
                value=preset_value,
                command=lambda p=preset_value: self.set_theme(p),
            )

        settings_menu.add_separator()

        # Level Meter Preset submenu
        level_meter_menu = tk.Menu(
            settings_menu,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        settings_menu.add_cascade(label="Level Meter Preset", menu=level_meter_menu)

        # Create radio buttons for each preset
        self.level_meter_preset_var = tk.StringVar(
            value=getattr(
                self.settings_manager.settings, "level_meter_preset", "broadcast_ebu"
            )
        )

        for standard in RecordingStandard:
            if standard != RecordingStandard.CUSTOM:  # Skip CUSTOM for now
                level_meter_menu.add_radiobutton(
                    label=get_standard_description(standard),
                    variable=self.level_meter_preset_var,
                    value=standard.value,
                    command=lambda s=standard.value: self.set_level_meter_preset(s),
                )

        # Help menu
        help_menu = tk.Menu(
            self.menubar,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Keyboard Shortcuts",
            command=make_menu_cb("show_help"),
            accelerator=KeyBindings.SHOW_HELP,
        )
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def _show_about(self) -> None:
        """Show about dialog."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About Revoxx")
        about_window.geometry("400x200")
        about_window.resizable(False, False)
        about_window.configure(bg=UIConstants.COLOR_BACKGROUND)

        about_text = """REVOXX RECORDER

A tool for recording
high-quality speech datasets"""

        label = tk.Label(
            about_window,
            text=about_text,
            justify=tk.CENTER,
            padx=20,
            pady=20,
            bg=UIConstants.COLOR_BACKGROUND,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            font=(UIConstants.FONT_FAMILY_MONO[0], 12),
        )
        label.pack(fill=tk.BOTH, expand=True)

        close_btn = tk.Button(
            about_window,
            text="CLOSE",
            command=about_window.destroy,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_ACCENT,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            bd=1,
            highlightbackground=UIConstants.COLOR_BORDER,
            font=(UIConstants.FONT_FAMILY_MONO[0], 10),
        )
        close_btn.pack(pady=10)

        about_window.focus_set()

    def _toggle_mel_spectrogram_callback(self) -> None:
        """Callback for menu toggle mel spectrogram."""
        # Use app callback if available, otherwise just toggle locally
        if "toggle_mel_spectrogram" in self.app_callbacks:
            self.app_callbacks["toggle_mel_spectrogram"]()
        else:
            self.toggle_spectrogram()

    def toggle_level_meter_callback(self) -> None:
        """Callback for menu toggle level meter."""
        # Toggle embedded level meter
        show_meter = self.level_meter_var.get()
        if show_meter:
            self.show_level_meter()
            # Force a redraw to avoid white area on re-show
            if hasattr(self, "embedded_level_meter") and self.embedded_level_meter:
                self.root.after(10, self.embedded_level_meter.refresh)
        else:
            # Hide embedded level meter
            if hasattr(self, "level_meter_frame") and self.level_meter_frame:
                self.level_meter_frame.grid_forget()

        # Update settings
        self.settings_manager.update_setting(
            "show_level_meter", self.level_meter_var.get()
        )

    def _toggle_info_panel_callback(self) -> None:
        """Toggle the combined info panel visibility."""
        if hasattr(self, "info_panel"):
            # Toggle visibility
            if self.info_panel.winfo_viewable():
                self.info_panel.grid_forget()
                self.info_panel_visible = False
            else:
                # Grid info panel in row 3 (bottom position)
                self.info_panel.grid(
                    row=3, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
                )
                self.info_panel_visible = True
                # Update content if callback available
                if "update_info_panel" in self.app_callbacks:
                    self.root.after(10, self.app_callbacks["update_info_panel"])

            # Update settings
            self.settings_manager.update_setting(
                "show_info_panel", self.info_panel_visible
            )

    def _toggle_monitoring_callback(self) -> None:
        """Callback for menu toggle monitoring mode."""
        # Use app callback if available
        if "toggle_monitoring" in self.app_callbacks:
            self.app_callbacks["toggle_monitoring"]()

    def _toggle_fullscreen_callback(self) -> None:
        """Callback for menu toggle fullscreen."""
        self.toggle_fullscreen()

    def _show_session_settings(self) -> None:
        """Show the session settings dialog."""
        # Get current session from app callback
        if "get_current_session" in self.app_callbacks:
            session = self.app_callbacks["get_current_session"]()
            if session:
                dialog = SessionSettingsDialog(self.root, session)
                dialog.show()
            else:
                tk.messagebox.showwarning(
                    "No Session", "No session is currently loaded.", parent=self.root
                )
        else:
            tk.messagebox.showerror(
                "Error", "Session information is not available.", parent=self.root
            )

    def _set_window_icon(self) -> None:
        """Set the window icon.

        Creates a custom microphone icon for the application window.
        Falls back to a simple icon if the detailed one fails.
        """
        try:
            # Try to create the icon
            icon_path = Path(__file__).parent.parent / "resources" / "microphone.png"
            icon = AppIcon.create_icon(icon_path)
            if icon:
                self.root.iconphoto(True, icon)
                # For macOS dock icon
                self.root.wm_iconphoto(True, icon)
        except Exception as e:
            # Icon setting failed, but that's okay
            if self.manager_dict.get("debug", False):
                print(f"Could not set window icon: {e}")

    def _create_ui(self) -> None:
        """Create the UI elements.

        Creates the three-panel layout with appropriate spacing and padding.
        Initializes all UI widgets and applies initial font sizes.
        """
        # Main container
        self.main_frame = tk.Frame(self.root, bg=UIConstants.COLOR_BACKGROUND)
        self.main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=UIConstants.MAIN_FRAME_PADDING,
            pady=UIConstants.MAIN_FRAME_PADDING,
        )

        # Configure grid layout for main_frame
        self.main_frame.grid_rowconfigure(0, weight=0)  # Info Bar - fixed height
        self.main_frame.grid_rowconfigure(1, weight=1)  # Utterance - expands
        self.main_frame.grid_rowconfigure(2, weight=0)  # Spektrogramm - fixed height
        self.main_frame.grid_rowconfigure(3, weight=0)  # Info Panel - fixed height
        self.main_frame.grid_columnconfigure(0, weight=1)  # Full width

        # Top info bar
        self._create_info_bar()

        # Create middle area for utterance text
        self._create_utterance_display()

        # Spectrogram and level meter (now in row 2)
        self._create_spectrogram_area()

        # Create combined info panel at the bottom (now in row 3)
        self._create_combined_info_panel()

        # Calculate initial font sizes
        self._calculate_font_sizes()

        # Apply fonts
        self._apply_fonts()

        # Force update of all widgets
        self.root.update_idletasks()

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

        # Progress info
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

        # Create container for spectrogram and level meter
        self.spec_container = tk.Frame(
            self.main_frame,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            height=height,
            highlightthickness=0,
        )
        self.spec_container.grid(
            row=2, column=0, sticky="ew", pady=(UIConstants.FRAME_SPACING, 0)
        )
        self.spec_container.grid_propagate(False)  # Critical: prevent size changes

        # Configure grid layout for spec_container children
        self.spec_container.grid_columnconfigure(0, weight=1)  # spec_frame expands
        self.spec_container.grid_columnconfigure(1, weight=0)  # level_meter fixed width
        self.spec_container.grid_rowconfigure(0, weight=1)

        # Always create spectrogram widget, but hide if not enabled
        self._create_spectrogram_widget()

        # Create level meter widget
        self._create_embedded_level_meter()

        # Hide if not enabled in settings
        if not self.config.display.show_spectrogram:
            self.spec_frame.grid_forget()
            self.ui_state.spectrogram_visible = False

    def _create_spectrogram_widget(self) -> None:
        """Create the mel spectrogram widget.

        Creates a frame and initializes the MelSpectrogramWidget for
        real-time audio visualization.
        """
        # Create frame for spectrogram with padding to contain it
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

        # Defer creation of mel_spectrogram widget until window is ready
        # This avoids the negative figure size error
        self.mel_spectrogram = None
        self.root.after(100, self._create_mel_spectrogram_widget_deferred)

        self.ui_state.spectrogram_visible = True

    def _create_mel_spectrogram_widget_deferred(self) -> None:
        """Create the mel spectrogram widget after window is properly sized."""
        if not hasattr(self, "spec_frame"):
            return

        # Ensure frame has proper size
        self.root.update_idletasks()

        # Only create if not already created and frame has size
        if self.mel_spectrogram is None:
            if (
                self.spec_frame.winfo_width() > 10
                and self.spec_frame.winfo_height() > 10
            ):
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

        # Get mono font
        mono_font = UIConstants.FONT_FAMILY_MONO[0]
        try:
            if mono_font not in tkfont.families():
                mono_font = "Courier"
        except Exception:
            mono_font = "Courier"

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
        # Don't pack yet - will be managed by toggle methods

        # Create LED level meter
        if self.shared_audio_state:
            self.embedded_level_meter = LEDLevelMeter(
                self.level_meter_frame, self.shared_audio_state
            )
            self.embedded_level_meter.pack(fill=tk.BOTH, expand=True)

            # Apply saved preset from settings
            preset_str = getattr(
                self.settings_manager.settings, "level_meter_preset", "broadcast_ebu"
            )
            self.root.after(100, lambda: self.set_level_meter_preset(preset_str))
        else:
            self.embedded_level_meter = None

    def _calculate_font_sizes(self) -> None:
        """Calculate dynamic font sizes based on window dimensions.

        Scales fonts proportionally to window size to maintain
        readability at different resolutions. Uses a base font size
        from configuration and applies scaling factors.
        """
        # Scale factor based on window size
        scale_factor = min(
            self.ui_state.window_width / 1200, self.ui_state.window_height / 900
        )

        self.ui_state.calculate_font_sizes(self.config.ui.base_font_size, scale_factor)

    def _apply_fonts(self) -> None:
        """Apply calculated fonts to widgets.

        Updates all UI elements with appropriately scaled fonts.
        Also adjusts text wrapping width based on window size.
        """
        # Try to use modern fonts with fallbacks
        mono_font = None
        sans_font = None
        for font in UIConstants.FONT_FAMILY_MONO:
            try:
                if font in tkfont.families():
                    mono_font = font
                    break
            except Exception:
                pass
        if not mono_font:
            mono_font = "Courier"

        for font in UIConstants.FONT_FAMILY_SANS:
            try:
                if font in tkfont.families():
                    sans_font = font
                    break
            except Exception:
                pass
        if not sans_font:
            sans_font = "Helvetica"

        # Large font for main text
        self.text_display.config(
            font=(sans_font, self.ui_state.font_size_large, "normal"),
            wraplength=int(self.ui_state.window_width * UIConstants.TEXT_WRAP_RATIO),
        )

        # Fixed font sizes for top panel
        self.status_label.config(font=(mono_font, 24))
        self.rec_indicator.config(font=(mono_font, 24, "bold"))
        self.progress_label.config(font=(mono_font, 24, "bold"))

    def _on_window_resize(self, event: tk.Event) -> None:
        """Handle window resize events.

        Recalculates and applies font sizes when the window is resized
        to maintain correct text scaling.

        Args:
            event: Tkinter resize event
        """
        if event.widget == self.root:
            # Update window dimensions
            self.ui_state.window_width = event.width
            self.ui_state.window_height = event.height

            # Recalculate and apply fonts
            self._calculate_font_sizes()
            self._apply_fonts()

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
            self.text_var.set(self.recording_state.utterances[index])
            # Show progress
            self.progress_var.set(
                f"{display_position}/{len(self.recording_state.utterances)}"
            )

        # Update recording indicator
        if is_recording:
            self.text_display.config(fg=UIConstants.COLOR_TEXT_RECORDING)
            self.rec_indicator.config(fg=UIConstants.COLOR_TEXT_RECORDING)
            self.status_var.set("RECORDING...")
        else:
            self.text_display.config(fg=UIConstants.COLOR_TEXT_NORMAL)
            self.rec_indicator.config(fg=UIConstants.COLOR_TEXT_INACTIVE)
            self.status_var.set("")  # Clear status when not recording

    def set_status(self, message: str) -> None:
        """Set status message.

        Args:
            message: Status text to display in the info bar
        """
        self.status_var.set(message)

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

        # Always hide center message and ensure labels are visible
        self.info_center_message.place_forget()

        # Ensure labels are packed if not visible
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
            # Clear file info and duration
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

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode.

        Switches between fullscreen and windowed modes, adjusting
        window dimensions and recalculating font sizes accordingly.
        Saves window position before going fullscreen and restores
        it when exiting fullscreen.
        """
        current = self.root.attributes("-fullscreen")

        if not current:
            self._enter_fullscreen()
        else:
            self._exit_fullscreen()

        # Update menu checkbutton
        if hasattr(self, "fullscreen_var"):
            self.fullscreen_var.set(not current)

        # Save preference
        self.settings_manager.update_setting("fullscreen", not current)

    def _enter_fullscreen(self) -> None:
        """Enter fullscreen mode.

        Saves current window geometry and switches to fullscreen.
        """
        # Save current geometry
        self.ui_state.saved_window_geometry = self.root.geometry()

        # Enter fullscreen
        self.root.attributes("-fullscreen", True)

        # Update window dimensions
        self.ui_state.window_width = self.ui_state.screen_width
        self.ui_state.window_height = self.ui_state.screen_height

        # Trigger resize event
        self._trigger_resize_event()

    def _exit_fullscreen(self) -> None:
        """Exit fullscreen mode.

        Restores saved window geometry or applies default positioning.
        Uses withdraw/deiconify to prevent visual artifacts.
        """
        # Hide window during transition to prevent black window flash
        self.root.withdraw()

        # Exit fullscreen
        self.root.attributes("-fullscreen", False)

        # Restore geometry
        self._restore_window_geometry()

        # Force window update and show window again
        self.root.update_idletasks()
        self.root.deiconify()

        # Trigger resize event
        self._trigger_resize_event()

    def _restore_window_geometry(self) -> None:
        """Restore window geometry after exiting fullscreen.

        Uses saved geometry if available, otherwise centers window
        with default dimensions.
        """
        if self.ui_state.saved_window_geometry:
            # Restore saved position and size
            self.root.geometry(self.ui_state.saved_window_geometry)
            # Update window dimensions from saved geometry
            parts = self.ui_state.saved_window_geometry.split("+")[0].split("x")
            self.ui_state.window_width = int(parts[0])
            self.ui_state.window_height = int(parts[1])
        else:
            # Fallback to default size centered
            self.ui_state.window_width = int(
                self.ui_state.screen_width * UIConstants.DEFAULT_WINDOW_SIZE_RATIO
            )
            self.ui_state.window_height = int(
                self.ui_state.screen_height * UIConstants.DEFAULT_WINDOW_SIZE_RATIO
            )
            # Center window
            x = (self.ui_state.screen_width - self.ui_state.window_width) // 2
            y = (self.ui_state.screen_height - self.ui_state.window_height) // 2
            self.root.geometry(
                f"{self.ui_state.window_width}x{self.ui_state.window_height}+{x}+{y}"
            )

    def _trigger_resize_event(self) -> None:
        """Trigger a resize event to update fonts and layout."""
        event = tk.Event()
        event.widget = self.root
        event.width = self.ui_state.window_width
        event.height = self.ui_state.window_height
        self._on_window_resize(event)

    def _trigger_initial_resize(self) -> None:
        """Trigger initial resize after window is fully mapped."""
        # Update window dimensions from actual window
        self.root.update_idletasks()
        self.ui_state.window_width = self.root.winfo_width()
        self.ui_state.window_height = self.root.winfo_height()
        # Trigger resize event
        self._trigger_resize_event()

    def toggle_spectrogram(
        self, update_external_state: Optional[Callable] = None
    ) -> None:
        """Toggle mel spectrogram visibility.

        Shows or hides the mel spectrogram widget in the control area.
        Updates UI state and displays a status message.

        Args:
            update_external_state: Optional callback to update external state
        """
        if hasattr(self, "spec_frame") and self.spec_frame:
            if self.spec_frame.winfo_viewable():
                # Hide spectrogram
                self.spec_frame.grid_forget()
                self.ui_state.spectrogram_visible = False
            else:
                # Show spectrogram with proper padding
                self.spec_frame.grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                    padx=(UIConstants.FRAME_SPACING, UIConstants.FRAME_SPACING // 2),
                    pady=UIConstants.FRAME_SPACING,
                )
                self.ui_state.spectrogram_visible = True

                # Create mel_spectrogram widget if it doesn't exist yet
                if self.mel_spectrogram is None and hasattr(
                    self, "_mel_spectrogram_pending"
                ):
                    # Ensure the frame has a size before creating the widget
                    self.root.update_idletasks()

                    # Only create if frame has reasonable size
                    if (
                        self.spec_frame.winfo_width() > 10
                        and self.spec_frame.winfo_height() > 10
                    ):
                        self.mel_spectrogram = MelSpectrogramWidget(
                            self.spec_frame,
                            self.config.audio,
                            self.config.display,
                            self.manager_dict,
                            self.shared_audio_state,
                        )
                        self._mel_spectrogram_pending = False

                # Force redraw to avoid white display
                if hasattr(self, "mel_spectrogram") and self.mel_spectrogram:
                    self.root.update_idletasks()
                    self.mel_spectrogram.canvas.draw_idle()

        # Update menu checkbutton
        if hasattr(self, "mel_spectrogram_var"):
            self.mel_spectrogram_var.set(self.ui_state.spectrogram_visible)

        # Call external state update if provided
        if update_external_state:
            update_external_state()

    def show_message(self, message: str, duration: int = 2000) -> None:
        """Show a temporary message.

        Displays a message in the main text area temporarily,
        then restores the normal display.

        Args:
            message: Message text to display
            duration: Display duration in milliseconds (default: 2000)
        """
        self.text_var.set(message)
        self.root.after(duration, self._restore_display)

    def _restore_display(self) -> None:
        """Restore normal display after temporary message.

        Called automatically after show_message() timeout to restore
        the current utterance display.
        """
        self.update_display(
            self.recording_state.current_index,
            self.recording_state.is_recording,
            self.recording_state.display_position,
        )

    def focus_window(self) -> None:
        """Bring window to front and focus.

        Ensures the window is visible and has keyboard focus.
        Uses platform-specific techniques for reliable focus,
        especially on macOS.
        """
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(
            UIConstants.FOCUS_DELAY_MS, lambda: self.root.attributes("-topmost", False)
        )
        self.root.focus_force()

        # Platform-specific focus
        if platform.system() == "Darwin":  # macOS
            self.root.after(UIConstants.FOCUS_DELAY_MS, lambda: self.root.focus_force())

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

        # Update the menu variable
        if hasattr(self, "level_meter_preset_var"):
            self.level_meter_preset_var.set(preset_name)

    def set_theme(self, theme_preset: str) -> None:
        """Set the application theme.

        Args:
            theme_preset: Theme preset name (e.g., 'classic', 'modern')
        """
        try:
            # Convert string to ThemePreset enum
            preset = ThemePreset(theme_preset)
            theme_manager.set_theme(preset)

            # Refresh UIConstants with new theme colors
            UIConstants.refresh()

            # Save the setting
            self.settings_manager.update_setting("theme", theme_preset)

            # Update the menu variable
            if hasattr(self, "theme_var"):
                self.theme_var.set(theme_preset)

            # Apply theme to all widgets
            self._apply_theme_to_widgets()

            # Redraw everything
            self._force_redraw()

            # Show message to user
            self.set_status(f"Theme changed to {theme_manager.current_theme.name}")
        except ValueError:
            print(f"Unknown theme preset: {theme_preset}")

    def _apply_theme_to_widgets(self) -> None:
        """Apply current theme colors to all widgets."""
        # Update main window background
        self.root.configure(bg=UIConstants.COLOR_BACKGROUND)

        # Update main frame
        if hasattr(self, "main_frame"):
            self.main_frame.configure(bg=UIConstants.COLOR_BACKGROUND)

        # Update info frame
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

    def _force_redraw(self) -> None:
        """Force redraw of all components."""
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
        self.root.update_idletasks()

    # Info overlay methods removed - using info_panel instead

    def _new_session_callback(self) -> None:
        """Handle New Session menu item."""
        # Use app callback if available
        if "new_session" in self.app_callbacks:
            self.app_callbacks["new_session"]()
        else:
            self.set_status("New Session dialog not yet implemented")

    def _open_session_callback(self) -> None:
        """Handle Open Session menu item."""
        # Use app callback if available
        if "open_session" in self.app_callbacks:
            self.app_callbacks["open_session"]()
        else:
            self.set_status("Open Session dialog not yet implemented")

    def _create_dataset_callback(self):
        """Show dataset creation dialog."""
        # Get base directory from settings or use default
        base_dir = getattr(
            self.settings_manager.settings,
            "base_sessions_dir",
            Path.home() / "revoxx_sessions",
        )

        # If we're in a session, use its parent directory
        if (
            hasattr(self, "app_callbacks")
            and "get_current_session" in self.app_callbacks
        ):
            current_session = self.app_callbacks["get_current_session"]()
            if current_session and current_session.session_dir:
                base_dir = current_session.session_dir.parent

        dialog = DatasetDialog(self.root, base_dir, self.settings_manager)
        result = dialog.show()

        if result:
            # Dataset(s) were created successfully
            if isinstance(result, list):
                if len(result) == 1:
                    self.set_status(f"Dataset created: {result[0].name}")
                else:
                    self.set_status(f"{len(result)} datasets created")
            else:
                self.set_status(f"Dataset created: {result.name}")

    def _open_recent_session(self, session_path: Path) -> None:
        """Handle opening a recent session.

        Args:
            session_path: Path to the session directory
        """
        if "open_recent_session" in self.app_callbacks:
            self.app_callbacks["open_recent_session"](session_path)
        else:
            self.set_status(f"Opening: {session_path.name}")

    def _update_recent_sessions_menu(self) -> None:
        """Update the Recent Sessions submenu."""
        # Clear existing items
        self.recent_menu.delete(0, tk.END)

        # Get recent sessions from app if available
        if "get_recent_sessions" in self.app_callbacks:
            recent_sessions = self.app_callbacks["get_recent_sessions"]()

            if recent_sessions:
                for session_path in recent_sessions[:10]:  # Max 10 recent sessions
                    session_name = (
                        session_path.name
                        if isinstance(session_path, Path)
                        else str(session_path)
                    )
                    self.recent_menu.add_command(
                        label=session_name,
                        command=lambda p=session_path: self._open_recent_session(p),
                    )
            else:
                self.recent_menu.add_command(
                    label="(No recent sessions)", state=tk.DISABLED
                )
        else:
            self.recent_menu.add_command(
                label="(No recent sessions)", state=tk.DISABLED
            )

    def update_session_title(self, session_name: str = None) -> None:
        """Update window title with session name.

        Args:
            session_name: Name of the current session, or None for default title
        """
        if session_name:
            self.root.title(f"Revoxx - {session_name}")
        else:
            self.root.title("Revoxx")
