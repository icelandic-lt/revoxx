"""Application menu for Revoxx.

This module provides the main application menu bar, handling all menu creation
and menu-specific callbacks through the application's controllers.
"""

import tkinter as tk
from tkinter import messagebox
import platform
from typing import TYPE_CHECKING
from pathlib import Path

from ...constants import UIConstants, KeyBindings
from ..themes import theme_manager, ThemePreset
from ..level_meter.config import RecordingStandard, get_standard_description
from ..dialogs.dataset_dialog import DatasetDialog
from .audio_devices import AudioDevicesMenuBuilder
from ...utils.device_manager import get_device_manager

if TYPE_CHECKING:
    from ...app import Revoxx


class ApplicationMenu:
    """Main application menu bar.

    This class creates and manages the complete menu system for Revoxx,
    delegating actions to appropriate controllers rather than directly
    accessing window components.
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the application menu.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self.root = app.window.window
        self.menubar = None

        # Menu variables - centrally managed
        self.menu_vars = {
            "meters": tk.BooleanVar(value=app.settings_manager.settings.show_meters),
            "info_panel": tk.BooleanVar(
                value=app.settings_manager.settings.show_info_panel
            ),
            "monitoring": tk.BooleanVar(value=False),
            "fullscreen": tk.BooleanVar(value=app.config.ui.fullscreen),
            "theme": tk.StringVar(
                value=getattr(
                    app.settings_manager.settings, "theme", ThemePreset.CYAN.value
                )
            ),
            "level_meter_preset": tk.StringVar(
                value=getattr(
                    app.settings_manager.settings, "level_meter_preset", "broadcast_ebu"
                )
            ),
        }

        # Menu references for dynamic updates
        self.recent_menu = None
        self.second_window_menu = None
        self.second_window_menu_indices = {}

        # Add variables for monitor1 (second window)
        monitor1_enabled = False
        monitor1_meters = False
        monitor1_info = True

        # Check if monitor1 settings exist
        if (
            app.settings_manager.settings.windows
            and "monitor1" in app.settings_manager.settings.windows
        ):
            monitor1_settings = app.settings_manager.settings.windows["monitor1"]
            monitor1_enabled = monitor1_settings.get("enabled", False)
            monitor1_meters = monitor1_settings.get("meters_visible", False)
            monitor1_info = monitor1_settings.get("info_panel_visible", True)

        self.menu_vars["second_window"] = tk.BooleanVar(value=monitor1_enabled)
        self.menu_vars["second_window_meters"] = tk.BooleanVar(value=monitor1_meters)
        self.menu_vars["second_window_info"] = tk.BooleanVar(value=monitor1_info)

    def create_menu(self) -> None:
        """Create the complete application menu bar."""
        self.menubar = tk.Menu(
            self.root,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            borderwidth=0,
        )
        self.root.config(menu=self.menubar)

        self._create_file_menu()
        self._create_edit_menu()
        self._create_view_menu()
        self._create_settings_menu()
        self._create_help_menu()

    def _create_file_menu(self) -> None:
        """Create the File menu."""
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

        # Platform-specific accelerator display
        accel_mod = "Cmd" if platform.system() == "Darwin" else "Ctrl"

        # Session management
        file_menu.add_command(
            label="New Session...",
            command=self._new_session,
            accelerator=f"{accel_mod}+N",
        )
        file_menu.add_command(
            label="Open Session...",
            command=self._open_session,
            accelerator=f"{accel_mod}+O",
        )

        file_menu.add_separator()

        # Import text to script
        file_menu.add_command(
            label="Import Text to Script...",
            command=self._import_text_to_script,
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
        self.update_recent_sessions()

        file_menu.add_separator()

        # Create Dataset
        file_menu.add_command(
            label="Create Dataset...",
            command=self._create_dataset,
        )

        file_menu.add_separator()

        # Quit
        file_menu.add_command(label="Quit", command=self.app._quit, accelerator="Q")

    def _create_edit_menu(self) -> None:
        """Create the Edit menu."""
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
            command=self.app.dialog_controller.show_find_dialog,
            accelerator=find_accel,
        )

        edit_menu.add_separator()

        # Delete recording
        delete_accel = "Cmd+D" if platform.system() == "Darwin" else "Ctrl+D"
        edit_menu.add_command(
            label="Delete Recording",
            command=self.app.file_operations_controller.delete_current_recording,
            accelerator=delete_accel,
        )

        edit_menu.add_separator()

        # Utterance Order
        order_accel = "Cmd+U" if platform.system() == "Darwin" else "Ctrl+U"
        edit_menu.add_command(
            label="Utterance Order...",
            command=self.app.dialog_controller.show_utterance_order_dialog,
            accelerator=order_accel,
        )

    def _create_view_menu(self) -> None:
        """Create the View menu."""
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

        # Session Settings
        accel = "Cmd+I" if platform.system() == "Darwin" else "Ctrl+I"
        view_menu.add_command(
            label="Session Settings...",
            command=self._show_session_settings,
            accelerator=accel,
        )

        view_menu.add_separator()

        # Meters checkbox
        view_menu.add_checkbutton(
            label="Show Mel Spectrogram & Level Meter",
            variable=self.menu_vars["meters"],
            command=self._toggle_meters,
            accelerator="M",
        )

        # Info Panel checkbox
        view_menu.add_checkbutton(
            label="Show Info Panel",
            variable=self.menu_vars["info_panel"],
            command=self._toggle_info_panel,
            accelerator="I",
        )

        view_menu.add_separator()

        # Monitoring mode checkbox
        view_menu.add_checkbutton(
            label="Monitor Input Levels",
            variable=self.menu_vars["monitoring"],
            command=self._toggle_monitoring,
            accelerator="O",
        )

        view_menu.add_separator()

        # Fullscreen checkbox
        view_menu.add_checkbutton(
            label="Fullscreen",
            variable=self.menu_vars["fullscreen"],
            command=self._toggle_fullscreen,
            accelerator="F10",
        )

    def _create_settings_menu(self) -> None:
        """Create the Settings menu."""
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

        # 2nd Window submenu
        self._create_second_window_menu(settings_menu)

        settings_menu.add_separator()

        # Audio Devices submenu
        self._create_audio_devices_menu(settings_menu)

        # Theme submenu
        self._create_theme_menu(settings_menu)

        settings_menu.add_separator()

        # Level Meter Preset submenu
        self._create_level_meter_menu(settings_menu)

    def _create_second_window_menu(self, parent_menu: tk.Menu) -> None:
        """Create the 2nd Window submenu."""
        self.second_window_menu = tk.Menu(
            parent_menu,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        parent_menu.add_cascade(label="2nd Window", menu=self.second_window_menu)

        # Enable checkbox
        self.second_window_menu.add_checkbutton(
            label="Enable 2nd Window",
            variable=self.menu_vars["second_window"],
            command=self._toggle_second_window,
        )

        self.second_window_menu.add_separator()

        # Meters checkbox
        self.second_window_menu_indices["meters"] = (
            self.second_window_menu.index(tk.END) + 1
        )
        self.second_window_menu.add_checkbutton(
            label="Show Mel Spectrogram & Level Meter",
            variable=self.menu_vars["second_window_meters"],
            command=self._update_second_window_config,
            state="disabled" if not self.menu_vars["second_window"].get() else "normal",
            accelerator="Shift+M",
        )

        # Info panel checkbox
        self.second_window_menu_indices["info"] = (
            self.second_window_menu.index(tk.END) + 1
        )
        self.second_window_menu.add_checkbutton(
            label="Show Info Panel",
            variable=self.menu_vars["second_window_info"],
            command=self._update_second_window_config,
            state="disabled" if not self.menu_vars["second_window"].get() else "normal",
            accelerator="Shift+I",
        )

        self.second_window_menu.add_separator()

        # Fullscreen command
        self.second_window_menu_indices["fullscreen"] = (
            self.second_window_menu.index(tk.END) + 1
        )
        self.second_window_menu.add_command(
            label="Toggle Fullscreen",
            command=self._toggle_second_window_fullscreen,
            accelerator="Shift+F10",
            state="disabled" if not self.menu_vars["second_window"].get() else "normal",
        )

    def _create_audio_devices_menu(self, parent_menu: tk.Menu) -> None:
        """Create the Audio Devices submenu."""

        def _call_controller(method_name: str, *args):
            """Helper to call device controller methods."""
            method = getattr(self.app.device_controller, method_name, None)
            if method:
                method(*args)

        # Convert device names to indices for menu initialization
        device_manager = get_device_manager()

        initial_input_index = None
        if self.app.config.audio.input_device:
            initial_input_index = device_manager.get_device_index_by_name(
                self.app.config.audio.input_device
            )

        initial_output_index = None
        if self.app.config.audio.output_device:
            initial_output_index = device_manager.get_device_index_by_name(
                self.app.config.audio.output_device
            )

        def _on_rescan_devices():
            """Send refresh commands to audio processes when devices are rescanned."""
            if self.app.queue_manager:
                self.app.queue_manager.refresh_playback_devices()
                self.app.queue_manager.refresh_record_devices()

        self.audio_devices_menu = AudioDevicesMenuBuilder(
            parent_menu,
            on_select_input=lambda idx: _call_controller("set_input_device", idx),
            on_select_output=lambda idx: _call_controller("set_output_device", idx),
            on_select_input_channels=lambda m: _call_controller(
                "set_input_channel_mapping", m
            ),
            on_select_output_channels=lambda m: _call_controller(
                "set_output_channel_mapping", m
            ),
            on_rescan_devices=_on_rescan_devices,
            initial_input_index=initial_input_index,
            initial_output_index=initial_output_index,
            initial_input_mapping=getattr(
                self.app.settings_manager.settings, "input_channel_mapping", None
            ),
            initial_output_mapping=getattr(
                self.app.settings_manager.settings, "output_channel_mapping", None
            ),
            debug=self.app.debug,
        )

    def _create_theme_menu(self, parent_menu: tk.Menu) -> None:
        """Create the Theme submenu."""
        theme_menu = tk.Menu(
            parent_menu,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        parent_menu.add_cascade(label="Theme", menu=theme_menu)

        for preset_value, theme_name in theme_manager.get_available_themes().items():
            theme_menu.add_radiobutton(
                label=theme_name,
                variable=self.menu_vars["theme"],
                value=preset_value,
                command=lambda p=preset_value: self._set_theme(p),
            )

    def _create_level_meter_menu(self, parent_menu: tk.Menu) -> None:
        """Create the Level Meter Preset submenu."""
        level_meter_menu = tk.Menu(
            parent_menu,
            tearoff=0,
            bg=UIConstants.COLOR_BACKGROUND_SECONDARY,
            fg=UIConstants.COLOR_TEXT_NORMAL,
            activebackground=UIConstants.COLOR_ACCENT,
            activeforeground=UIConstants.COLOR_BACKGROUND,
            selectcolor=UIConstants.COLOR_ACCENT,
        )
        parent_menu.add_cascade(label="Level Meter Preset", menu=level_meter_menu)

        for standard in RecordingStandard:
            if standard != RecordingStandard.CUSTOM:
                level_meter_menu.add_radiobutton(
                    label=get_standard_description(standard),
                    variable=self.menu_vars["level_meter_preset"],
                    value=standard.value,
                    command=lambda s=standard.value: self._set_level_meter_preset(s),
                )

    def _create_help_menu(self) -> None:
        """Create the Help menu."""
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
            command=self.app.dialog_controller.show_help,
            accelerator=KeyBindings.SHOW_HELP,
        )
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    # ============= Menu Callbacks =============

    def _new_session(self) -> None:
        """Handle New Session menu item."""
        self.app._new_session()

    def _open_session(self) -> None:
        """Handle Open Session menu item."""
        self.app._open_session()

    def _create_dataset(self) -> None:
        """Show dataset creation dialog."""
        base_dir = getattr(
            self.app.settings_manager.settings,
            "base_sessions_dir",
            Path.home() / "revoxx_sessions",
        )

        if self.app.current_session and self.app.current_session.session_dir:
            base_dir = self.app.current_session.session_dir.parent

        dialog = DatasetDialog(self.root, base_dir, self.app.settings_manager)
        result = dialog.show()

        if result:
            if isinstance(result, list):
                if len(result) == 1:
                    self.app.display_controller.set_status(
                        f"Dataset created: {result[0].name}"
                    )
                else:
                    self.app.display_controller.set_status(
                        f"{len(result)} datasets created"
                    )
            else:
                self.app.display_controller.set_status(
                    f"Dataset created: {result.name}"
                )

    def _show_session_settings(self) -> None:
        """Show the session settings dialog."""
        if self.app.current_session:
            from ..dialogs import SessionSettingsDialog

            dialog = SessionSettingsDialog(self.root, self.app.current_session)
            dialog.show()
        else:
            messagebox.showwarning(
                "No Session", "No session is currently loaded.", parent=self.root
            )

    def _toggle_meters(self) -> None:
        """Toggle meters visibility."""
        self.app.display_controller.toggle_meters()
        self.app.settings_manager.update_setting(
            "show_meters", self.menu_vars["meters"].get()
        )

    def _toggle_info_panel(self) -> None:
        """Toggle info panel visibility."""
        new_state = self.app.display_controller.toggle_info_panel()
        self.app.settings_manager.update_setting("show_info_panel", new_state)

    def _toggle_monitoring(self) -> None:
        """Toggle monitoring mode."""
        self.app.audio_controller.toggle_monitoring()

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        new_state = self.app.display_controller.toggle_fullscreen()
        self.menu_vars["fullscreen"].set(new_state)
        self.app.settings_manager.update_setting("fullscreen", new_state)

    def _toggle_second_window(self) -> None:
        """Toggle second window on/off."""
        enabled = self.menu_vars["second_window"].get()

        if enabled:
            self.app.display_controller.open_window("monitor1")
            # Enable menu items
            for key in ["meters", "info", "fullscreen"]:
                if key in self.second_window_menu_indices:
                    self.second_window_menu.entryconfig(
                        self.second_window_menu_indices[key], state="normal"
                    )
        else:
            self.app.display_controller.close_window("monitor1")
            # Disable menu items
            for key in ["meters", "info", "fullscreen"]:
                if key in self.second_window_menu_indices:
                    self.second_window_menu.entryconfig(
                        self.second_window_menu_indices[key], state="disabled"
                    )

        # Save window enabled state in new windows structure
        self.app.settings_manager.save_window_settings("monitor1", {"enabled": enabled})

    def _update_second_window_config(self) -> None:
        """Update second window configuration."""
        if self.app.has_active_second_window:
            config = {
                "show_meters": self.menu_vars["second_window_meters"].get(),
                "show_info_panel": self.menu_vars["second_window_info"].get(),
            }
            self.app.display_controller.update_window_config("monitor1", config)

            # Save settings
            self.app.settings_manager.save_window_settings(
                "monitor1",
                {
                    "meters_visible": config["show_meters"],
                    "info_panel_visible": config["show_info_panel"],
                },
            )

    def _toggle_second_window_fullscreen(self) -> None:
        """Toggle fullscreen for second window."""
        self.app.display_controller.toggle_window_fullscreen("monitor1")

    def _set_theme(self, theme_preset: str) -> None:
        """Set application theme."""
        self.app.display_controller.set_theme(theme_preset)
        self.app.settings_manager.update_setting("theme", theme_preset)

    def _set_level_meter_preset(self, preset: str) -> None:
        """Set level meter preset."""
        self.app.display_controller.set_level_meter_preset(preset)
        self.app.settings_manager.update_setting("level_meter_preset", preset)

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

    def update_recent_sessions(self) -> None:
        """Update the Recent Sessions submenu."""
        if not self.recent_menu:
            return

        self.recent_menu.delete(0, tk.END)

        recent_sessions = self.app.session_manager.get_recent_sessions()

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

    def _open_recent_session(self, session_path: Path) -> None:
        """Open a recent session."""
        try:
            session = self.app.session_manager.load_session(session_path)
            self.app.current_session = session
            self.app.session_controller.load_session(session)
            self.app.display_controller.update_window_title()
            self.update_recent_sessions()
            self.app.display_controller.set_status(
                f"Loaded session: {session.session_dir.name}"
            )
        except Exception as e:
            self.app.display_controller.set_status(f"Error loading session: {e}")

    def _import_text_to_script(self) -> None:
        """Show the import text to script dialog."""
        from ..dialogs.import_text_dialog import ImportTextDialog

        # Pass settings manager for saving/loading directory preferences
        # The dialog will handle default directories internally
        dialog = ImportTextDialog(
            self.root,
            default_dir=None,  # Let dialog use saved settings or home directory
            settings_manager=self.app.settings_manager,
        )
        result = dialog.show()

        if result:
            self.app.display_controller.set_status(f"Script created: {result.name}")

            # Optionally ask if user wants to create a new session with this script
            response = messagebox.askyesno(
                "Create Session",
                "Script file created successfully.\n\nWould you like to create a new session with this script?",
                parent=self.root,
            )

            if response:
                # Pass the script path to new session dialog
                self.app._new_session(default_script=result)

    def update_menu_states(self, states: dict) -> None:
        """Update menu checkboxes based on current application state.

        Args:
            states: Dictionary of current states
        """
        for key, value in states.items():
            if key in self.menu_vars:
                self.menu_vars[key].set(value)

    def on_second_window_closed(self) -> None:
        """Handle second window being closed by user."""
        self.menu_vars["second_window"].set(False)
        # Disable menu items
        for key in ["meters", "info", "fullscreen"]:
            if key in self.second_window_menu_indices:
                self.second_window_menu.entryconfig(
                    self.second_window_menu_indices[key], state="disabled"
                )
