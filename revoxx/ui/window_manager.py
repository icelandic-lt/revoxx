"""Window Manager for multi-window support in Revoxx.

This module provides centralized window management for N-screen support
with dynamic window creation and broadcasting capabilities.
"""

from collections import OrderedDict
from typing import Dict, Optional, List, Any, Callable, Union
import tkinter as tk

from .window_factory import WindowFactory
from .window_base import WindowBase


class WindowManager:
    """Central manager for all application windows.

    Manages creation, tracking, and broadcasting to N windows dynamically.
    All windows are treated according to their configuration settings.
    """

    # Default configurations for window types
    DEFAULT_CONFIGS = {
        "main": {
            "enabled": True,  # Statically enabled
            "has_menu": True,
            "has_icon": True,
            "can_quit_app": True,
            "min_size": (1024, 768),
            "default_meters": True,
            "default_info": True,
        },
        # Template for all monitor windows
        "_default": {
            "enabled": False,
            "has_menu": False,
            "has_icon": False,
            "can_quit_app": False,
            "min_size": (800, 600),
            "default_meters": False,
            "default_info": True,
        },
    }

    def __init__(self, app):
        """Initialize WindowManager.

        Args:
            app: Application instance with config and settings
        """
        self.app = app
        self.windows: OrderedDict[str, WindowBase] = OrderedDict()
        self.window_configs: Dict[str, dict] = {}

    def get_window_config(self, window_id: str) -> dict:
        """Get configuration for a window.

        Args:
            window_id: Window identifier

        Returns:
            Configuration dictionary for the window
        """
        if window_id in self.window_configs:
            return self.window_configs[window_id]

        if window_id == "main":
            return self.DEFAULT_CONFIGS["main"].copy()
        else:
            return self.DEFAULT_CONFIGS["_default"].copy()

    def create_window(
        self,
        window_id: Optional[str] = None,
        parent: Optional[tk.Widget] = None,
        window_type: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> WindowBase:
        """Create and register a new window.

        Args:
            window_id: Unique identifier (auto-generated if None)
            parent: Parent widget (auto-determined if None)
            window_type: Type for factory ('main', 'secondary')
            config: Configuration override (uses defaults if None)

        Returns:
            Created WindowBase instance
        """
        # Auto-generate ID if not provided
        if window_id is None:
            window_id = self._generate_next_id()

        window_config = config or self.get_window_config(window_id)
        self.window_configs[window_id] = window_config

        # Determine parent & window type if not provided
        if parent is None and window_id != "main":
            root = self.get_root_window()
            parent = root.window if root else None

        if window_type is None:
            window_type = "main" if window_id == "main" else "secondary"

        window = WindowFactory.create(
            parent=parent,
            window_id=window_id,
            window_type=window_type,
            config=self.app.config,
            recording_state=self.app.state.recording,
            ui_state=self.app.state.ui,
            manager_dict=self.app.manager_dict,
            app_callbacks=self._get_app_callbacks(),
            settings_manager=self.app.settings_manager,
            shared_audio_state=getattr(self.app, "shared_state", None),
        )

        self._apply_config(window, window_config)
        self.windows[window_id] = window
        self._restore_window_position(window)

        return window

    def _generate_next_id(self) -> str:
        """Generate next available window ID.

        Returns:
            Next available ID like 'monitor1', 'monitor2', etc.
        """
        i = 1
        while f"monitor{i}" in self.windows:
            i += 1
        return f"monitor{i}"

    @staticmethod
    def _apply_config(window: WindowBase, config: dict) -> None:
        """Apply configuration to window.

        Args:
            window: Window to configure
            config: Configuration dictionary
        """
        # Apply initial visibility settings
        if not config.get("default_meters", True):
            window.set_meters_visibility(False)

        if not config.get("default_info", True) and hasattr(window, "info_panel"):
            window.info_panel.grid_forget()
            window.info_panel_visible = False

    def get_root_window(self) -> Optional[WindowBase]:
        """Get the main Tk root window.

        Returns:
            Main window or None if not created yet
        """
        return self.windows.get("main")

    def get_window(self, window_id: str) -> Optional[WindowBase]:
        """Get window by ID.

        Args:
            window_id: Window identifier

        Returns:
            Window instance or None if not found
        """
        return self.windows.get(window_id)

    def get_active_windows(self) -> List[WindowBase]:
        """Get all currently active windows.

        Returns:
            List of active window instances
        """
        return [w for w in self.windows.values() if w.is_active]

    def broadcast(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call method on all active windows.

        Args:
            method_name: Name of method to call
            *args: Positional arguments for method
            **kwargs: Keyword arguments for method

        Returns:
            List of results from each window
        """
        results = []
        for window in self.get_active_windows():
            if hasattr(window, method_name):
                try:
                    method = getattr(window, method_name)
                    results.append(method(*args, **kwargs))
                except tk.TclError:
                    pass  # Window might be closed
        return results

    def execute_on_windows(
        self, window_ids: Union[str, List[str]], method_name: str, *args, **kwargs
    ) -> List[Any]:
        """Execute method on specific windows.

        Args:
            window_ids: Single ID or list of window IDs
            method_name: Method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of results from each window
        """
        if isinstance(window_ids, str):
            window_ids = [window_ids]

        results = []
        for wid in window_ids:
            window = self.windows.get(wid)
            if window and window.is_active:
                method = getattr(window, method_name, None)
                if method and callable(method):
                    try:
                        results.append(method(*args, **kwargs))
                    except tk.TclError:
                        pass
        return results

    def for_each_window(self, action: Callable[[WindowBase], None]) -> None:
        """Execute action on each active window.

        Args:
            action: Function to call with each window
        """
        for window in self.get_active_windows():
            try:
                action(window)
            except tk.TclError:
                pass  # Window might be closed

    def close_window(self, window_id: str) -> None:
        """Close and unregister a window.

        Args:
            window_id: ID of window to close
        """
        if window_id == "main":
            return  # Cannot close main window this way

        window = self.windows.get(window_id)
        if not window:
            return

        # Save final state
        self._save_window_state(window_id)

        # Mark as inactive
        window.is_active = False

        # Destroy window
        try:
            window.window.destroy()
        except tk.TclError:
            pass

        # Remove from registry
        del self.windows[window_id]

    def restore_saved_windows(self) -> None:
        """Restore windows that were previously enabled."""
        if not self.app.settings_manager:
            return

        settings = self.app.settings_manager.settings

        # Check for windows structure
        if hasattr(settings, "windows"):
            for window_id, window_settings in settings.windows.items():
                if window_id != "main" and window_settings.get("enabled", False):
                    self._restore_window(window_id, window_settings)

    def _restore_window(self, window_id: str, window_settings: dict) -> None:
        """Restore a specific window from settings.

        Args:
            window_id: Window identifier
            window_settings: Saved settings for the window
        """
        # Create window
        window = self.create_window(window_id)

        # Apply saved visibility settings
        if "meters_visible" in window_settings:
            window.set_meters_visibility(window_settings["meters_visible"])

        if "info_panel_visible" in window_settings:
            if window_settings["info_panel_visible"] and hasattr(window, "info_panel"):
                if not window.info_panel_visible:
                    window.info_panel.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
                    window.info_panel_visible = True
            elif hasattr(window, "info_panel"):
                window.info_panel.grid_forget()
                window.info_panel_visible = False

        # Apply fullscreen if saved
        if window_settings.get("fullscreen", False):

            def apply_fullscreen():
                window.window.update_idletasks()
                window.window.attributes("-fullscreen", True)

            window.window.after(500, apply_fullscreen)

    def _restore_window_position(self, window: WindowBase) -> None:
        """Restore window position from settings.

        Args:
            window: Window to position
        """
        if not self.app.settings_manager:
            return

        settings = self.app.settings_manager.settings
        window_id = window.window_id

        # Get geometry from windows structure
        if hasattr(settings, "windows") and window_id in settings.windows:
            geometry = settings.windows[window_id].get("geometry")
        else:
            geometry = None

        if geometry:
            try:
                window.window.geometry(geometry)
                window.window.update_idletasks()
            except tk.TclError:
                pass

    def _save_window_state(self, window_id: str) -> None:
        """Save window state to settings.

        Args:
            window_id: Window identifier
        """
        window = self.windows.get(window_id)
        if not window or not self.app.settings_manager:
            return

        try:
            geometry = window.window.geometry()
            is_fullscreen = window.window.attributes("-fullscreen")
            meters_visible = window.meters_visible
            info_visible = window.info_panel_visible

            # Save to windows structure
            window_settings = {
                "geometry": geometry,
                "fullscreen": is_fullscreen,
                "meters_visible": meters_visible,
                "info_panel_visible": info_visible,
                "enabled": True,  # Was open, so enabled
            }
            self.app.settings_manager.save_window_settings(window_id, window_settings)
        except (tk.TclError, AttributeError):
            pass

    def save_all_positions(self) -> None:
        """Save positions and states of all windows."""
        for window_id in self.windows:
            self._save_window_state(window_id)

    def focus_main_window(self) -> None:
        """Set focus back to the main window."""
        main_window = self.windows.get("main")
        if main_window and main_window.is_active:
            try:
                main_window.window.focus_set()
                main_window.window.lift()
            except tk.TclError:
                pass

    def _get_app_callbacks(self) -> dict:
        """Get callbacks dictionary for windows.

        Returns:
            Dictionary of application callbacks
        """
        return getattr(self.app, "app_callbacks", {})
