"""Window Manager for multi-window support in Revoxx.

This module provides centralized window management with simple error handling
and position restoration.
"""

from typing import Dict, Optional
import tkinter as tk

from .window_factory import WindowFactory
from .window_base import WindowBase


class WindowManager:
    """Central manager for all application windows.

    Manages creation, tracking, and broadcasting to windows.
    Features simple fallback strategies without complex monitor detection.
    """

    def __init__(self, app):
        """Initialize WindowManager.

        Args:
            app: Application instance with config and settings
        """
        self.app = app
        self.windows: Dict[str, WindowBase] = {}

    def restore_saved_windows(self) -> None:
        """Restore any windows that were enabled in settings."""
        if self.app.settings_manager.settings.second_window_enabled:
            second = self.create_window(
                window_id="second",
                parent=(
                    self.windows.get("main").window if "main" in self.windows else None
                ),
                window_type="secondary",
            )

            if hasattr(self.app, "menu") and self.app.menu:
                if self.app.menu.menu_vars.get("second_window_meters"):
                    show_meters = self.app.menu.menu_vars["second_window_meters"].get()
                    second.set_meters_visibility(show_meters)

                if self.app.menu.menu_vars.get("second_window_info"):
                    show_info = self.app.menu.menu_vars["second_window_info"].get()
                    if show_info and hasattr(second, "info_panel"):
                        if not second.info_panel_visible:
                            second.info_panel.grid(
                                row=3, column=0, sticky="ew", padx=5, pady=2
                            )
                            second.info_panel_visible = True
                    elif not show_info and hasattr(second, "info_panel"):
                        second.info_panel.grid_forget()
                        second.info_panel_visible = False

            if self.app.settings_manager.settings.second_fullscreen:
                # First the geometry is restored, then we apply fullscreen on the correct monitor
                def apply_fullscreen():
                    second.window.update_idletasks()
                    second.window.attributes("-fullscreen", True)

                second.window.after(500, apply_fullscreen)

            # Return focus to main window after restoration
            self.focus_main_window()

    def create_window(
        self,
        window_id: str,
        parent: Optional[tk.Widget] = None,
        window_type: str = "standard",
    ) -> WindowBase:
        """Create and register a new window.

        Args:
            window_id: Unique identifier for this window
            parent: Parent widget (None for root window)
            window_type: Type of window ('main', 'secondary', etc.)

        Returns:
            Created WindowBase instance
        """
        # Use factory to create configured window
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

        # Register window
        self.windows[window_id] = window

        # Try to restore saved position
        self._restore_window_position(window)

        return window

    def _get_app_callbacks(self) -> dict:
        """Get callbacks dictionary for windows.

        Returns:
            Dictionary of application callbacks
        """
        # Return app callbacks if they exist, otherwise empty dict
        return getattr(self.app, "app_callbacks", {})

    def _restore_window_position(self, window: WindowBase) -> None:
        """Restore window position from settings with simple fallback.

        Args:
            window: Window to position
        """
        if not self.app.settings_manager:
            return

        settings = self.app.settings_manager.settings
        geometry_key = f"{window.window_id}_geometry"

        saved_geometry = getattr(settings, geometry_key, None)
        if not saved_geometry:
            return

        try:
            # Try to apply saved geometry
            window.window.geometry(saved_geometry)
            window.window.update_idletasks()
            # Trust the saved coordinates for multi-monitor setups
        except Exception:
            # Geometry restoration failed, let window use default position
            pass

    def get_window(self, window_id: str) -> Optional[WindowBase]:
        """Get window by ID.

        Args:
            window_id: Window identifier

        Returns:
            Window instance or None if not found
        """
        return self.windows.get(window_id)

    def get_active_windows(self) -> list[WindowBase]:
        """Get all currently active windows.

        Returns:
            List of active window instances
        """
        return [w for w in self.windows.values() if w.is_active]

    def broadcast(self, method_name: str, *args, **kwargs):
        """Call method on all active windows.

        Args:
            method_name: Name of method to call
            *args: Positional arguments for method
            **kwargs: Keyword arguments for method
        """
        for window in self.get_active_windows():
            if hasattr(window, method_name):
                method = getattr(window, method_name)
                try:
                    method(*args, **kwargs)
                except Exception:
                    # Window might have been closed, ignore errors
                    pass

    def close_window(self, window_id: str) -> None:
        """Close and unregister a window.

        Args:
            window_id: ID of window to close
        """
        window = self.windows.get(window_id)
        if not window:
            return

        # Mark as inactive
        window.is_active = False

        if self.app.settings_manager:
            try:
                is_fullscreen = window.window.attributes("-fullscreen")
                geometry = window.window.geometry()

                # Update settings in memory
                setattr(
                    self.app.settings_manager.settings,
                    f"{window_id}_geometry",
                    geometry,
                )
                setattr(
                    self.app.settings_manager.settings,
                    f"{window_id}_fullscreen",
                    is_fullscreen,
                )
            except Exception:
                # Failed to save window state, ignore
                pass

        # Destroy window
        try:
            window.window.destroy()
        except Exception:
            pass

        # Remove from registry
        del self.windows[window_id]

    def save_all_positions(self) -> None:
        """Save positions of all windows to settings."""
        if not self.app.settings_manager:
            return

        for window_id, window in self.windows.items():
            if window.is_active:
                try:
                    geometry = window.window.geometry()
                    is_fullscreen = window.window.attributes("-fullscreen")

                    # Update settings in memory only (not saved to disk yet)
                    setattr(
                        self.app.settings_manager.settings,
                        f"{window_id}_geometry",
                        geometry,
                    )
                    setattr(
                        self.app.settings_manager.settings,
                        f"{window_id}_fullscreen",
                        is_fullscreen,
                    )
                except Exception:
                    # Failed to save position, ignore
                    pass

    def focus_main_window(self) -> None:
        """Set focus back to the main window."""
        main_window = self.windows.get("main")
        if main_window and main_window.is_active:
            try:
                main_window.window.focus_set()
                main_window.window.lift()
            except Exception:
                # Focus setting failed, ignore
                pass
