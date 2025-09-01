"""Window Factory for creating configured windows.

This module implements the factory pattern for window creation,
using configuration templates.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import tkinter as tk

from .window_base import WindowBase
from .icon import AppIcon
from ..utils.config import RecorderConfig
from ..utils.state import UIState, RecordingState
from ..utils.settings_manager import SettingsManager


class WindowFactory:
    """Factory for creating windows with type-based configuration.

    Uses feature flags and configuration templates to customize
    window behavior.
    """

    # Window type templates define feature sets and defaults
    WINDOW_TYPES: Dict[str, Dict[str, Any]] = {
        "main": {
            "features": {
                "has_icon": True,
                "has_menu": True,
                "can_quit_app": True,
                "remember_session": True,
                "is_primary": True,
            },
            "default_visibility": {
                "meters": True,
                "info_panel": True,
            },
            "window_title": "Revoxx",
            "min_size": (800, 600),
        },
        "secondary": {
            "features": {
                "has_icon": False,
                "has_menu": False,
                "can_quit_app": False,
                "remember_session": False,
                "is_primary": False,
            },
            "default_visibility": {
                "meters": False,  # Speaker view typically doesn't need meters
                "info_panel": True,
            },
            "window_title": "Revoxx - Speaker",
            "min_size": (600, 400),
        },
        "standard": {
            # Generic window with minimal features
            "features": {
                "has_icon": False,
                "has_menu": False,
                "can_quit_app": False,
                "remember_session": False,
                "is_primary": False,
            },
            "default_visibility": {
                "meters": True,
                "info_panel": True,
            },
            "window_title": "Revoxx",
            "min_size": (600, 400),
        },
    }

    @classmethod
    def create(
        cls,
        parent: Optional[tk.Widget],
        window_id: str,
        window_type: str,
        config: RecorderConfig,
        recording_state: RecordingState,
        ui_state: UIState,
        manager_dict: dict,
        app_callbacks: dict,
        settings_manager: Optional[SettingsManager],
        shared_audio_state: Any,
    ) -> WindowBase:
        """Create a window with type-based configuration.

        Args:
            parent: Parent widget (None for root window)
            window_id: Unique window identifier
            window_type: Type of window to create
            config: Application configuration
            recording_state: Recording state
            ui_state: UI state
            manager_dict: Shared state dictionary
            app_callbacks: Application callbacks
            settings_manager: Settings manager
            shared_audio_state: Shared audio state

        Returns:
            Configured WindowBase instance
        """
        # Get template for window type
        template = cls.WINDOW_TYPES.get(window_type, cls.WINDOW_TYPES["standard"])

        window = WindowBase(
            parent=parent,
            window_id=window_id,
            features=template["features"].copy(),
            config=config,
            recording_state=recording_state,
            ui_state=ui_state,
            manager_dict=manager_dict,
            app_callbacks=app_callbacks,
            settings_manager=settings_manager,
            shared_audio_state=shared_audio_state,
        )

        cls._configure_window(window, template)
        cls._setup_window_ui(window)
        cls._apply_initial_visibility(window, template)
        cls._setup_window_features(window)

        return window

    @classmethod
    def _configure_window(cls, window: WindowBase, template: dict) -> None:
        """Configure basic window properties.

        Args:
            window: Window to configure
            template: Configuration template
        """
        window.window.title(template["window_title"])

        min_width, min_height = template["min_size"]
        window.window.minsize(min_width, min_height)

        if window._is_root:
            # Root window gets larger default size
            screen_width = window.window.winfo_screenwidth()
            screen_height = window.window.winfo_screenheight()

            # Use 80% of screen for main window
            width = int(screen_width * 0.8)
            height = int(screen_height * 0.8)

            # Center on screen
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            window.window.geometry(f"{width}x{height}+{x}+{y}")
        else:
            # Secondary windows get smaller size with offset
            width, height = 800, 600
            window.window.geometry(f"{width}x{height}+100+100")

    @classmethod
    def _setup_window_ui(cls, window: WindowBase) -> None:
        """Setup the window's UI components.

        Args:
            window: Window to setup
        """
        from ..constants import UIConstants

        window.main_frame = tk.Frame(window.window, bg=UIConstants.COLOR_BACKGROUND)
        window.main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=0,
            pady=0,
        )

        window.main_frame.grid_rowconfigure(0, weight=0)  # Info Bar
        window.main_frame.grid_rowconfigure(1, weight=1)  # Utterance
        window.main_frame.grid_rowconfigure(2, weight=0)  # Spectrogram
        window.main_frame.grid_rowconfigure(3, weight=0)  # Info Panel
        window.main_frame.grid_columnconfigure(0, weight=1)

        window._create_info_bar()
        window._create_utterance_display()
        window._create_spectrogram_area()
        window._create_combined_info_panel()

        # Mark that fonts haven't been initialized yet
        window._fonts_initialized = False

    @classmethod
    def _apply_initial_visibility(cls, window: WindowBase, template: dict) -> None:
        """Apply initial panel visibility from template.

        Args:
            window: Window to configure
            template: Configuration template
        """
        visibility = template.get("default_visibility", {})

        # Apply meters visibility
        show_meters = visibility.get("meters", True)
        if not show_meters:
            window.set_meters_visibility(False)

        # Apply info panel visibility
        show_info = visibility.get("info_panel", True)
        if not show_info and hasattr(window, "info_panel"):
            window.info_panel.grid_forget()
            window.info_panel_visible = False

    @classmethod
    def _setup_window_features(cls, window: WindowBase) -> None:
        """Setup window-specific features based on feature flags.

        Args:
            window: Window to setup
        """
        if window.features.get("has_icon"):
            cls._set_window_icon(window)

        if window.features.get("can_quit_app"):
            # Main window quits the app
            window.window.protocol(
                "WM_DELETE_WINDOW", lambda: cls._handle_main_close(window)
            )
        else:
            # Secondary windows just close themselves
            window.window.protocol(
                "WM_DELETE_WINDOW", lambda: cls._handle_window_close(window)
            )

        # Bind resize event
        window.window.bind("<Configure>", lambda e: cls._on_window_resize(window, e))

    @classmethod
    def _set_window_icon(cls, window: WindowBase) -> None:
        """Set window icon for platforms that support it.

        Args:
            window: Window to set icon for
        """
        icon_path = None

        # Try development path first
        dev_path = Path(__file__).parent.parent / "resources" / "microphone.png"
        if dev_path.exists():
            icon_path = dev_path
        else:
            # Try installed package (Python 3.9+ required)
            try:
                from importlib.resources import files

                resource_path = files("revoxx.resources") / "microphone.png"
                # Convert to Path object for consistency
                icon_path = Path(str(resource_path))
            except (ImportError, AttributeError, FileNotFoundError, TypeError):
                # importlib.resources not available or resource not found
                pass

        # Set icon if we found it
        if icon_path and icon_path.exists():
            icon = AppIcon.create_icon(icon_path)
            if icon:
                try:
                    window.window.iconphoto(True, icon)
                except tk.TclError:
                    # Some platforms don't support iconphoto
                    pass

    @classmethod
    def _handle_main_close(cls, window: WindowBase) -> None:
        """Handle main window close (quits application).

        Args:
            window: Main window being closed
        """
        # This will be connected to app's quit method when integrated
        if window.app_callbacks and "quit" in window.app_callbacks:
            window.app_callbacks["quit"]()
        else:
            # Fallback: just destroy window
            window.window.quit()

    @classmethod
    def _handle_window_close(cls, window: WindowBase) -> None:
        """Handle secondary window close.

        Args:
            window: Window being closed
        """
        window.is_active = False

        # Save geometry if settings manager available
        if window.settings_manager:
            try:
                geometry = window.window.geometry()
                window.settings_manager.update_setting(
                    f"{window.window_id}_geometry", geometry
                )
            except Exception:
                pass

        # Destroy window
        try:
            window.window.destroy()
        except Exception:
            pass

    @classmethod
    def _on_window_resize(cls, window: WindowBase, event: tk.Event) -> None:
        """Handle window resize events.

        Args:
            window: Window that was resized
            event: Tkinter resize event
        """
        # Only process resize events for the actual window
        if event.widget != window.window:
            return

        window.font_manager.calculate_base_sizes(event.width, event.height)
        window._update_fixed_ui_fonts()

        # Debounce text font recalculation
        if hasattr(window, "_resize_timer"):
            window.window.after_cancel(window._resize_timer)
        if hasattr(window, "text_var") and window.text_var.get():
            window._invalidate_layout_cache()
            window._resize_timer = window.window.after(
                150, lambda: window.refresh_text_layout()
            )

        # Fire FontsReady event on first resize
        if hasattr(window, "_fonts_initialized") and not window._fonts_initialized:
            window._fonts_initialized = True
            window.window.event_generate("<<FontsReady>>")

        # Save geometry if settings available
        if window.settings_manager and window.features.get("remember_session"):
            try:
                geometry = window.window.geometry()
                window.settings_manager.update_setting(
                    f"{window.window_id}_geometry", geometry
                )
            except Exception:
                pass
