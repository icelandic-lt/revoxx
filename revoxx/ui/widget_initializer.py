"""Widget initialization helper for event-based initialization."""

import tkinter as tk
from typing import Callable, Optional


class WidgetInitializer:
    """Helper class for event-based widget initialization.

    Eliminates repetitive event binding/unbinding code for widget initialization.
    """

    @staticmethod
    def when_ready(
        widget: tk.Widget,
        callback: Callable,
        check_condition: Optional[Callable[[], bool]] = None,
        event_type: str = "<Configure>",
        min_width: int = 1,
        min_height: int = 1,
    ) -> None:
        """Execute callback when widget is ready.

        Args:
            widget: Widget to monitor
            callback: Function to call when ready
            check_condition: Optional custom readiness check
            event_type: Event to bind to (default: <Configure>)
            min_width: Minimum width to consider ready
            min_height: Minimum height to consider ready
        """
        # Default condition checks for minimum dimensions
        if check_condition is None:

            def default_check():
                return (
                    widget.winfo_width() > min_width
                    and widget.winfo_height() > min_height
                )

            check_condition = default_check

        # Check without forcing update first
        if check_condition():
            # Already ready, execute immediately
            callback()
        else:
            # Force update only if first check failed
            widget.update_idletasks()

            if check_condition():
                # Now ready after update
                callback()
            else:
                # Wait for event
                def on_event(event):
                    if check_condition():
                        widget.unbind(event_type, binding_id)
                        callback()

                binding_id = widget.bind(event_type, on_event)

    @staticmethod
    def when_mapped(widget: tk.Widget, callback: Callable) -> None:
        """Execute callback when widget is mapped to screen.

        Args:
            widget: Widget to monitor
            callback: Function to call when mapped
        """
        if widget.winfo_viewable():
            callback()
        else:

            def on_map(event):
                if event.widget == widget:
                    widget.unbind("<Map>", map_binding_id)
                    callback()

            map_binding_id = widget.bind("<Map>", on_map)

    @staticmethod
    def when_configured(
        widget: tk.Widget,
        callback: Callable[[tk.Event], None],
        min_width: int = 1,
        min_height: int = 1,
        once: bool = True,
    ) -> Optional[str]:
        """Execute callback when widget is configured with minimum dimensions.

        Args:
            widget: Widget to monitor
            callback: Function to call with event
            min_width: Minimum width required
            min_height: Minimum height required
            once: If True, unbind after first call

        Returns:
            Binding ID if event was bound, None if executed immediately
        """
        # Check if already configured without forcing update
        if widget.winfo_width() > min_width and widget.winfo_height() > min_height:
            # Create fake event for immediate execution
            fake_event = type(
                "Event",
                (),
                {
                    "widget": widget,
                    "width": widget.winfo_width(),
                    "height": widget.winfo_height(),
                },
            )()
            callback(fake_event)
            return None

        # Bind to Configure event
        binding_id = None

        def on_configure(event):
            nonlocal binding_id
            if (
                event.widget == widget
                and event.width > min_width
                and event.height > min_height
            ):
                if once and binding_id:
                    widget.unbind("<Configure>", binding_id)
                    binding_id = None
                callback(event)

        binding_id = widget.bind("<Configure>", on_configure)
        return binding_id

    @staticmethod
    def fire_when_all_ready(
        window: tk.Tk, event_name: str, *widgets: tk.Widget, min_dimensions: int = 1
    ) -> None:
        """Fire custom event when all widgets are ready.

        Args:
            window: Window to fire event on
            event_name: Name of custom event to fire
            *widgets: Widgets that must be ready
            min_dimensions: Minimum width/height for readiness
        """
        pending = list(widgets)

        def check_widget(widget):
            if (
                widget.winfo_width() > min_dimensions
                and widget.winfo_height() > min_dimensions
            ):
                if widget in pending:
                    pending.remove(widget)
                if not pending:
                    window.event_generate(event_name)

        for widget in widgets:
            WidgetInitializer.when_ready(
                widget,
                lambda w=widget: check_widget(w),
                min_width=min_dimensions,
                min_height=min_dimensions,
            )
