"""Tkinter compatibility utilities for Tcl/Tk 8.6 and 9.x."""

import tkinter as tk
from tkinter import messagebox


def trace_var_write(var: tk.Variable, callback) -> str:
    """Add a write trace to a Tkinter variable (compatible with Tcl/Tk 8.6 and 9.x).

    In Tcl/Tk 9, the old trace("w", ...) syntax was removed.
    This function uses trace_add("write", ...) if available, falling back to
    trace("w", ...) for older versions.

    Args:
        var: The Tkinter variable (StringVar, IntVar, etc.)
        callback: The callback function to call on write

    Returns:
        The trace ID (can be used with untrace_var_write to remove the trace)
    """
    try:
        # Tcl/Tk 9.x syntax
        return var.trace_add("write", callback)
    except AttributeError:
        # Tcl/Tk 8.6 syntax (trace_add doesn't exist)
        return var.trace("w", callback)
    except tk.TclError:
        # Tcl/Tk 8.6 but trace_add exists with different behavior
        return var.trace("w", callback)


def untrace_var_write(var: tk.Variable, trace_id: str) -> None:
    """Remove a write trace from a Tkinter variable (compatible with Tcl/Tk 8.6 and 9.x).

    Args:
        var: The Tkinter variable
        trace_id: The trace ID returned by trace_var_write
    """
    try:
        # Tcl/Tk 9.x syntax
        var.trace_remove("write", trace_id)
    except (AttributeError, tk.TclError):
        # Tcl/Tk 8.6 syntax
        var.trace_vdelete("w", trace_id)


def deferred_warning(parent: tk.Widget, title: str, message: str) -> None:
    """Show a warning messagebox deferred to the next event loop cycle.

    Use this instead of messagebox.showwarning() when the parent window
    may not be visible yet (e.g. during startup before deiconify).

    Args:
        parent: The parent widget (typically the main window)
        title: Dialog title
        message: Warning message text
    """
    parent.after(0, lambda: messagebox.showwarning(title, message, parent=parent))


def clear_menu(menu: tk.Menu) -> None:
    """Clear all items from a menu, safe for empty menus.

    In some Tcl/Tk versions, menu.delete(0, END) on an empty menu
    raises TclError because menu.index("end") returns "" instead
    of None.

    Args:
        menu: The Tkinter menu to clear
    """
    try:
        menu.delete(0, tk.END)
    except tk.TclError:
        pass
