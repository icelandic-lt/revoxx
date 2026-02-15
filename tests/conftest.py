"""Pytest configuration and shared fixtures."""

import os
import warnings


def _check_tcl_library():
    """Warn if TCL_LIBRARY is not set and Tcl/Tk initialization will likely fail.

    Standalone Python distributions (e.g. installed by uv) may have
    build-time Tcl paths compiled in that don't exist on the host.
    Tests that create tk.Tk() will fail with:

        TclError: Cannot find a usable init.tcl

    Fix: set TCL_LIBRARY to point to the installed Tcl library, e.g.:

        export TCL_LIBRARY=/opt/homebrew/opt/tcl-tk/lib/tcl9.0

    In PyCharm: Run > Edit Configurations > Environment variables.
    """
    if "TCL_LIBRARY" in os.environ:
        return

    try:
        import tkinter as tk

        root = tk.Tk()
        root.destroy()
    except Exception:
        warnings.warn(
            "TCL_LIBRARY is not set and tk.Tk() initialization failed. "
            "Tests requiring Tkinter will fail. "
            "Set TCL_LIBRARY to your Tcl installation, e.g.: "
            "export TCL_LIBRARY=/opt/homebrew/opt/tcl-tk/lib/tcl9.0",
            stacklevel=1,
        )


_check_tcl_library()
