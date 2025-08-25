"""Utility functions for dialog positioning and management."""

import tkinter as tk
from typing import Tuple, Callable, Optional


def get_dialog_size(dialog: tk.Toplevel) -> Tuple[int, int]:
    """Extract dialog size from geometry string or fall back to requested size.

    Note: Withdrawn windows in Tkinter often report geometry as "1x1" even after
    setting a different size. This function detects this case and falls back to
    winfo_reqwidth/reqheight which gives the requested (i.e. intended) size.

    Args:
        dialog: The dialog window

    Returns:
        Tuple of (width, height) in pixels
    """
    geometry = dialog.geometry()

    if "x" in geometry:
        # Try to extract width and height from geometry string
        size_part = geometry.split("+")[0] if "+" in geometry else geometry
        width, height = map(int, size_part.split("x"))

        # Check if size is valid (1x1 is the default for withdrawn windows)
        if width == 1 and height == 1:
            # Fallback to requested size
            width = dialog.winfo_reqwidth()
            height = dialog.winfo_reqheight()
        else:
            pass  # Size extracted from geometry
    else:
        # No valid geometry, use requested size
        width = dialog.winfo_reqwidth()
        height = dialog.winfo_reqheight()

    return width, height


def get_parent_bounds(parent: tk.Tk) -> Tuple[int, int, int, int]:
    """Get parent window position and size.

    Args:
        parent: The parent window

    Returns:
        Tuple of (x, y, width, height)
    """
    return (
        parent.winfo_x(),
        parent.winfo_y(),
        parent.winfo_width(),
        parent.winfo_height(),
    )


def calculate_centered_position(
    parent_bounds: Tuple[int, int, int, int], dialog_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Calculate centered position for dialog within parent bounds.

    Args:
        parent_bounds: Parent (x, y, width, height)
        dialog_size: Dialog (width, height)

    Returns:
        Tuple of (x, y) for dialog position
    """
    parent_x, parent_y, parent_width, parent_height = parent_bounds
    dialog_width, dialog_height = dialog_size

    # Calculate center position
    x = parent_x + (parent_width - dialog_width) // 2
    y = parent_y + (parent_height - dialog_height) // 2

    return x, y


def constrain_to_parent(
    position: Tuple[int, int],
    dialog_size: Tuple[int, int],
    parent_bounds: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    """Constrain dialog position to stay within parent window bounds.

    Args:
        position: Proposed (x, y) position
        dialog_size: Dialog (width, height)
        parent_bounds: Parent (x, y, width, height)

    Returns:
        Adjusted (x, y) position within parent bounds
    """
    x, y = position
    dialog_width, dialog_height = dialog_size
    parent_x, parent_y, parent_width, parent_height = parent_bounds

    # Constrain horizontally
    if x < parent_x:
        x = parent_x
    elif x + dialog_width > parent_x + parent_width:
        x = parent_x + parent_width - dialog_width
        if x < parent_x:  # Dialog wider than parent
            x = parent_x

    # Constrain vertically
    if y < parent_y:
        y = parent_y
    elif y + dialog_height > parent_y + parent_height:
        y = parent_y + parent_height - dialog_height
        if y < parent_y:  # Dialog taller than parent
            y = parent_y

    return x, y


def center_dialog_on_parent(
    dialog: tk.Toplevel, parent: tk.Tk, explicit_size: Optional[Tuple[int, int]] = None
) -> None:
    """Center a dialog window on its parent window.

    This function centers a dialog by calculating the position
    for the top-left corner, taking into account both the parent window's
    position and the dialog's own dimensions. The dialog will be
    constrained to not extend beyond the parent window bounds.

    Args:
        dialog: The dialog window to center
        parent: The parent window to center on
        explicit_size: Optional explicit (width, height) to use instead of detected size
    """
    # Ensure both dialog and parent dimensions are calculated
    parent.update_idletasks()
    dialog.update_idletasks()

    # Get parent bounds
    parent_bounds = get_parent_bounds(parent)
    parent_x, parent_y, parent_width, parent_height = parent_bounds

    # Get dialog size - use explicit size if provided
    if explicit_size:
        dialog_width, dialog_height = explicit_size
    else:
        dialog_width, dialog_height = get_dialog_size(dialog)

    # Calculate centered position
    ideal_x, ideal_y = calculate_centered_position(
        parent_bounds, (dialog_width, dialog_height)
    )

    # Constrain to parent bounds
    x, y = constrain_to_parent(
        (ideal_x, ideal_y), (dialog_width, dialog_height), parent_bounds
    )

    # Ensure on screen (may adjust position if dialog would be off-screen)
    x, y = ensure_on_screen(dialog, x, y, dialog_width, dialog_height)

    # Re-apply parent constraints (parent bounds have priority over screen bounds)
    # This ensures that if ensure_on_screen moved the dialog outside parent bounds,
    # we move it back to stay within parent (even if partially off-screen)
    x, y = constrain_to_parent((x, y), (dialog_width, dialog_height), parent_bounds)

    # Apply position and size
    dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")


def center_dialog_on_screen(dialog: tk.Toplevel) -> None:
    """Center a dialog window on the screen.

    Args:
        dialog: The dialog window to center
    """
    # Ensure dialog dimensions are calculated
    dialog.update_idletasks()

    # Get screen dimensions
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()

    # Get dialog size - use reqwidth/reqheight for withdrawn windows
    dialog_width = dialog.winfo_reqwidth()
    dialog_height = dialog.winfo_reqheight()

    # Calculate center position for the dialog's top-left corner
    x = (screen_width - dialog_width) // 2
    y = (screen_height - dialog_height) // 2

    # Apply the position
    dialog.geometry(f"+{x}+{y}")


def ensure_on_screen(
    dialog: tk.Toplevel, x: int, y: int, dialog_width: int, dialog_height: int
) -> Tuple[int, int]:
    """Ensure a dialog position keeps the window fully on screen.

    Args:
        dialog: The dialog window (used to get screen dimensions)
        x: Proposed x position
        y: Proposed y position
        dialog_width: Dialog width in pixels
        dialog_height: Dialog height in pixels

    Returns:
        Adjusted (x, y) position that keeps dialog on screen
    """
    # Get screen dimensions
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()

    # Adjust x position if needed
    if x < 0:
        x = 0
    elif x + dialog_width > screen_width:
        x = screen_width - dialog_width

    # Adjust y position if needed
    if y < 0:
        y = 0
    elif y + dialog_height > screen_height:
        y = screen_height - dialog_height

    return x, y


def setup_dialog_window(
    dialog: tk.Toplevel,
    parent: tk.Tk,
    title: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    center_on_parent: bool = True,
    size_callback: Optional[Callable[[], Tuple[int, int]]] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
) -> None:
    """Set up a dialog window with common settings.

    This function handles the common setup pattern:
    1. Set title
    2. Hide window initially to prevent flashing
    3. Make transient to parent
    4. Set minimum size if specified
    5. Call size_callback if provided to determine size
    6. Set size
    7. Center the dialog
    8. Show the dialog

    This order is important to prevent sudden "blitzes" when constructing the dialog.

    Args:
        dialog: The dialog window to set up
        parent: The parent window
        title: Dialog title
        width: Dialog width in pixels (ignored if size_callback provided)
        height: Dialog height in pixels (ignored if size_callback provided)
        center_on_parent: If True, center on parent; if False, center on screen
        size_callback: Optional callback that returns (width, height) tuple.
                      Called after widgets are created but before showing
        min_width: Optional minimum width
        min_height: Optional minimum height
    """
    # Set title
    dialog.title(title)

    # Hide window initially to prevent flashing
    dialog.withdraw()

    # Make dialog transient to parent
    dialog.transient(parent)

    # Set minimum size if specified
    if min_width is not None and min_height is not None:
        dialog.minsize(min_width, min_height)

    # Determine size - either from callback or parameters
    if size_callback:
        # Callback can access already created widgets
        width, height = size_callback()
    elif width is None or height is None:
        raise ValueError("Either provide width/height or size_callback")

    # Set size
    dialog.geometry(f"{width}x{height}")

    # Update to ensure dimensions are calculated
    dialog.update_idletasks()

    # Center the dialog
    # Note: We pass explicit size because withdrawn windows report incorrect geometry (1x1)
    # even after setting the size. The explicit_size parameter ensures correct centering.
    if center_on_parent:
        center_dialog_on_parent(dialog, parent, explicit_size=(width, height))
    else:
        center_dialog_on_screen(dialog)

    # Show the dialog
    dialog.deiconify()

    # Grab focus
    dialog.grab_set()
