"""Dialog controller for managing UI dialogs and user interactions."""

from typing import Optional, TYPE_CHECKING
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from ..ui.dialogs import NewSessionDialog
from ..ui.dialogs.find_dialog import FindDialog
from ..ui.dialogs.help_dialog import HelpDialog
from ..ui.dialogs.open_session_dialog import OpenSessionDialog
from ..ui.dialogs.utterance_order_dialog import UtteranceOrderDialog
from ..ui.dialogs.utterance_list_base import SortDirection
from ..utils.device_manager import get_device_manager

if TYPE_CHECKING:
    from ..app import Revoxx


class DialogController:
    """Handles dialog interactions and user prompts.

    This controller manages:
    - File dialogs (open/save)
    - Message dialogs
    - Find dialog
    - Utterance order dialog
    - Settings dialog
    - Device selection dialog
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the dialog controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self.find_dialog = None
        self.order_dialog = None
        self.settings_dialog = None
        self.device_dialog = None

    def show_open_session_dialog(self) -> Optional[Path]:
        """Show dialog to open a session directory.

        Returns:
            Selected directory path or None if cancelled
        """
        initial_dir = self.app.session_manager.get_default_base_dir()
        dialog = OpenSessionDialog(self.app.window.window, initial_dir)
        return dialog.show()

    def show_new_session_dialog(self, default_script=None):
        """Show dialog to create a new session.

        Args:
            default_script: Optional path to a script file to use for the new session

        Returns:
            Dialog result object with all session parameters or None if cancelled
        """
        default_base_dir = None
        if self.app.current_session:
            default_base_dir = self.app.current_session.session_dir.parent
        else:
            # Try to get from settings
            default_base_dir = self.app.session_manager.get_default_base_dir()

        if not default_base_dir:
            default_base_dir = Path.cwd()  # Fallback to current working directory

        dialog = NewSessionDialog(
            self.app.window.window,
            default_base_dir,
            self.app.config.audio.sample_rate,
            self.app.config.audio.bit_depth,
            self.app.config.audio.input_device,
            default_script=default_script,
        )
        return dialog.show()

    def show_export_dialog(self) -> Optional[Path]:
        """Show dialog to select export directory.

        Returns:
            Selected directory path or None if cancelled
        """
        export_dir = filedialog.askdirectory(
            title="Select Export Directory", parent=self.app.window.window
        )

        if export_dir:
            return Path(export_dir)
        return None

    def show_export_file_dialog(self) -> Optional[Path]:
        """Show dialog to select export file.

        Returns:
            Selected file path or None if cancelled
        """
        export_file = filedialog.asksaveasfilename(
            title="Export Recording",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            parent=self.app.window.window,
        )

        if export_file:
            return Path(export_file)
        return None

    def show_find_dialog(self) -> None:
        """Show the find utterance dialog."""
        if (
            self.find_dialog
            and hasattr(self.find_dialog, "dialog")
            and self.find_dialog.dialog.winfo_exists()
        ):
            self.find_dialog.dialog.lift()
            return

        # Check if we have active recordings
        if not self.app.active_recordings:
            messagebox.showerror(
                "No Session",
                "Please load or create a session first.",
                parent=self.app.window.window,
            )
            return

        # Get sorted indices from active recordings
        sorted_indices = self.app.active_recordings.get_sorted_indices()

        # Convert boolean sort_reverse to SortDirection enum
        sort_direction = (
            SortDirection.DOWN
            if self.app.active_recordings.sort_reverse
            else SortDirection.UP
        )

        self.find_dialog = FindDialog(
            self.app.window.window,
            self.app.state.recording.utterances,
            self.app.state.recording.labels,
            self.app.file_manager,
            self.app.state.recording.current_index,
            self._on_find_select,
            sorted_indices,
            self.app.active_recordings.sort_column,
            sort_direction,
        )
        self.find_dialog.show()

    def _on_find_select(self, index: int) -> None:
        """Handle utterance selection from find dialog.

        Args:
            index: Selected utterance index
        """
        self.app.navigation_controller.find_utterance(index)
        if self.find_dialog:
            self.find_dialog.dialog.destroy()
            self.find_dialog = None

    def show_utterance_order_dialog(self) -> None:
        """Show the utterance order management dialog."""
        # Check if we have active recordings
        if not self.app.active_recordings:
            messagebox.showerror(
                "No Session",
                "Please load or create a session first.",
                parent=self.app.window.window,
            )
            return

        # Get current sorted order
        sorted_indices = self.app.active_recordings.get_sorted_indices()

        # Convert boolean sort_reverse to SortDirection enum
        sort_direction = (
            SortDirection.DOWN
            if self.app.active_recordings.sort_reverse
            else SortDirection.UP
        )

        # Create and show dialog
        dialog = UtteranceOrderDialog(
            self.app.window.window,
            self.app.state.recording.utterances,
            self.app.state.recording.labels,
            self.app.file_manager,
            sorted_indices,
            self.app.active_recordings.sort_column,
            sort_direction,
            self.app.state.recording.current_index,
        )

        result = dialog.show()

        if result is not None:
            # Result should be (sort_column, sort_reverse)
            sort_column, sort_reverse = result

            # Update active_recordings
            self.app.active_recordings.set_sort(sort_column, sort_reverse)

            # Save to session
            if self.app.current_session:
                self.app.current_session.sort_column = sort_column
                self.app.current_session.sort_reverse = sort_reverse
                self.app.current_session.save()
                self.app.window.set_status("Utterance order saved")

            # Update display
            self.app.display_controller.update_display()

    def _on_order_apply(self) -> None:
        """Handle order changes from utterance order dialog."""
        # Update display after order change
        self.app.display_controller.update_display()

        # Update take status to reflect new position
        self.app.navigation_controller.update_take_status()

        if self.order_dialog:
            self.order_dialog.dialog.destroy()
            self.order_dialog = None

    def show_settings_dialog(self) -> None:
        """Show the settings dialog."""
        if self.settings_dialog and self.settings_dialog.winfo_exists():
            self.settings_dialog.lift()
            return

        # Create settings dialog
        self.settings_dialog = tk.Toplevel(self.app.window.window)
        self.settings_dialog.title("Settings")
        self.settings_dialog.geometry("500x400")
        self.settings_dialog.transient(self.app.window.window)

        # TODO: Implement full settings UI
        # For now, just show audio settings

        frame = tk.Frame(self.settings_dialog)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Audio Settings", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10
        )

        # Sample rate
        tk.Label(frame, text="Sample Rate:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Label(frame, text=f"{self.app.config.audio.sample_rate} Hz").grid(
            row=1, column=1, sticky="w", pady=5
        )

        # Bit depth
        tk.Label(frame, text="Bit Depth:").grid(row=2, column=0, sticky="w", pady=5)
        tk.Label(frame, text=f"{self.app.config.audio.bit_depth} bit").grid(
            row=2, column=1, sticky="w", pady=5
        )

        # Channels
        tk.Label(frame, text="Channels:").grid(row=3, column=0, sticky="w", pady=5)
        tk.Label(frame, text=str(self.app.config.audio.channels)).grid(
            row=3, column=1, sticky="w", pady=5
        )

        # Close button
        tk.Button(
            self.settings_dialog, text="Close", command=self.settings_dialog.destroy
        ).pack(pady=10)

        # Center dialog
        self.settings_dialog.update_idletasks()
        x = (self.settings_dialog.winfo_screenwidth() // 2) - (
            self.settings_dialog.winfo_width() // 2
        )
        y = (self.settings_dialog.winfo_screenheight() // 2) - (
            self.settings_dialog.winfo_height() // 2
        )
        self.settings_dialog.geometry(f"+{x}+{y}")

    def _get_device_info(self, device_type: str):
        """Get device list and configuration based on type.

        Args:
            device_type: Either 'input' or 'output'

        Returns:
            Tuple of (devices, title, current_device)
        """
        device_manager = get_device_manager()

        if device_type == "input":
            devices = device_manager.get_input_devices()
            title = "Select Input Device"
            current_device = self.app.config.audio.input_device
        else:
            devices = device_manager.get_output_devices()
            title = "Select Output Device"
            current_device = self.app.config.audio.output_device

        return devices, title, current_device

    def _populate_device_listbox(
        self, listbox, devices, device_type: str, current_device
    ):
        """Populate listbox with devices and select current one.

        Args:
            listbox: Tkinter listbox widget
            devices: List of device dictionaries
            device_type: Either 'input' or 'output'
            current_device: Currently selected device index
        """
        # Add system default option
        listbox.insert(tk.END, "System Default")

        # Add devices
        channel_key = (
            "max_input_channels" if device_type == "input" else "max_output_channels"
        )
        for device in devices:
            name = device["name"]
            index = device["index"]
            channels = device.get(channel_key, 0)
            listbox.insert(tk.END, f"{name} (ID: {index}, {channels} ch)")

        # Select current device
        if current_device is None:
            listbox.selection_set(0)
        else:
            for i, device in enumerate(devices, 1):
                if device["index"] == current_device:
                    listbox.selection_set(i)
                    break

    def show_device_selection_dialog(self, device_type: str) -> None:
        """Show device selection dialog.

        Args:
            device_type: Either 'input' or 'output'
        """
        # Get device information
        devices, title, current_device = self._get_device_info(device_type)

        if not devices:
            messagebox.showerror(
                "Error",
                f"No {device_type} devices found",
                parent=self.app.window.window,
            )
            return

        # Create device selection dialog
        self.device_dialog = tk.Toplevel(self.app.window.window)
        self.device_dialog.title(title)
        self.device_dialog.transient(self.app.window.window)
        self.device_dialog.grab_set()

        # Device list
        listbox = tk.Listbox(self.device_dialog, width=50, height=15)
        listbox.pack(padx=10, pady=10)

        # Populate listbox
        self._populate_device_listbox(listbox, devices, device_type, current_device)

        def select_device():
            selection = listbox.curselection()
            if not selection:
                return

            idx = selection[0]
            if idx == 0:
                # System default selected
                device_index = None
            else:
                device_index = devices[idx - 1]["index"]

            # Set the device
            if device_type == "input":
                self.app.device_controller.set_input_device(device_index)
            else:
                self.app.device_controller.set_output_device(device_index)

            self.device_dialog.destroy()
            self.device_dialog = None

        # Buttons
        button_frame = tk.Frame(self.device_dialog)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Select", command=select_device).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(button_frame, text="Cancel", command=self.device_dialog.destroy).pack(
            side=tk.LEFT, padx=5
        )

        # Center dialog
        self.device_dialog.update_idletasks()
        x = (self.device_dialog.winfo_screenwidth() // 2) - (
            self.device_dialog.winfo_width() // 2
        )
        y = (self.device_dialog.winfo_screenheight() // 2) - (
            self.device_dialog.winfo_height() // 2
        )
        self.device_dialog.geometry(f"+{x}+{y}")

    def confirm_quit(self) -> bool:
        """Show confirmation dialog for quitting.

        Returns:
            True if user confirms quit
        """
        if self.app.state.recording.is_recording:
            result = messagebox.askyesno(
                "Recording in Progress",
                "Recording is in progress. Do you want to stop and quit?",
                parent=self.app.window.window,
            )
            return result

        # Check for unsaved changes if needed
        return True

    def show_error(self, title: str, message: str) -> None:
        """Show error dialog.

        Args:
            title: Dialog title
            message: Error message
        """
        messagebox.showerror(title, message, parent=self.app.window.window)

    def show_info(self, title: str, message: str) -> None:
        """Show information dialog.

        Args:
            title: Dialog title
            message: Information message
        """
        messagebox.showinfo(title, message, parent=self.app.window.window)

    def show_warning(self, title: str, message: str) -> None:
        """Show warning dialog.

        Args:
            title: Dialog title
            message: Warning message
        """
        messagebox.showwarning(title, message, parent=self.app.window.window)

    def cleanup(self) -> None:
        """Clean up open dialogs."""
        if (
            self.find_dialog
            and hasattr(self.find_dialog, "dialog")
            and self.find_dialog.dialog.winfo_exists()
        ):
            self.find_dialog.dialog.destroy()
        if (
            self.order_dialog
            and hasattr(self.order_dialog, "dialog")
            and self.order_dialog.dialog.winfo_exists()
        ):
            self.order_dialog.dialog.destroy()
        if self.settings_dialog and self.settings_dialog.winfo_exists():
            self.settings_dialog.destroy()
        if self.device_dialog and self.device_dialog.winfo_exists():
            self.device_dialog.destroy()

    def show_help(self) -> None:
        """Show help dialog with keyboard shortcuts."""
        dialog = HelpDialog(self.app.window.window)
        dialog.show()
