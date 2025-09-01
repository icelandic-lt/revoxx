"""Device controller for managing audio input/output devices."""

from typing import Optional, List, Dict, Any, TYPE_CHECKING

from ..utils.device_manager import get_device_manager
from ..constants import FileConstants, MsgType

if TYPE_CHECKING:
    from ..app import Revoxx
    from ..session.models import SessionConfig


class DeviceController:
    """Handles audio device management and configuration.

    This controller manages:
    - Input/output device selection
    - Channel mapping configuration
    - Device validation and fallback
    - Audio settings updates
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the device controller.

        Args:
            app: Reference to the main application instance
        """
        self.app = app
        self._default_input_in_effect = False
        self._default_output_in_effect = False
        self._notified_default_input = False
        self._notified_default_output = False

    def apply_saved_settings(self) -> None:
        """Apply saved device settings from configuration."""
        device_manager = get_device_manager()

        # Apply saved input device (convert name to index)
        saved_input = self.app.settings_manager.get_setting("input_device")

        if saved_input is not None:
            # Convert device name to index
            input_index = device_manager.get_device_index_by_name(saved_input)

            if input_index is not None:
                self.set_input_device(input_index, save=False)
            else:
                # Device not found, use system default
                self.set_input_device(None, save=False)

        # Apply saved output device (convert name to index)
        saved_output = self.app.settings_manager.get_setting("output_device")

        if saved_output is not None:
            # Convert device name to index
            output_index = device_manager.get_device_index_by_name(saved_output)

            if output_index is not None:
                self.set_output_device(output_index, save=False)
            else:
                # Device not found, use system default
                self.set_output_device(None, save=False)

        # Apply saved input channel mapping
        saved_input_mapping = self.app.settings_manager.get_setting(
            "input_channel_mapping"
        )
        if saved_input_mapping is not None:
            self.set_input_channel_mapping(saved_input_mapping, save=False)

        # Apply saved output channel mapping
        saved_output_mapping = self.app.settings_manager.get_setting(
            "output_channel_mapping"
        )
        if saved_output_mapping is not None:
            self.set_output_channel_mapping(saved_output_mapping, save=False)

    def set_input_device(self, index: Optional[int], save: bool = True) -> None:
        """Set the input device for recording.

        Args:
            index: Device index to set (None for system default)
            save: Whether to save the setting persistently
        """
        # Convert index to device name for persistent storage
        device_manager = get_device_manager()
        if index is None:
            device_name = None
            display_name = "System Default"
        else:
            device_name = device_manager.get_device_name_by_index(index)
            if not device_name:
                # Device not found
                self.app.window.set_status(
                    f"Device index {index} not found", MsgType.ERROR
                )
                return
            display_name = device_name

        # Check if device is changing
        old_device_name = self.app.config.audio.input_device
        device_changed = old_device_name != device_name

        # Update configuration with device NAME (not index)
        self.app.config.audio.input_device = device_name

        # Update shared state with index for compatibility
        try:
            self.app.shared_state.set_input_device_index(index)
        except AttributeError:
            pass

        # Send device name to record process (will convert to index there)
        self.app.queue_manager.set_input_device(device_name)

        # Update device status flags
        if device_name is None:
            self._default_input_in_effect = True
            self._notified_default_input = False
        else:
            self._default_input_in_effect = False
            self._notified_default_input = False

        # Save to settings if requested
        if save:
            self.app.settings_manager.update_setting("input_device", device_name)
            # Update session settings
            self._update_session_input_device(index, device_changed)

        # Update UI status
        self.app.window.set_status(f"Input device: {display_name}")

    def set_input_channel_mapping(
        self, mapping: Optional[List[int]], save: bool = True
    ) -> None:
        """Set the input channel mapping.

        Args:
            mapping: List of channel indices or None for default
            save: Whether to save the setting persistently
        """
        # Update configuration
        self.app.config.audio.input_channel_mapping = mapping

        # Send to record process
        self.app.queue_manager.set_input_channel_mapping(mapping)

        # Save to settings if requested
        if save:
            self.app.settings_manager.update_setting("input_channel_mapping", mapping)
            # Update session settings
            if self.app.current_session and self.app.current_session.audio_config:
                self.app.current_session.audio_config.input_channel_mapping = mapping
                self.app.current_session.save()

        # Update status
        if mapping:
            self.app.window.set_status(f"Input channel mapping: {mapping}")
        else:
            self.app.window.set_status("Input channel mapping: Default")

    def set_output_device(self, index: int, save: bool = True) -> None:
        """Set the output device for playback.

        Args:
            index: Device index to set (None for system default)
            save: Whether to save the setting persistently
        """

        # Convert index to device name for persistent storage
        device_manager = get_device_manager()
        if index is None:
            device_name = None
            display_name = "System Default"
        else:
            device_name = device_manager.get_device_name_by_index(index)
            if not device_name:
                # Device not found
                self.app.window.set_status(
                    f"Device index {index} not found", MsgType.ERROR
                )
                return
            display_name = device_name

        # Check if device is changing
        old_device_name = self.app.config.audio.output_device
        device_changed = old_device_name != device_name

        # Update configuration with device NAME (not index)
        self.app.config.audio.output_device = device_name

        # Update shared state with index for compatibility
        try:
            self.app.shared_state.set_output_device_index(index)
        except AttributeError:
            pass

        # Send device name to playback process (will convert to index there)
        self.app.queue_manager.set_output_device(device_name)

        # Update device status flags
        if device_name is None:
            self._default_output_in_effect = True
            self._notified_default_output = False
        else:
            self._default_output_in_effect = False
            self._notified_default_output = False

        # Save to settings if requested
        if save:
            self.app.settings_manager.update_setting("output_device", device_name)
            # Update session settings
            self._update_session_output_device(index, device_changed)

        # Update UI status
        self.app.window.set_status(f"Output device: {display_name}")

    def set_output_channel_mapping(
        self, mapping: Optional[List[int]], save: bool = True
    ) -> None:
        """Set the output channel mapping.

        Args:
            mapping: List of channel indices or None for default
            save: Whether to save the setting persistently
        """
        # Update configuration
        self.app.config.audio.output_channel_mapping = mapping

        # Send to playback process
        self.app.queue_manager.set_output_channel_mapping(mapping)

        # Save to settings if requested
        if save:
            self.app.settings_manager.update_setting("output_channel_mapping", mapping)
            if self.app.current_session and self.app.current_session.audio_config:
                self.app.current_session.audio_config.output_channel_mapping = mapping
                self.app.current_session.save()

        # Update status
        if mapping:
            self.app.window.set_status(f"Output channel mapping: {mapping}")
        else:
            self.app.window.set_status("Output channel mapping: Default")

    def update_audio_settings(self) -> None:
        """Update audio settings across all processes."""
        # Update shared state with current audio configuration
        format_type = 1 if FileConstants.AUDIO_FILE_EXTENSION == ".flac" else 0

        try:
            self.app.shared_state.update_audio_settings(
                sample_rate=self.app.config.audio.sample_rate,
                bit_depth=self.app.config.audio.bit_depth,
                channels=self.app.config.audio.channels,
                format_type=format_type,
            )
        except AttributeError:
            pass

    @staticmethod
    def get_available_input_devices() -> List[Dict[str, Any]]:
        """Get list of available input devices.

        Returns:
            List of device information dictionaries
        """
        device_manager = get_device_manager()
        return device_manager.get_input_devices()

    @staticmethod
    def get_available_output_devices() -> List[Dict[str, Any]]:
        """Get list of available output devices.

        Returns:
            List of device information dictionaries
        """
        device_manager = get_device_manager()
        return device_manager.get_output_devices()

    def validate_device_compatibility(
        self, device_name: str, is_input: bool = True
    ) -> bool:
        """Check if a device is compatible with current audio settings.

        Args:
            device_name: Name of the device to check
            is_input: True for input device, False for output device

        Returns:
            True if device is compatible with current settings
        """
        device_manager = get_device_manager()
        return device_manager.check_device_compatibility(
            device_name=device_name,
            sample_rate=self.app.config.audio.sample_rate,
            bit_depth=self.app.config.audio.bit_depth,
            channels=self.app.config.audio.channels,
            is_input=is_input,
        )

    def find_compatible_device(self, is_input: bool = True) -> Optional[int]:
        """Find any compatible device for current audio settings.

        Args:
            is_input: True to find input device, False for output device

        Returns:
            Device index or None if no compatible device found
        """
        device_manager = get_device_manager()
        devices = (
            device_manager.get_input_devices()
            if is_input
            else device_manager.get_output_devices()
        )

        for device in devices:
            if self.validate_device_compatibility(device["name"], is_input):
                return device["index"]

        return None

    @staticmethod
    def refresh_devices() -> None:
        """Refresh the device list from the system."""
        try:
            device_manager = get_device_manager()
            device_manager.refresh()
        except (ImportError, RuntimeError):
            # Device manager might not be available
            pass

    @property
    def is_default_input_active(self) -> bool:
        """Check if default input device is being used."""
        return self._default_input_in_effect

    @property
    def is_default_output_active(self) -> bool:
        """Check if default output device is being used."""
        return self._default_output_in_effect

    @property
    def has_notified_default_input(self) -> bool:
        """Check if user has been notified about default input."""
        return self._notified_default_input

    @property
    def has_notified_default_output(self) -> bool:
        """Check if user has been notified about default output."""
        return self._notified_default_output

    def mark_input_notified(self) -> None:
        """Mark that user has been notified about default input."""
        self._notified_default_input = True

    def mark_output_notified(self) -> None:
        """Mark that user has been notified about default output."""
        self._notified_default_output = True

    def _get_device_name_from_index(self, index: Optional[int]) -> str:
        """Convert device index to device name.

        Args:
            index: Device index (None for system default)

        Returns:
            Device name or "default" for system default
        """
        if index is None:
            return "default"

        device_manager = get_device_manager()
        device_name = device_manager.get_device_name_by_index(index)
        return device_name if device_name else "default"

    def _update_session_input_device(
        self, index: Optional[int], device_changed: bool
    ) -> None:
        """Update session configuration with new input device.

        Args:
            index: Device index (None for default)
            device_changed: Whether the device actually changed
        """
        if not self.app.current_session or not self.app.current_session.audio_config:
            return

        # Convert index to device name using shared method
        device_name = self._get_device_name_from_index(index)

        # Update session
        self.app.current_session.audio_config.input_device = device_name

        # If device changed, reset channel mapping
        if device_changed:
            self.app.current_session.audio_config.input_channel_mapping = None
            # Also reset runtime channel mapping
            self.set_input_channel_mapping(None, save=False)

        # Save session
        self.app.current_session.save()

    def _update_session_output_device(
        self, index: Optional[int], device_changed: bool
    ) -> None:
        """Update session configuration with new output device.

        Args:
            index: Device index (None for default)
            device_changed: Whether the device actually changed
        """
        if not self.app.current_session or not self.app.current_session.audio_config:
            return

        # Convert index to device name using shared method
        device_name = self._get_device_name_from_index(index)

        # Update session
        self.app.current_session.audio_config.output_device = device_name

        # If device changed, reset channel mapping
        if device_changed:
            self.app.current_session.audio_config.output_channel_mapping = None
            # Also reset runtime channel mapping
            self.set_output_channel_mapping(None, save=False)

        # Save session
        self.app.current_session.save()

    def apply_session_audio_settings(self, session_config: "SessionConfig") -> None:
        """Apply audio settings from session configuration.

        This method is called when loading a session to restore all
        device and channel settings with proper fallback handling.

        Args:
            session_config: Session audio configuration to apply
        """
        if not session_config:
            return

        device_manager = get_device_manager()

        # Apply input device
        if session_config.input_device:
            if session_config.input_device == "default":
                # Use system default
                self.set_input_device(None, save=False)
                # No channel mapping for default device
                self.set_input_channel_mapping(None, save=False)
            else:
                # Try to find the saved device
                device_index = device_manager.get_device_index_by_name(
                    session_config.input_device
                )

                if device_index is not None:
                    # Device found, apply it
                    self.set_input_device(device_index, save=False)

                    # Apply channel mapping only if device matches
                    if session_config.input_channel_mapping:
                        # TODO: Validate channel mapping against device capabilities
                        self.set_input_channel_mapping(
                            session_config.input_channel_mapping, save=False
                        )
                    else:
                        self.set_input_channel_mapping(None, save=False)
                else:
                    # Device not found, fallback to default
                    print(
                        f"Warning: Input device '{session_config.input_device}' not found, using default"
                    )
                    self.app.window.set_status("Input device not found, using default")
                    self.set_input_device(None, save=False)
                    self.set_input_channel_mapping(None, save=False)

        # Apply output device (if configured)
        if hasattr(session_config, "output_device") and session_config.output_device:
            if session_config.output_device == "default":
                # Use system default (None represents system default)
                self.set_output_device(None, save=False)
                self.set_output_channel_mapping(None, save=False)
            else:
                # Try to find the saved device
                device_index = device_manager.get_device_index_by_name(
                    session_config.output_device
                )

                if device_index is not None:
                    # Device found, apply it
                    self.set_output_device(device_index, save=False)

                    # Apply channel mapping only if device matches
                    if (
                        hasattr(session_config, "output_channel_mapping")
                        and session_config.output_channel_mapping
                    ):
                        # TODO: Validate channel mapping against device capabilities
                        self.set_output_channel_mapping(
                            session_config.output_channel_mapping, save=False
                        )
                    else:
                        self.set_output_channel_mapping(None, save=False)
                else:
                    # Device not found, fallback to default
                    print(
                        f"Warning: Output device '{session_config.output_device}' not found, using default"
                    )
                    self.app.window.set_status("Output device not found, using default")
                    self.set_output_device(None, save=False)
                    self.set_output_channel_mapping(None, save=False)
