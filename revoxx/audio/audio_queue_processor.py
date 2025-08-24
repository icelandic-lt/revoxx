"""Audio queue processor for handling real-time audio data transfer to UI.

This module processes audio data from the queue and updates UI components
in a thread-safe manner.
"""

import threading
import queue
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..app import Revoxx


class AudioQueueProcessor:
    """Processes audio data from queue and updates UI components.

    This class handles the low-level details of:
    - Managing the audio transfer thread
    - Processing different audio data formats
    - Updating UI components in a thread-safe manner
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the audio queue processor.

        Args:
            app: Reference to the main application
        """
        self.app = app
        self.transfer_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the audio queue processing thread."""
        if self._running:
            return

        self._running = True
        self.app.manager_dict["audio_queue_active"] = True

        self.transfer_thread = threading.Thread(target=self._worker_loop)
        self.transfer_thread.daemon = True
        self.transfer_thread.start()

    def stop(self) -> None:
        """Stop the audio queue processing thread."""
        self._running = False
        if self.app.manager_dict:
            self.app.manager_dict["audio_queue_active"] = False

        if self.transfer_thread and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=1.0)
            self.transfer_thread = None

    def update_state(self) -> None:
        """Update processing state based on UI visibility."""
        level_meter_visible = (
            hasattr(self.app.window, "level_meter_var")
            and self.app.window.level_meter_var.get()
        )
        needs_audio = self.app.state.ui.spectrogram_visible or level_meter_visible

        if self.app.manager_dict:
            self.app.manager_dict["audio_queue_active"] = needs_audio

    def _worker_loop(self) -> None:
        """Main worker loop for processing audio queue."""
        try:
            while self._should_continue():
                self._process_queue_item()
        except (BrokenPipeError, OSError, EOFError):
            # IPC endpoints closed during shutdown
            pass

    def _should_continue(self) -> bool:
        """Check if processing should continue.

        Returns:
            True if processing should continue, False otherwise
        """
        if not self._running:
            return False

        try:
            return self.app.manager_dict.get("audio_queue_active", False)
        except (AttributeError, KeyError):
            return False

    def _process_queue_item(self) -> None:
        """Process a single item from the audio queue."""
        try:
            audio_data = self.app.queue_manager.get_audio_data(timeout=0.1)
            self._route_audio_data(audio_data)
        except queue.Empty:
            pass  # Timeout is normal
        except queue.Full:
            pass  # Output queue full, skip
        except (BrokenPipeError, OSError, EOFError):
            self._running = False
            raise  # Re-raise to exit worker loop
        except Exception as e:
            if "closed" not in str(e).lower():
                print(f"Error processing audio queue: {e}")
            self._running = False
            raise

    def _route_audio_data(self, audio_data: Any) -> None:
        """Route audio data based on its type.

        Args:
            audio_data: The audio data to process (dict, tuple, or raw array)
        """
        if isinstance(audio_data, dict):
            self._handle_dict_format(audio_data)
        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
            self._handle_legacy_tuple_format(audio_data)
        else:
            self._handle_raw_format(audio_data)

    def _handle_dict_format(self, data: dict) -> None:
        """Handle dictionary-formatted audio data.

        Args:
            data: Dictionary with 'type' key indicating data type
        """
        data_type = data.get("type")

        if data_type == "audio":
            audio_array = data.get("data")
            if audio_array is not None:
                self._update_spectrogram(audio_array)
        elif data_type == "level":
            level = data.get("level", 0.0)
            self._update_level_meter(level)

    def _handle_legacy_tuple_format(self, data: tuple) -> None:
        """Handle legacy (audio_array, sample_rate) format.

        Args:
            data: Tuple of (audio_array, sample_rate)
        """
        audio_array, _ = data  # sample_rate not used currently
        self._update_spectrogram(audio_array)

    def _handle_raw_format(self, audio_data: Any) -> None:
        """Handle raw numpy array format.

        Args:
            audio_data: Raw audio data array
        """
        self._update_spectrogram(audio_data)

    def _update_spectrogram(self, audio_array: Any) -> None:
        """Update mel spectrogram with audio data.

        Args:
            audio_array: Audio data to display in spectrogram
        """
        if not self._has_spectrogram():
            return

        # Use after() to update in main thread
        self.app.root.after(
            0,
            lambda data=audio_array: self.app.window.mel_spectrogram.update_audio(data),
        )

    def _update_level_meter(self, level: float) -> None:
        """Update level meter with new level.

        Args:
            level: Audio level value
        """
        if not hasattr(self.app.window, "embedded_level_meter"):
            return

        # Use after() to update in main thread
        self.app.root.after(0, self.app.window.embedded_level_meter.update_level, level)

    def _has_spectrogram(self) -> bool:
        """Check if mel spectrogram is available.

        Returns:
            True if spectrogram exists and is initialized
        """
        return (
            hasattr(self.app.window, "mel_spectrogram")
            and self.app.window.mel_spectrogram is not None
        )
