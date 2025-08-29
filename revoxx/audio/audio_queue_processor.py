"""Audio queue processor for handling real-time audio data transfer to UI.

This module processes audio data from the queue and updates UI components
in a thread-safe manner.
"""

import threading
import queue
import time
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
        self.app.process_manager.set_audio_queue_active(True)

        self.transfer_thread = threading.Thread(target=self._worker_loop)
        self.transfer_thread.daemon = True
        self.transfer_thread.start()

    def stop(self) -> None:
        """Stop the audio queue processing thread."""
        self._running = False
        if self.app.manager_dict:
            self.app.process_manager.set_audio_queue_active(False)

        if self.transfer_thread and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=1.0)
            self.transfer_thread = None

    def update_state(self) -> None:
        """Update processing state based on UI visibility."""
        # Check any window setting
        needs_audio = False

        if hasattr(self.app, "window_manager") and self.app.window_manager:
            active_windows = self.app.window_manager.get_active_windows()
            for window in active_windows:
                meters_vis = getattr(window, "meters_visible", False)
                if meters_vis:
                    needs_audio = True
                    break

        if self.app.manager_dict:
            self.app.process_manager.set_audio_queue_active(needs_audio)

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
        # Only check _running - thread should keep running even when meters are off
        if not self._running:
            return False

        # Thread keeps running regardless of audio_queue_active
        return True

    def _process_queue_item(self) -> None:
        """Process a single item from the audio queue."""
        # Check if we should process audio data
        is_active = self.app.process_manager.is_audio_queue_active()
        if not is_active:
            # Meters are off - just sleep a bit to avoid busy waiting
            time.sleep(0.1)
            return

        # Meters are on - process normally
        try:
            audio_data = self.app.queue_manager.get_audio_data(timeout=0.1)

            # Process audio data based on its type
            if isinstance(audio_data, dict):
                self._handle_dict_format(audio_data)
            else:
                # Raw numpy array format
                self._handle_raw_format(audio_data)
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
        # Broadcast to ALL active windows with visible meters
        if hasattr(self.app, "window_manager") and self.app.window_manager:
            active_windows = self.app.window_manager.get_active_windows()

            for window in active_windows:
                has_spectrogram = (
                    hasattr(window, "mel_spectrogram") and window.mel_spectrogram
                )
                meters_visible = getattr(window, "meters_visible", False)

                if has_spectrogram and meters_visible:
                    try:
                        window.mel_spectrogram.audio_queue.put_nowait(audio_array)
                    except queue.Full:
                        pass
                    except AttributeError:
                        pass  # Widget not ready

    def _update_level_meter(self, level: float) -> None:
        """Update level meter with new level.

        Args:
            level: Audio level value
        """
        # Use after() to update in main thread
        # Widget must exist when keyboard bindings are active
        if self.app.window and self.app.window.embedded_level_meter:
            self.app.window.window.after(
                0, self.app.window.embedded_level_meter.update_level, level
            )
