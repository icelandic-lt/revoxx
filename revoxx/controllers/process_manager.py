"""Process manager for handling background processes and inter-process communication."""

import multiprocessing as mp
import queue
import threading
from typing import Optional, TYPE_CHECKING
from multiprocessing.managers import SyncManager
from multiprocessing.sharedctypes import Synchronized

from ..audio.recorder import record_process
from ..audio.player import playback_process

if TYPE_CHECKING:
    from ..app import Revoxx


class ProcessManager:
    """Manages background processes and inter-process communication.

    This controller handles:
    - Starting and stopping background processes
    - Managing process communication queues
    - Audio queue processing for visualizations
    - Process lifecycle management
    """

    def __init__(self, app: "Revoxx"):
        """Initialize the process manager.

        Args:
            app: Reference to the main application instance
        """
        self.app = app

        # Process references
        self.record_process: Optional[mp.Process] = None
        self.playback_process: Optional[mp.Process] = None
        self.transfer_thread: Optional[threading.Thread] = None

        # Manager and shared resources
        self.manager: Optional[SyncManager] = None
        self.shutdown_event: Optional[Synchronized] = None
        self.manager_dict: Optional[dict] = None
        self.audio_queue: Optional[mp.Queue] = None
        self.record_queue: Optional[mp.Queue] = None
        self.playback_queue: Optional[mp.Queue] = None

        # Initialize resources
        self._initialize_resources()

    def _initialize_resources(self) -> None:
        """Initialize multiprocessing resources."""
        # Create multiprocessing manager
        self.manager = mp.Manager()

        # Create shared resources
        self.shutdown_event = mp.Event()
        self.manager_dict = self.manager.dict()
        self.manager_dict["audio_queue_active"] = False
        self.manager_dict["save_path"] = None

        # Create communication queues
        self.audio_queue = mp.Queue(maxsize=100)
        self.record_queue = mp.Queue(maxsize=10)
        self.playback_queue = mp.Queue(maxsize=10)

        # Store references in app for other controllers
        self.app.shutdown_event = self.shutdown_event
        self.app.manager_dict = self.manager_dict
        self.app.audio_queue = self.audio_queue
        self.app.record_queue = self.record_queue
        self.app.playback_queue = self.playback_queue

    def start_processes(self) -> None:
        """Start background recording and playback processes."""
        # Start recording process
        self.record_process = mp.Process(
            target=record_process,
            args=(
                self.app.config.audio,
                self.audio_queue,
                self.app.shared_state.name,
                self.record_queue,
                self.manager_dict,
                self.shutdown_event,
            ),
        )
        self.record_process.start()

        # Start playback process
        self.playback_process = mp.Process(
            target=playback_process,
            args=(
                self.app.config.audio,
                self.playback_queue,
                self.app.shared_state.name,
                self.shutdown_event,
            ),
        )
        self.playback_process.start()

    def start_audio_queue_processing(self) -> None:
        """Start processing audio queue for real-time display."""
        self.manager_dict["audio_queue_active"] = True

        # Start transfer thread
        def audio_transfer_thread():
            try:
                while True:
                    # Check active flag
                    try:
                        active = self.manager_dict.get("audio_queue_active", False)
                    except (AttributeError, KeyError):
                        break
                    if not active:
                        break

                    try:
                        audio_data = self.audio_queue.get(timeout=0.1)

                        # Update mel spectrogram if visible
                        if (
                            hasattr(self.app.window, "mel_spectrogram")
                            and self.app.window.ui_state.spectrogram_visible
                        ):
                            # Use after() to update in main thread
                            self.app.root.after(
                                0,
                                lambda data=audio_data: self.app.window.mel_spectrogram.update_audio(
                                    data
                                ),
                            )

                    except queue.Empty:
                        # Timeout is normal
                        pass
                    except EOFError:
                        # Queue was closed
                        break
                    except BrokenPipeError:
                        # IPC endpoints closed
                        break
                    except OSError as e:
                        if "handle is closed" not in str(e):
                            print(f"Error in audio transfer thread: {e}")
                        break
            except (BrokenPipeError, OSError, EOFError):
                # IPC endpoints closed during shutdown
                pass

        self.transfer_thread = threading.Thread(target=audio_transfer_thread)
        self.transfer_thread.daemon = True
        self.transfer_thread.start()

    def stop_audio_queue_processing(self) -> None:
        """Stop audio queue processing."""
        if self.manager_dict:
            self.manager_dict["audio_queue_active"] = False

        # Wait for transfer thread to finish
        if self.transfer_thread and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=1.0)

    def update_audio_queue_state(self, active: bool) -> None:
        """Update audio queue state.

        Args:
            active: Whether audio queue processing should be active
        """
        if self.manager_dict:
            self.manager_dict["audio_queue_active"] = active

    def set_save_path(self, path: Optional[str]) -> None:
        """Set the save path for recording.

        Args:
            path: Path to save recording or None
        """
        if self.manager_dict:
            self.manager_dict["save_path"] = path

    def send_record_command(self, command: dict) -> None:
        """Send command to record process.

        Args:
            command: Command dictionary to send
        """
        if self.record_queue:
            try:
                self.record_queue.put(command, block=False)
            except queue.Full:
                # Queue is full, command will be dropped
                pass

    def send_playback_command(self, command: dict) -> None:
        """Send command to playback process.

        Args:
            command: Command dictionary to send
        """
        if self.playback_queue:
            try:
                self.playback_queue.put(command, block=False)
            except queue.Full:
                # Queue is full, command will be dropped
                pass

    def clear_audio_queue(self) -> None:
        """Clear the audio queue."""
        if self.audio_queue:
            try:
                while True:
                    self.audio_queue.get_nowait()
            except queue.Empty:
                pass

    def shutdown(self) -> None:
        """Shutdown all processes and cleanup resources."""
        # Set shutdown event
        if self.shutdown_event:
            self.shutdown_event.set()

        # Stop audio queue processing
        self.stop_audio_queue_processing()

        # Terminate processes
        for process in [self.record_process, self.playback_process]:
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)

        # Close queues
        for queue_obj in [self.audio_queue, self.record_queue, self.playback_queue]:
            if queue_obj:
                try:
                    queue_obj.close()
                    queue_obj.join_thread()
                except (AttributeError, OSError):
                    pass

        # Shutdown manager
        if self.manager:
            try:
                self.manager.shutdown()
            except (BrokenPipeError, OSError):
                pass

        # Clear references
        self.record_process = None
        self.playback_process = None
        self.transfer_thread = None
        self.manager = None
        self.shutdown_event = None
        self.manager_dict = None
        self.audio_queue = None
        self.record_queue = None
        self.playback_queue = None

    def is_audio_queue_active(self) -> bool:
        """Check if audio queue processing is active.

        Returns:
            True if audio queue is active
        """
        if self.manager_dict:
            try:
                return self.manager_dict.get("audio_queue_active", False)
            except (AttributeError, KeyError):
                return False
        return False

    def are_processes_running(self) -> bool:
        """Check if background processes are running.

        Returns:
            True if processes are running
        """
        return (
            self.record_process is not None
            and self.record_process.is_alive()
            and self.playback_process is not None
            and self.playback_process.is_alive()
        )
