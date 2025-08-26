"""Process manager for handling background processes and inter-process communication."""

import multiprocessing as mp
import queue
import threading
from typing import Optional, TYPE_CHECKING
from multiprocessing.managers import SyncManager
from multiprocessing.sharedctypes import Synchronized

from ..audio.recorder import record_process
from ..audio.player import playback_process
from ..audio.queue_manager import AudioQueueManager

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
        self.queue_manager: Optional[AudioQueueManager] = None

        # Initialize resources
        self._initialize_resources()

    def _initialize_resources(self) -> None:
        """Initialize multiprocessing resources."""
        # Create multiprocessing manager
        self.manager = mp.Manager()

        # Create shared resources
        self.shutdown_event = mp.Event()
        self.manager_dict = self.manager.dict()

        # Create queue manager (which creates and owns the queues)
        self.queue_manager = AudioQueueManager()

        # Get queue references for process initialization
        self.audio_queue = self.queue_manager.audio_queue
        self.record_queue = self.queue_manager.record_queue
        self.playback_queue = self.queue_manager.playback_queue

        # Store references in app for other controllers
        self.app.shutdown_event = self.shutdown_event
        self.app.manager_dict = self.manager_dict
        self.app.queue_manager = self.queue_manager

        # Initialize shared state
        self.set_audio_queue_active(False)
        self.set_save_path(None)

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
        self.set_audio_queue_active(True)

        # Start transfer thread
        self.transfer_thread = threading.Thread(target=self._audio_transfer_worker)
        self.transfer_thread.daemon = True
        self.transfer_thread.start()

    def _audio_transfer_worker(self) -> None:
        """Worker thread for processing audio queue data."""
        try:
            while self._is_audio_transfer_active():
                self._process_single_audio_item()
        except (BrokenPipeError, OSError, EOFError):
            # IPC endpoints closed during shutdown - this is expected
            pass

    def _is_audio_transfer_active(self) -> bool:
        """Check if audio transfer should continue."""
        return self.is_audio_queue_active()

    def _process_single_audio_item(self) -> None:
        """Process one item from audio queue and update UI."""
        try:
            audio_data = self.queue_manager.get_audio_data(timeout=0.1)

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
            # Timeout is normal - no data available
            pass
        except (EOFError, BrokenPipeError):
            # Queue was closed or IPC endpoints closed - propagate to exit loop
            raise
        except OSError as e:
            if "handle is closed" not in str(e):
                print(f"Error in audio transfer thread: {e}")
            raise

    def stop_audio_queue_processing(self) -> None:
        """Stop audio queue processing."""
        self.set_audio_queue_active(False)

        # Wait for transfer thread to finish
        if self.transfer_thread and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=1.0)

    def get_save_path(self) -> Optional[str]:
        """Get the current save path for recording.

        Returns:
            Path to save recording or None
        """
        if self.manager_dict:
            try:
                return self.manager_dict.get("save_path")
            except (AttributeError, KeyError):
                return None
        return None

    def set_save_path(self, path: Optional[str]) -> None:
        """Set the save path for recording.

        Args:
            path: Path to save recording or None
        """
        if self.manager_dict:
            self.manager_dict["save_path"] = path

    def shutdown(self) -> None:
        """Shutdown all processes and cleanup resources."""
        # Signal shutdown to all processes
        if self.shutdown_event:
            self.shutdown_event.set()

        # Stop audio queue processing thread
        self.stop_audio_queue_processing()

        # Terminate all processes
        self._terminate_all_processes()

        # Cleanup IPC resources (queues and manager)
        self._cleanup_ipc_resources()

        # Clear all references
        self._clear_all_references()

    def _terminate_all_processes(self) -> None:
        """Terminate recording and playback processes gracefully."""
        for process_name, process in [
            ("record", self.record_process),
            ("playback", self.playback_process),
        ]:
            if not process or not process.is_alive():
                continue

            # Try graceful termination first
            process.terminate()
            process.join(timeout=2.0)

            # Force kill if still alive
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

    def _cleanup_ipc_resources(self) -> None:
        """Close queues and shutdown multiprocessing manager."""
        # Close all queues
        for queue_obj in [self.audio_queue, self.record_queue, self.playback_queue]:
            if queue_obj:
                try:
                    queue_obj.close()
                    queue_obj.join_thread()
                except (AttributeError, OSError):
                    pass  # Queue already closed or invalid

        # Shutdown multiprocessing manager
        if self.manager:
            try:
                self.manager.shutdown()
            except (BrokenPipeError, OSError):
                pass  # Manager already shutdown or pipe broken

    def _clear_all_references(self) -> None:
        """Clear all object references to allow garbage collection."""
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

    def set_audio_queue_active(self, active: bool) -> None:
        """Set audio queue processing state.

        Args:
            active: Whether audio queue should be active
        """
        if self.manager_dict:
            self.manager_dict["audio_queue_active"] = active

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
