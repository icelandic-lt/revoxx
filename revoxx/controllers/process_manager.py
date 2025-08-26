"""Process manager for handling background processes and inter-process communication."""

import multiprocessing as mp
import os
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
        self.manager_watchdog_thread: Optional[threading.Thread] = None

        # Manager and shared resources
        self.manager: Optional[SyncManager] = None
        self.manager_pid: Optional[int] = None
        self.shutdown_event: Optional[Synchronized] = None
        self.manager_dict: Optional[dict] = None
        self.audio_queue: Optional[mp.Queue] = None
        self.record_queue: Optional[mp.Queue] = None
        self.playback_queue: Optional[mp.Queue] = None
        self.queue_manager: Optional[AudioQueueManager] = None

        # Initialize resources
        self._initialize_resources()

        # Start a watchdog thread to kill manager if parent dies
        # This is needed because manager processes don't die with parent
        self._start_manager_watchdog()

    def _initialize_resources(self) -> None:
        """Initialize multiprocessing resources."""
        # Create multiprocessing manager
        if self.app.debug:
            print("[ProcessManager] Creating mp.Manager()...")

        # Use context manager approach which ensures cleanup
        # But we need to keep reference for later use
        self.manager = mp.Manager()

        # Note: The manager creates its own process that we cannot easily control
        # It won't die automatically when parent dies
        if hasattr(self.manager, "_process") and self.manager._process:
            self.manager_pid = self.manager._process.pid
            if self.app.debug:
                print(f"[ProcessManager] Manager process PID: {self.manager_pid}")

        # Create shared resources
        self.shutdown_event = mp.Event()
        self.manager_dict = self.manager.dict()

        # Store parent PID for child processes to monitor
        self.manager_dict["parent_pid"] = os.getpid()
        if self.app.debug:
            print(f"[ProcessManager] Parent PID: {os.getpid()}")

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

    def _start_manager_watchdog(self) -> None:
        """Note: A watchdog thread won't work because it dies with the parent.

        The mp.Manager() creates non-daemon processes that don't die automatically.
        This is a known limitation. Solutions:
        1. Use normal mp.Queue instead of Manager-based queues
        2. Accept that manager processes may remain after hard kill
        3. Clean up orphaned manager processes on next start
        """
        # For now, we accept this limitation and rely on proper cleanup
        # when the program exits normally
        pass

    def start_processes(self) -> None:
        """Start background recording and playback processes."""
        if self.app.debug:
            print("[ProcessManager] Starting background processes...")

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
        self.record_process.daemon = True  # Ensure process terminates with parent
        # Note: We could use preexec_fn=os.setpgrp on Unix to put in new process group
        # but that's not portable and mp.Process doesn't support it
        self.record_process.start()
        if self.app.debug:
            print(
                f"[ProcessManager] Started record process (PID: {self.record_process.pid})"
            )

        # Start playback process
        self.playback_process = mp.Process(
            target=playback_process,
            args=(
                self.app.config.audio,
                self.playback_queue,
                self.app.shared_state.name,
                self.manager_dict,  # Add manager_dict for parent PID monitoring
                self.shutdown_event,
            ),
        )
        self.playback_process.daemon = True  # Ensure process terminates with parent
        self.playback_process.start()
        if self.app.debug:
            print(
                f"[ProcessManager] Started playback process (PID: {self.playback_process.pid})"
            )

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
            self.transfer_thread.join(timeout=0.2)

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
        if self.app.debug:
            print("[ProcessManager] Starting shutdown sequence...")

        # Signal shutdown to all processes
        if self.shutdown_event:
            if self.app.debug:
                print("[ProcessManager] Setting shutdown event...")
            self.shutdown_event.set()

        # Stop audio queue processing thread
        if self.app.debug:
            print("[ProcessManager] Stopping audio queue processing...")
        self.stop_audio_queue_processing()

        # Terminate all processes
        if self.app.debug:
            print("[ProcessManager] Terminating all processes...")
        self._terminate_all_processes()

        # Cleanup IPC resources (queues and manager)
        if self.app.debug:
            print("[ProcessManager] Cleaning up IPC resources...")
        self._cleanup_ipc_resources()

        # Clear all references
        if self.app.debug:
            print("[ProcessManager] Clearing all references...")
        self._clear_all_references()
        if self.app.debug:
            print("[ProcessManager] Shutdown complete.")

    def _terminate_all_processes(self) -> None:
        """Terminate recording and playback processes gracefully."""
        for process_name, process in [
            ("record", self.record_process),
            ("playback", self.playback_process),
        ]:
            if not process:
                if self.app.debug:
                    print(f"[ProcessManager] {process_name} process: None")
                continue

            if not process.is_alive():
                if self.app.debug:
                    print(f"[ProcessManager] {process_name} process: already dead")
                continue

            if self.app.debug:
                print(
                    f"[ProcessManager] Terminating {process_name} process (PID: {process.pid})..."
                )
            # Try graceful termination first
            process.terminate()
            process.join(timeout=0.5)

            # Force kill if still alive
            if process.is_alive():
                if self.app.debug:
                    print(
                        f"[ProcessManager] Force killing {process_name} process (PID: {process.pid})..."
                    )
                process.kill()
                process.join(timeout=0.2)

                if process.is_alive():
                    if self.app.debug:
                        print(
                            f"[ProcessManager] WARNING: {process_name} process still alive after kill!"
                        )
            else:
                if self.app.debug:
                    print(
                        f"[ProcessManager] {process_name} process terminated gracefully."
                    )

    def _cleanup_ipc_resources(self) -> None:
        """Close queues and shutdown multiprocessing manager."""
        # Close all queues
        for queue_name, queue_obj in [
            ("audio", self.audio_queue),
            ("record", self.record_queue),
            ("playback", self.playback_queue),
        ]:
            if queue_obj:
                try:
                    if self.app.debug:
                        print(f"[ProcessManager] Closing {queue_name} queue...")
                    queue_obj.close()
                    queue_obj.join_thread()
                except (AttributeError, OSError) as e:
                    if self.app.debug:
                        print(f"[ProcessManager] Error closing {queue_name} queue: {e}")

        # Shutdown multiprocessing manager
        if self.manager:
            try:
                if self.app.debug:
                    print("[ProcessManager] Shutting down multiprocessing manager...")
                # Try graceful shutdown
                self.manager.shutdown()
                if self.app.debug:
                    print("[ProcessManager] Manager shutdown complete.")

                # Check if the manager process is still alive
                if hasattr(self.manager, "_process") and self.manager._process:
                    if self.manager._process.is_alive():
                        if self.app.debug:
                            print(
                                f"[ProcessManager] WARNING: Manager process {self.manager._process.pid} still alive after shutdown!"
                            )
                        # Force terminate the manager process
                        self.manager._process.terminate()
                        self.manager._process.join(timeout=0.2)
                        if self.manager._process.is_alive():
                            if self.app.debug:
                                print(
                                    "[ProcessManager] Force killing manager process..."
                                )
                            self.manager._process.kill()
            except (BrokenPipeError, OSError) as e:
                if self.app.debug:
                    print(f"[ProcessManager] Error shutting down manager: {e}")

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
