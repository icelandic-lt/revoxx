"""Process cleanup utilities for handling signals and process termination."""

import atexit
import signal
import sys
from typing import Callable, Dict


class ProcessCleanupManager:
    """Manages signal handlers and process cleanup for graceful shutdown.

    This class handles:
    - Signal registration (SIGINT, SIGTERM, etc.)
    - Parent PID monitoring for child processes
    - Emergency cleanup on unexpected exit
    - Proper signal handling for Tkinter applications
    """

    def __init__(self, cleanup_callback: Callable[[], None], debug: bool = False):
        """Initialize the cleanup manager.

        Args:
            cleanup_callback: Function to call for cleanup
            debug: Enable debug output
        """
        self.cleanup_callback = cleanup_callback
        self.debug = debug
        self._cleanup_done = False
        self._original_handlers: Dict[int, any] = {}

        # Register atexit handler
        atexit.register(self._emergency_cleanup)

    def setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown.

        Should be called during application initialization.
        """
        if self.debug:
            print("[ProcessCleanup] Registering signal handlers...")

        # Store and replace signal handlers
        signals = {
            signal.SIGINT: "SIGINT",  # Ctrl+C, PyCharm "Stop"
            signal.SIGTERM: "SIGTERM",  # Terminate
            signal.SIGQUIT: "SIGQUIT",  # Quit
            signal.SIGHUP: "SIGHUP",  # Hangup
        }

        for sig, name in signals.items():
            original = signal.signal(sig, self._signal_handler)
            self._original_handlers[sig] = original
            if self.debug:
                print(f"[ProcessCleanup] Registered {name} handler (was: {original})")

    def refresh_sigint_handler(self) -> None:
        """Re-register SIGINT handler (before Tkinter mainloop).

        Tkinter interferes with SIGINT (Ctrl+C). This ensures it's properly set before entering the main event loop.
        """
        if self.debug:
            print("[ProcessCleanup] Refreshing SIGINT handler before mainloop...")

        current = signal.signal(signal.SIGINT, self._signal_handler)
        if self.debug and current != self._signal_handler:
            print(f"[ProcessCleanup] SIGINT handler was changed to: {current}")

    @staticmethod
    def ignore_signals_in_child() -> None:
        """Configure child process to ignore signals.

        Should be called at the start of child processes to prevent
        them from handling signals that should be handled by parent.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Could add more signals if needed

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals for graceful shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        _ = frame
        signal_names = {
            signal.SIGHUP: "SIGHUP",
            signal.SIGINT: "SIGINT",
            signal.SIGQUIT: "SIGQUIT",
            signal.SIGTERM: "SIGTERM",
        }

        signal_name = signal_names.get(signum, f"UNKNOWN({signum})")

        if self.debug:
            print(
                f"[ProcessCleanup] Signal {signal_name} ({signum}) received", flush=True
            )

        # Special handling for SIGINT
        if signum == signal.SIGINT and self.debug:
            print(
                "[ProcessCleanup] SIGINT ",
                flush=True,
            )

        self._do_cleanup()
        sys.exit(0)

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup called by atexit."""
        if self.debug and not self._cleanup_done:
            print("[ProcessCleanup] Emergency cleanup triggered", flush=True)
        self._do_cleanup()

    def _do_cleanup(self) -> None:
        """Perform actual cleanup."""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        if self.cleanup_callback:
            try:
                self.cleanup_callback()
            except Exception as e:
                if self.debug:
                    print(f"[ProcessCleanup] Error during cleanup: {e}", flush=True)

    def cleanup_complete(self) -> None:
        """Mark cleanup as complete to prevent duplicate cleanup."""
        self._cleanup_done = True
