"""Adaptive frame rate controller for UI updates.

Automatically adjusts update frequency based on actual achieved frame rate,
providing smooth performance on fast hardware while gracefully degrading
on slower systems (e.g., x86 Macs with Tcl/Tk 9's CoreGraphics rendering).

Set environment variable REVOXX_DEBUG_FPS=1 to enable frame rate logging.
"""

import os
import time
from typing import Optional

DEBUG_FPS = os.environ.get("REVOXX_DEBUG_FPS", "").lower() in ("1", "true", "yes")


class AdaptiveFrameRate:
    """Adaptive frame rate controller that adjusts based on achieved frame rate.

    On fast systems (e.g., Apple Silicon), this allows high frame rates (~60fps).
    On slower systems (e.g., Intel x86 with Tcl/Tk 9), it automatically reduces
    the frame rate to match what the system can actually render.

    Note: with draw_idle(), actual rendering is asynchronous.
    We measure whether frames are delivered on time. If the actual time
    between frames exceeds the planned interval, we slow down.
    """

    def __init__(self, min_ms: int = 16, max_ms: int = 300, smoothing: float = 0.2):
        """Initialize the adaptive frame rate controller.

        Args:
            min_ms: Minimum delay between frames (caps max fps). Default 16ms = ~60fps.
            max_ms: Maximum delay between frames (floor for min fps). Default 300ms = ~3fps.
            smoothing: Smoothing factor for exponential moving average (0-1).
        """
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.smoothing = smoothing
        self._last_frame_time: Optional[float] = None
        self._current_interval = min_ms
        self._avg_overshoot = 0.0

    def frame_start(self) -> None:
        """Call at the start of each frame to measure timing."""
        now = time.perf_counter()

        if self._last_frame_time is not None:
            actual_ms = (now - self._last_frame_time) * 1000
            overshoot = actual_ms - self._current_interval

            self._avg_overshoot = (
                self.smoothing * overshoot + (1 - self.smoothing) * self._avg_overshoot
            )

            if self._avg_overshoot > 5:
                self._current_interval = min(
                    self.max_ms, int(self._current_interval + self._avg_overshoot * 0.5)
                )
            elif self._avg_overshoot < -2 and self._current_interval > self.min_ms:
                self._current_interval = max(self.min_ms, self._current_interval - 1)

        self._last_frame_time = now

    def frame_end(self) -> int:
        """Call at the end of each frame. Returns interval for next frame."""
        return self._current_interval

    def frame_tick(self) -> int:
        """Combined frame_start + frame_end for simpler usage."""
        self.frame_start()
        return self.frame_end()

    def get_current_fps(self) -> float:
        """Get the current target frames per second."""
        if self._current_interval > 0:
            return 1000.0 / self._current_interval
        return 60.0

    def get_overshoot(self) -> float:
        """Get average overshoot in ms (positive = behind, negative = ahead)."""
        return self._avg_overshoot

    def reset(self) -> None:
        """Reset the frame rate estimator to initial state."""
        self._last_frame_time = None
        self._current_interval = self.min_ms
        self._avg_overshoot = 0.0


# Singleton instance for shared use across playback and recording
_adaptive_frame_rate: Optional[AdaptiveFrameRate] = None


def get_adaptive_frame_rate() -> AdaptiveFrameRate:
    """Get the shared adaptive frame rate controller instance."""
    global _adaptive_frame_rate
    if _adaptive_frame_rate is None:
        _adaptive_frame_rate = AdaptiveFrameRate()
    return _adaptive_frame_rate
