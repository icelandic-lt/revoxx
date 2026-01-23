"""Shared state management for audio workers.

This module provides common state handling for audio player and recorder
workers to ensure consistent, idempotent command processing.
"""

from enum import Enum


class WorkerState(Enum):
    """State machine for audio workers (player/recorder).

    Using an explicit state enum instead of boolean flags ensures:
    - Idempotent operations (duplicate start/stop commands are safe)
    - Clear state transitions
    - No race conditions from lost queue messages
    """

    IDLE = "idle"
    ACTIVE = "active"
