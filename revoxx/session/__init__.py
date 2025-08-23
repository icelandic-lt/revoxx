"""Session management for Revoxx.

This module provides session management functionality for organizing
recordings into coherent sessions.
"""

from .manager import SessionManager
from .models import Session, SessionConfig, SpeakerInfo

__all__ = [
    "SessionManager",
    "Session",
    "SessionConfig",
    "SpeakerInfo",
]
