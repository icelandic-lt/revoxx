"""Revoxx Recorder - A tool for recording emotional speech."""

try:
    # Try to use versioningit for dynamic version detection
    from versioningit import get_version

    __version__ = get_version(root="../", config={})
except (ImportError, Exception):
    # Fallback if versioningit is not installed or fails
    __version__ = "1.0.0+dev"

__author__ = "Grammatek"

# Only import main entry point to avoid circular imports
__all__ = ["main", "Revoxx"]


def main():
    """Main entry point."""
    from .app import main as app_main

    app_main()


# Lazy import for Revoxx
def __getattr__(name):
    if name == "Revoxx":
        from .app import Revoxx

        return Revoxx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
