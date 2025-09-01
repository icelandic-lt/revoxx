"""Constants for the Revoxx Recorder application.

This module defines all constant values used throughout the application,
organized into logical groups for audio processing, user interface,
file handling, and keyboard bindings.
"""

from enum import Enum
import platform


class MsgType(Enum):
    """Message types for the status display.

    Defines different categories of status messages with different
    behaviors for display duration and persistence.
    """

    DEFAULT = "default"  # Shows current utterance/take info (static state)
    TEMPORARY = "temporary"  # Auto-clears after timeout (3 seconds)
    ACTIVE = "active"  # Shows ongoing operation (recording, monitoring)
    ERROR = "error"  # Permanent until resolved or manually cleared


class MsgConfig:
    """Configuration for status message behavior."""

    DEFAULT_TEMPORARY_DURATION_MS = 3000  # 3 seconds for temporary messages
    # Priority levels (higher number = higher priority)
    PRIORITY = {
        MsgType.DEFAULT: 0,
        MsgType.TEMPORARY: 1,
        MsgType.ACTIVE: 2,
        MsgType.ERROR: 3,
    }


class AudioConstants:
    """Audio processing related constants.

    Defines parameters for audio recording, processing, and analysis
    including FFT settings, mel spectrogram parameters, and normalization
    factors for different bit depths.
    """

    # Sample rates and channels
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_CHANNELS = 1
    DEFAULT_BIT_DEPTH = 24
    SUPPORTED_BIT_DEPTHS = [16, 24]

    # FFT Parameters
    N_FFT = 2048
    HOP_LENGTH = 512

    # Mel Spectrogram
    N_MELS = 96  # Increased for better low frequency resolution
    FMIN = 50  # Hz - Lower bound for fundamental frequency
    FMAX = 12000  # Hz - Extended for better high frequency detail

    # Display ranges
    DB_MIN = -70  # Background noise level
    DB_MAX = 0  # Maximum possible dB value
    DB_REFERENCE = 1e-10  # Reference for dB calculation

    # dB conversion factors
    AMPLITUDE_TO_DB_FACTOR = 20.0  # Factor for converting amplitude to dB (20 * log10)
    POWER_TO_DB_FACTOR = 10.0  # Factor for converting power to dB (10 * log10)
    NOISE_FLOOR = 1e-10  # Noise floor for dB calculations to avoid log(0)

    # Level meter constants
    MIN_DB_LEVEL = -60.0  # Minimum dB level for peak hold
    MS_TO_SEC = 1000.0  # Milliseconds to seconds conversion factor

    # Processing
    CLIPPING_THRESHOLD = 0.99  # 99% of maximum value
    AUDIO_CHUNK_SIZE = 1024
    MIN_CLIPPING_MARKER_DISTANCE = 5  # Frames between markers
    FREQUENCY_NOISE_FLOOR_DB = (
        -50
    )  # dB threshold for max frequency detection (-60 = more sensitive, -40 = less sensitive)
    # Threshold above dB min to consider a mel bin as having signal energy for max-frequency detection
    MAX_FREQ_ENERGY_THRESHOLD_DB = 20

    # Normalization factors
    NORM_FACTOR_16BIT = 32768.0  # 2^15
    NORM_FACTOR_32BIT = 2147483647.0  # 2^31 - 1 (max value for int32)


class UIConstants:
    """User interface related constants.

    Defines visual appearance settings, timing parameters, layout ratios,
    and display configuration for the graphical user interface.

    Color values are dynamically loaded from the theme system.
    Access colors directly as UIConstants.COLOR_* attributes.
    """

    # Lazy import to avoid circular dependencies
    _theme_manager = None
    _colors_initialized = False

    @classmethod
    def _init_colors(cls):
        """Initialize color attributes from theme manager."""
        if not cls._colors_initialized:
            # Lazy import here to avoid circular dependencies
            # This breaks the "all imports at top" rule but is necessary because:
            # 1. UIConstants needs theme_manager to get colors
            # 2. Other UI modules import UIConstants
            # 3. If we import theme_manager at module level, it creates circular imports
            from .ui.themes import theme_manager

            cls._theme_manager = theme_manager
            cls._update_colors()
            cls._colors_initialized = True

    @classmethod
    def _update_colors(cls):
        """Update color attributes from current theme."""
        if cls._theme_manager:
            colors = cls._theme_manager.colors
            cls.COLOR_BACKGROUND = colors.COLOR_BACKGROUND
            cls.COLOR_BACKGROUND_SECONDARY = colors.COLOR_BACKGROUND_SECONDARY
            cls.COLOR_BACKGROUND_TERTIARY = colors.COLOR_BACKGROUND_TERTIARY
            cls.COLOR_TEXT_NORMAL = colors.COLOR_TEXT_NORMAL
            cls.COLOR_TEXT_SECONDARY = colors.COLOR_TEXT_SECONDARY
            cls.COLOR_TEXT_RECORDING = colors.COLOR_TEXT_RECORDING
            cls.COLOR_TEXT_INACTIVE = colors.COLOR_TEXT_INACTIVE
            cls.COLOR_ACCENT = colors.COLOR_ACCENT
            cls.COLOR_WARNING = colors.COLOR_WARNING
            cls.COLOR_SUCCESS = colors.COLOR_SUCCESS
            cls.COLOR_BORDER = colors.COLOR_BORDER
            cls.COLOR_CLIPPING = colors.COLOR_CLIPPING
            cls.COLOR_PLAYBACK_LINE = colors.COLOR_PLAYBACK_LINE
            cls.COLOR_EDGE_INDICATOR = colors.COLOR_EDGE_INDICATOR

    @classmethod
    def refresh(cls):
        """Refresh colors from current theme (call after theme change)."""
        cls._init_colors()
        cls._update_colors()

    def __class_getattr__(cls, name):
        """Get color attributes dynamically."""
        if name.startswith("COLOR_"):
            cls._init_colors()
            return getattr(cls, name)
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    # Clipping display
    CLIPPING_LINE_WIDTH = 3
    CLIPPING_LINE_ALPHA = 0.7
    CLIPPING_WARNING_SYMBOL = "!"
    CLIPPING_WARNING_SIZE = 20
    CLIPPING_WARNING_POSITION = (0.02, 0.95)  # Relative position

    # Playback display
    PLAYBACK_LINE_WIDTH = 2
    PLAYBACK_LINE_ALPHA = 0.8
    PLAYBACK_UPDATE_MS = 10
    PLAYBACK_INITIAL_CHECK_MS = 50
    PLAYBACK_IDLE_RETRY_MS = 20
    PLAYBACK_WATCHDOG_NEAR_END_RATIO = 0.95
    PLAYBACK_WATCHDOG_STALL_MS = 200
    PLAYBACK_FADEOUT_MS = 200
    PLAYBACK_FADEOUT_STEPS = 8

    # Edge indicator display
    EDGE_INDICATOR_WIDTH = 2
    EDGE_INDICATOR_ALPHA = 0.8
    EDGE_INDICATOR_TIMEOUT_MS = 350

    # Timing (milliseconds)
    ANIMATION_UPDATE_MS = 20
    PLAYBACK_CHECK_MS = 50
    PLAYBACK_STOP_DELAY_MS = 50  # Delay after stopping playback
    FOCUS_DELAY_MS = 100
    POST_RECORDING_DELAY_MS = 500
    INITIAL_DISPLAY_DELAY_MS = 500  # Delay before showing initial recording
    STATUS_RESET_DELAY_MS = 1000

    # Process timing (seconds)
    AUDIO_PROCESS_SLEEP = 0.1
    PROCESS_JOIN_TIMEOUT = 0.1
    PLAYBACK_STOP_DELAY = 0.05  # Same as PLAYBACK_STOP_DELAY_MS but in seconds

    # Window layout ratios
    INFO_FRAME_HEIGHT_RATIO = 0.06
    CONTROL_FRAME_HEIGHT_RATIO = 0.12
    TEXT_WRAP_RATIO = 0.9
    DEFAULT_WINDOW_SIZE_RATIO = 0.8

    # Padding
    MAIN_FRAME_PADDING = 20
    FRAME_SPACING = 10

    # Font scaling
    FONT_SCALE_MEDIUM = 0.7
    FONT_SCALE_SMALL = 0.5
    MIN_FONT_SIZE_LARGE = 40
    MIN_FONT_SIZE_MEDIUM = 28
    MIN_FONT_SIZE_SMALL = 14
    # Linux/X11 may have rendering issues with very large font sizes
    # This setting seems to ensure correct character rendering throughout
    FONT_SIZE_LARGE = 75

    # Platform-specific font configuration
    _system = platform.system()
    if _system == "Darwin":  # macOS
        FONT_FAMILY_MONO = ("SF Mono", "Monaco", "Menlo", "Courier New")
        FONT_FAMILY_SANS = ("SF Pro Display", "Helvetica Neue", "Arial")
    elif _system == "Linux":
        # Linux fonts with good Unicode support
        # Ubuntu fonts first - if available - followed by other common fonts
        FONT_FAMILY_MONO = (
            "Ubuntu Mono",
            "DejaVu Sans Mono",
            "Liberation Mono",
            "Noto Sans Mono",
            "Bitstream Vera Sans Mono",
            "FreeMono",
            "Courier New",
            "monospace",
        )
        FONT_FAMILY_SANS = (
            "Ubuntu",
            "DejaVu Sans",
            "Liberation Sans",
            "Noto Sans",
            "Bitstream Vera Sans",
            "FreeSans",
            "Arial",
            "sans-serif",
        )
    else:  # Windows and other systems
        FONT_FAMILY_MONO = ("Consolas", "Courier New", "Lucida Console")
        FONT_FAMILY_SANS = ("Segoe UI", "Arial", "Tahoma")

    # Spectrogram display
    SPECTROGRAM_WIDTH_INCHES = 8
    SPECTROGRAM_HEIGHT_INCHES = 1.5  # Reduced from 2 to make it more compact
    SPECTROGRAM_DPI = 100
    SPECTROGRAM_DISPLAY_SECONDS = 3.0

    # Layout calculations
    ADAPTIVE_MARGIN_MIN_WIDTH_INCHES = 6.0  # Width below which we use maximum margin
    ADAPTIVE_MARGIN_MAX_WIDTH_INCHES = 16.0  # Width above which we use minimum margin
    ADAPTIVE_MARGIN_MIN = 0.04  # Minimum left margin for wide windows
    ADAPTIVE_MARGIN_MAX = 0.10  # Maximum left margin for narrow windows
    SUBPLOT_MARGIN_RIGHT = 0.98  # Right margin for subplots
    SUBPLOT_MARGIN_TOP = 0.95  # Top margin for subplots
    SUBPLOT_MARGIN_BOTTOM = 0.08  # Bottom margin for subplots

    # Display tolerances
    FIGURE_SIZE_CHANGE_THRESHOLD = 0.1  # Minimum change in inches to trigger resize
    DPI_CHANGE_THRESHOLD = 5  # Minimum DPI change to trigger update

    # Axis settings
    AXIS_LABEL_FONTSIZE = 8
    AXIS_TICK_FONTSIZE = 6
    N_TIME_TICKS = 7
    N_FREQUENCY_TICKS = 8
    # Fraction of frequency ticks allocated to the lower third of the spectrum
    FREQ_TICKS_LOWER_FRACTION = 0.6


class FileConstants:
    """File and path related constants.

    Defines default file paths, extensions, and audio format
    specifications for file operations.
    """

    DEFAULT_SCRIPT_FILE = "utts.data"
    DEFAULT_RECORDING_DIR = "recordings"
    AUDIO_FILE_EXTENSION = ".flac"
    LEGACY_AUDIO_FILE_EXTENSION = ".wav"
    NOT_FOUND_AUDIO = "not_found.wav"

    # File formats
    PCM_16_SUBTYPE = "PCM_16"
    PCM_24_SUBTYPE = "PCM_24"
    FLAC_SUBTYPE = "FLAC"


class KeyBindings:
    """Keyboard shortcuts.

    Defines all keyboard bindings for application control.
    Most keys use lowercase; special keys use Tkinter notation.
    Platform-specific modifiers are handled at binding time.
    """

    RECORD = "space"
    PLAY = "p"
    NAVIGATE_UP = "Up"
    NAVIGATE_DOWN = "Down"
    BROWSE_TAKES_LEFT = "Left"
    BROWSE_TAKES_RIGHT = "Right"
    TOGGLE_SPECTROGRAM = ["m", "M"]
    TOGGLE_LEVEL_METER = ["l", "L"]
    TOGGLE_MONITORING = "o"
    DELETE_RECORDING = "d"  # Used with Cmd/Ctrl modifier
    TOGGLE_FULLSCREEN = "F10"
    SHOW_HELP = "F1"  # Standard help key
    SHOW_INFO = "i"
    FIND_UTTERANCE = "f"  # Used with Cmd/Ctrl modifier

    # Session management - use platform-specific modifier
    NEW_SESSION = "n"  # Ctrl/Cmd+N
    OPEN_SESSION = "o"  # Ctrl/Cmd+O
    QUIT = "q"  # Ctrl/Cmd+Q
