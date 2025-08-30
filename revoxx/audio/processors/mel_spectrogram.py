"""Optimized mel spectrogram processor using only numpy.

This is an optimized lightweight alternative implementation that doesn't require librosa
and is ~2.4x faster (on a Mac M4).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from .processor_base import AudioProcessor
from ...constants import AudioConstants
from ...utils.audio_utils import normalize_audio


def hz_to_mel(frequencies: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequencies in Hz to mel scale.

    Uses the standard mel scale formula from Stevens, Volkmann & Newman (1937).
    This is the most widely used mel scale definition in audio processing,
    compatible with librosa, MATLAB, Kaldi, and other standard tools.

    Formula: mel = 2595 * log10(1 + f/700)

    Args:
        frequencies: Frequency in Hz (scalar or array)

    Returns:
        Frequency in mel scale
    """
    # Standard mel scale constants:
    # 2595 = 1000 / log10(2) * log10(1 + 1000/700) - calibrated so 1000 Hz = 1000 mel
    # 700 = experimentally determined frequency scaling factor
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def mel_to_hz(mels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mel scale to frequencies in Hz.

    Inverse of the standard mel scale formula.

    Formula: f = 700 * (10^(mel/2595) - 1)

    Args:
        mels: Mel scale value (scalar or array)

    Returns:
        Frequency in Hz
    """
    # Inverse transformation using the same standard constants
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _prepare_mel_filter_bank_params(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Prepare common parameters for mel filter bank creation.

    Args:
        sample_rate: Sample rate of the audio
        n_fft: Number of FFT points
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (defaults to sample_rate/2)

    Returns:
        Tuple of (fft_freqs, hz_points, n_freqs)
    """
    if fmax is None:
        fmax = sample_rate / 2

    # Frequency bins for FFT
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Mel scale points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    return fft_freqs, hz_points, n_freqs


def create_mel_filter_bank_loop(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Create a mel filter bank matrix using a loop-based approach with vectorized operations.

    This implementation uses a loop over mel bands but vectorizes operations within
    each band. It's more efficient for large matrices (>82k elements) due to better
    cache locality.

    Args:
        sample_rate: Sample rate of the audio
        n_fft: Number of FFT points
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (defaults to sample_rate/2)

    Returns:
        Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    fft_freqs, hz_points, n_freqs = _prepare_mel_filter_bank_params(
        sample_rate, n_fft, n_mels, fmin, fmax
    )

    # Pre-allocate filter bank
    filter_bank = np.zeros((n_mels, n_freqs))

    # Pre-compute all divisors
    center_left_diffs = hz_points[1 : n_mels + 1] - hz_points[:n_mels]
    right_center_diffs = hz_points[2 : n_mels + 2] - hz_points[1 : n_mels + 1]

    # Vectorized filter creation
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Find indices only once and reuse
        # Rising edge
        rising_idx = np.where((fft_freqs >= left) & (fft_freqs < center))[0]
        if rising_idx.size > 0:
            rising_freqs = fft_freqs[rising_idx]  # Extract once
            filter_bank[i, rising_idx] = (rising_freqs - left) / center_left_diffs[i]

        # Falling edge
        falling_idx = np.where((fft_freqs >= center) & (fft_freqs < right))[0]
        if falling_idx.size > 0:
            falling_freqs = fft_freqs[falling_idx]  # Extract once
            filter_bank[i, falling_idx] = (right - falling_freqs) / right_center_diffs[
                i
            ]

    # Normalize filters in-place (vectorized)
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filter_bank *= enorm[:, np.newaxis]

    return filter_bank


def create_mel_filter_bank_vectorized(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Create a mel filter bank matrix using fully vectorized operations.

    This implementation uses NumPy broadcasting to eliminate all loops.
    It's more efficient for small to medium matrices (<82k elements).

    Args:
        sample_rate: Sample rate of the audio
        n_fft: Number of FFT points
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (defaults to sample_rate/2)

    Returns:
        Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    fft_freqs, hz_points, n_freqs = _prepare_mel_filter_bank_params(
        sample_rate, n_fft, n_mels, fmin, fmax
    )

    # Create 2D arrays for vectorized computation
    # Shape: (n_mels, n_freqs)
    freqs_2d = fft_freqs[np.newaxis, :]

    # Shape: (n_mels, 1)
    left_points = hz_points[:-2, np.newaxis]
    center_points = hz_points[1:-1, np.newaxis]
    right_points = hz_points[2:, np.newaxis]

    # Compute triangular filters using broadcasting with clip (more efficient than min/max)
    # Pre-compute divisors
    left_center_diffs = center_points - left_points
    right_center_diffs = right_points - center_points

    # Rising edge - use clip instead of nested min/max
    rising = np.clip((freqs_2d - left_points) / left_center_diffs, 0, 1)

    # Falling edge
    falling = np.clip((right_points - freqs_2d) / right_center_diffs, 0, 1)

    # Combine rising and falling edges
    filter_bank = np.minimum(rising, falling)

    # The filter_bank already has zeros outside the range due to clipping,
    # but we still need to ensure strict boundaries
    filter_bank = np.where(
        (freqs_2d >= left_points) & (freqs_2d < right_points), filter_bank, 0
    )

    # Normalize filters in-place
    enorm = 2.0 / (right_points.squeeze() - left_points.squeeze())
    filter_bank *= enorm[:, np.newaxis]

    return filter_bank


def create_mel_filter_bank_adaptive(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Create a mel filter bank matrix using the optimal method for the given size.

    This function automatically selects between loop-based and vectorized implementations
    based on the matrix size. The crossover point was empirically determined through
    benchmarking.

    Performance characteristics (measured on Apple M4):
    - Small matrices (<82,000 elements): Vectorized is 2-5x faster
    - Large matrices (>82,000 elements): Loop-based is 1.5-2.5x faster
    - Crossover occurs around 88.2kHz sample rate with typical settings

    Note: The exact crossover point may vary between platforms:
    - Apple Silicon (M1/M2/M3/M4): ~82,000 elements due to unified memory
    - Intel/AMD x86: May differ due to different cache hierarchies

    Args:
        sample_rate: Sample rate of the audio
        n_fft: Number of FFT points
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (defaults to sample_rate/2)

    Returns:
        Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    # Calculate matrix size
    n_freqs = n_fft // 2 + 1
    matrix_size = n_mels * n_freqs

    # Empirically determined crossover point (platform-dependent)
    # This value is optimized for Apple Silicon (M4), but will perform
    # not too shabby with other processors
    CROSSOVER_SIZE = 82000

    if matrix_size < CROSSOVER_SIZE:
        return create_mel_filter_bank_vectorized(sample_rate, n_fft, n_mels, fmin, fmax)
    else:
        return create_mel_filter_bank_loop(sample_rate, n_fft, n_mels, fmin, fmax)


def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Get center frequencies of mel bands.

    Args:
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz

    Returns:
        Array of center frequencies for each mel band without edge bins
    """
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    # Return center frequencies (skip edges)
    return hz_points[1:-1]


@dataclass(frozen=True)
class MelConfig:
    """Configuration for mel spectrogram parameters.

    This class manages mel spectrogram parameters based on sample rate
    and provides methods to calculate adaptive parameters.
    """

    # Base parameters (for 48kHz reference)
    BASE_SAMPLE_RATE: int = 48000
    BASE_FMIN: int = 50
    BASE_FMAX: int = 24000
    BASE_N_MELS: int = 96

    # Computed base range
    BASE_FREQ_RANGE: int = BASE_FMAX - BASE_FMIN  # 23950

    # Limits
    MIN_N_MELS: int = 80
    MAX_N_MELS: int = 110

    @classmethod
    def calculate_params(cls, sample_rate: int, fmin: float) -> dict:
        """Calculate mel parameters for a given sample rate.

        Args:
            sample_rate: Target sample rate in Hz
            fmin: Minimum frequency in Hz

        Returns:
            Dictionary with calculated parameters:
                - nyquist: Nyquist frequency
                - fmax: Maximum frequency (limited by Nyquist)
                - freq_range: Frequency range (fmax - fmin)
                - scale_factor: Scaling factor relative to base
                - n_mels: Number of mel bins (adaptive)
        """
        nyquist = sample_rate / 2

        # Adaptive fmax calculation to prevent empty mel filters
        if sample_rate <= 48000:
            # For standard rates, use BASE_FMAX or Nyquist
            fmax = min(nyquist, cls.BASE_FMAX)
        else:
            # For high sample rates, use 48% of sample rate to ensure valid mel filters
            # This prevents empty filter banks at frequencies like 192kHz
            fmax = sample_rate * 0.48

        # Ensure fmax doesn't exceed Nyquist (with small margin for numerical stability)
        fmax = min(fmax, nyquist - 1)

        freq_range = fmax - fmin

        # Calculate scale factor and adjust n_mels more conservatively for high rates
        if sample_rate <= 48000:
            scale_factor = freq_range / cls.BASE_FREQ_RANGE
            n_mels = max(
                cls.MIN_N_MELS, min(cls.MAX_N_MELS, int(cls.BASE_N_MELS * scale_factor))
            )
        else:
            # For high sample rates, use logarithmic scaling to prevent too many mel bins
            # This ensures mel filters have enough frequency coverage
            scale_factor = np.log2(sample_rate / cls.BASE_SAMPLE_RATE)
            # Keep n_mels moderate to ensure each filter has enough frequency range
            n_mels = min(
                int(cls.BASE_N_MELS * (1 + scale_factor * 0.25)), cls.MAX_N_MELS
            )

        return {
            "nyquist": nyquist,
            "fmax": fmax,
            "freq_range": freq_range,
            "scale_factor": scale_factor,
            "n_mels": n_mels,
        }


class MelSpectrogramProcessor(AudioProcessor[Tuple[np.ndarray, Optional[float]]]):
    """Processes audio to generate mel spectrograms using optimized numpy operations.

    This processor converts audio signals to mel-scale spectrograms,
    which provide a perceptually-motivated frequency representation
    of audio. Mel spectrograms are commonly used for speech visualization
    and analysis.

    The mel scale approximates human auditory perception, with higher
    resolution at lower frequencies where speech information is concentrated.

    Attributes:
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        mel_filter: Pre-computed mel filterbank matrix
        window: Pre-computed Hanning window
    """

    def __init__(
        self,
        sample_rate: int = AudioConstants.DEFAULT_SAMPLE_RATE,
        n_fft: int = AudioConstants.N_FFT,
        hop_length: int = AudioConstants.HOP_LENGTH,
        n_mels: int = AudioConstants.N_MELS,
        fmin: float = AudioConstants.FMIN,
        fmax: float = AudioConstants.FMAX,
    ):
        """Initialize the mel spectrogram processor.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size (default: 2048)
            hop_length: Hop between frames (default: 512)
            n_mels: Number of mel bands (default: 80)
            fmin: Minimum frequency in Hz (default: 0)
            fmax: Maximum frequency in Hz (default: 8000)

        Note:
            The processor automatically selects the optimal mel filter bank
            implementation based on matrix size. For sample rates ≤48kHz,
            a fully vectorized approach is used. For higher rates (≥88.2kHz),
            a loop-based approach is more efficient on Apple Silicon.
        """
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Pre-compute mel filterbank for efficiency
        # Clamp fmax to Nyquist frequency if needed
        actual_fmax = min(fmax, sample_rate / 2)

        # Use adaptive method that automatically chooses the best implementation
        # based on matrix size (platform-optimized for Apple Silicon M1-M4)
        self.mel_filter = create_mel_filter_bank_adaptive(
            sample_rate, n_fft, n_mels, fmin, actual_fmax
        )

        self.actual_fmax = actual_fmax

        # Pre-compute mel frequencies for efficiency
        self.mel_frequencies = mel_frequencies(n_mels, fmin, actual_fmax)

        # Pre-compute window for efficiency (avoid recreating every time)
        self.window = np.hanning(n_fft)

        # Pre-compute frequency per bin for highest frequency detection
        self.freq_per_bin = sample_rate / n_fft

    @classmethod
    def create_for(
        cls, sample_rate: int, fmin: float = None
    ) -> Tuple["MelSpectrogramProcessor", int]:
        """Create processor with adaptive parameters for sample rate.

        Args:
            sample_rate: Target sample rate in Hz
            fmin: Minimum frequency (uses MelConfig.BASE_FMIN if None)

        Returns:
            Tuple of (processor, n_mels)
        """
        if fmin is None:
            fmin = MEL_CONFIG.BASE_FMIN

        params = MEL_CONFIG.calculate_params(sample_rate, fmin)
        processor = cls(
            sample_rate=sample_rate,
            n_mels=params["n_mels"],
            fmin=fmin,
            fmax=params["fmax"],
        )
        return processor, params["n_mels"]

    def process(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Convert audio frame to mel-scale dB values and detect the highest frequency.

        Optimized version with pre-computed window and efficient operations.

        Args:
            audio_data: Audio frame (n_fft samples)

        Returns:
            Tuple of:
                - np.ndarray: Mel-scale magnitudes in dB (n_mels values)
                - float: Highest frequency with significant energy (Hz) or None

        Note:
            Automatically detects if input is already normalized (max <= 1.0)
            to handle both live recording and loaded audio files correctly.
        """
        # Use centralized normalization function
        audio_norm = normalize_audio(audio_data)

        # Apply pre-computed window (reuse if audio length matches)
        if len(audio_norm) == self.n_fft:
            windowed = audio_norm * self.window
        else:
            # Fallback for different lengths
            windowed = audio_norm * np.hanning(len(audio_norm))

        # Compute FFT
        fft = np.fft.rfft(windowed, n=self.n_fft)

        # Compute power spectrum more efficiently
        # Use np.real(fft * np.conj(fft)) which is faster than np.abs(fft) ** 2
        power = np.real(fft * np.conj(fft))

        # Apply mel filterbank
        mel_power = np.dot(self.mel_filter, power[: self.n_fft // 2 + 1])

        # Convert to dB using log10
        # Combine operations to reduce memory access
        mel_db = AudioConstants.POWER_TO_DB_FACTOR * np.log10(
            mel_power + AudioConstants.DB_REFERENCE
        )

        # Clamp to display range in-place
        np.clip(mel_db, AudioConstants.DB_MIN, 0, out=mel_db)

        # Detect the highest frequency with significant energy
        # Compute power_db only for frequency detection (not for mel)
        power_db = AudioConstants.POWER_TO_DB_FACTOR * np.log10(
            power[: self.n_fft // 2 + 1] + AudioConstants.DB_REFERENCE
        )

        # Find the highest frequency above noise floor
        significant_bins = np.where(power_db > AudioConstants.FREQUENCY_NOISE_FLOOR_DB)[
            0
        ]

        if significant_bins.size > 0:
            highest_bin = int(significant_bins[-1])
            # Convert bin to frequency
            highest_freq = float(
                min(highest_bin * self.freq_per_bin, self.sample_rate / 2)
            )
        else:
            highest_freq = None

        return mel_db, highest_freq


# Global configuration instance
MEL_CONFIG = MelConfig()
