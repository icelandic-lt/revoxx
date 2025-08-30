"""Audio processors module.

This module provides audio processing components for the Revoxx Recorder.
"""

from .processor_base import AudioProcessor
from .clipping_detector import ClippingDetector
from .mel_spectrogram import (
    MelSpectrogramProcessor,
    MEL_CONFIG,
    mel_frequencies,
    create_mel_filter_bank_adaptive,
    create_mel_filter_bank_vectorized,
    create_mel_filter_bank_loop,
)

__all__ = [
    "AudioProcessor",
    "ClippingDetector",
    "MelSpectrogramProcessor",
    "MEL_CONFIG",
    "mel_frequencies",
    "create_mel_filter_bank_adaptive",
    "create_mel_filter_bank_vectorized",
    "create_mel_filter_bank_loop",
]
