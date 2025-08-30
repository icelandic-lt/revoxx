"""Frequency axis management for mel spectrograms."""

from typing import List, Tuple
import numpy as np
from matplotlib.axes import Axes

from ..constants import UIConstants
from ..audio.processors import MEL_CONFIG
from ..audio.processors.mel_spectrogram import mel_frequencies as get_mel_frequencies


class FrequencyAxisManager:
    """Manages frequency axis display for mel spectrograms.

    Handles frequency tick calculation, label formatting, and special
    highlighting (e.g., maximum detected frequency in orange).
    """

    # Minimum separation between tick labels as fraction of axis height
    MIN_TICK_SEPARATION_RATIO = 0.03  # 3% of axis height

    def __init__(self, ax: Axes):
        """Initialize frequency axis manager.

        Args:
            ax: Matplotlib axes to manage
        """
        self.ax = ax
        self._peak_indicator_position = None  # Track current peak indicator position
        self._base_ticks = []  # Store base frequency ticks
        self._base_labels = []  # Store base frequency labels

    def update_default_axis(self, n_mels: int, fmin: float, fmax: float) -> None:
        """Update frequency axis with default settings.

        Args:
            n_mels: Number of mel bins
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
        """
        # Reset peak indicator when updating axis
        self._peak_indicator_position = None

        mel_freqs = self._get_mel_frequencies(n_mels, fmin, fmax)
        ticks, labels = self._calculate_ticks_and_labels(mel_freqs, fmax)
        self._apply_ticks_and_labels(ticks, labels)
        self._reset_label_styles()

    def update_recording_axis(self, sample_rate: int, fmin: float) -> Tuple[int, float]:
        """Update frequency axis for a specific recording.

        Args:
            sample_rate: Recording sample rate in Hz
            fmin: Minimum frequency in Hz

        Returns:
            Tuple of (adaptive_n_mels, adaptive_fmax)
        """
        # Reset peak indicator when updating axis
        self._peak_indicator_position = None

        params = MEL_CONFIG.calculate_params(sample_rate, fmin)
        adaptive_n_mels = params["n_mels"]
        adaptive_fmax = params["fmax"]

        # Update axis
        mel_freqs = self._get_mel_frequencies(adaptive_n_mels, fmin, adaptive_fmax)
        # For display, use the exact Nyquist frequency
        display_fmax = sample_rate / 2
        ticks, labels = self._calculate_ticks_and_labels(mel_freqs, display_fmax)
        self._apply_ticks_and_labels(ticks, labels)
        self._reset_label_styles()

        return adaptive_n_mels, adaptive_fmax

    def highlight_max_frequency(
        self, max_freq: float, n_mels: int, fmin: float, fmax: float
    ) -> None:
        """Add or update orange highlight for maximum detected frequency.

        Args:
            max_freq: Maximum detected frequency in Hz
            n_mels: Number of mel bins
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
        """
        if max_freq <= 0 or not self._base_ticks:
            return

        # Find mel bin for max frequency
        mel_freqs = self._get_mel_frequencies(n_mels, fmin, fmax)
        max_freq_bin = np.argmin(np.abs(mel_freqs - max_freq))

        # Only update if peak position has changed significantly
        if (
            self._peak_indicator_position is not None
            and abs(max_freq_bin - self._peak_indicator_position) < 0.5
        ):
            return  # Peak hasn't moved enough to warrant update

        # Start with base ticks and labels
        all_ticks = self._base_ticks.copy()
        all_labels = self._base_labels.copy()

        # Check if we need to replace a nearby tick
        min_distance = n_mels / 20  # 5% separation
        replace_idx = None

        for i, tick in enumerate(all_ticks):
            if abs(tick - max_freq_bin) < min_distance:
                replace_idx = i
                break

        if 0 <= max_freq_bin < n_mels:
            if replace_idx is not None:
                # Replace nearby tick
                all_ticks[replace_idx] = max_freq_bin
                all_labels[replace_idx] = self._format_frequency(max_freq)
            else:
                # Add new tick at correct position
                insert_idx = 0
                for i, tick in enumerate(all_ticks):
                    if tick > max_freq_bin:
                        insert_idx = i
                        break
                else:
                    insert_idx = len(all_ticks)

                all_ticks.insert(insert_idx, max_freq_bin)
                all_labels.insert(insert_idx, self._format_frequency(max_freq))

            # Filter ticks with priority for max frequency
            filtered_ticks, filtered_labels = self._filter_ticks_with_priority(
                all_ticks, all_labels, max_freq_bin, n_mels
            )

            # Apply updates
            self.ax.set_yticks(filtered_ticks)
            self.ax.set_yticklabels(filtered_labels)

            # Store the new peak indicator position
            self._peak_indicator_position = max_freq_bin

            # Apply highlighting to the max frequency tick
            self._highlight_specific_tick(filtered_ticks, max_freq_bin)

    @staticmethod
    def _get_mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
        """Get mel frequency values for each bin."""
        return get_mel_frequencies(n_mels, fmin, fmax)

    def _filter_overlapping_ticks(
        self, tick_positions: np.ndarray, n_mels: int
    ) -> np.ndarray:
        """Filter out tick positions that would cause label overlap.

        Args:
            tick_positions: Array of tick positions in mel bins
            n_mels: Total number of mel bins

        Returns:
            Filtered array of tick positions
        """
        # Minimum separation in mel bins (roughly corresponds to label height)
        min_separation = n_mels * self.MIN_TICK_SEPARATION_RATIO

        filtered = [tick_positions[0]]  # Always keep first tick

        # Check middle ticks
        for i in range(1, len(tick_positions) - 1):
            if tick_positions[i] - filtered[-1] >= min_separation:
                filtered.append(tick_positions[i])

        # Always keep last tick if it's far enough from the previous
        if len(tick_positions) > 1:
            if (
                len(filtered) == 1
                or tick_positions[-1] - filtered[-1] >= min_separation
            ):
                filtered.append(tick_positions[-1])
            elif tick_positions[-1] != filtered[-1]:
                # Replace the last filtered tick with the actual last tick
                # to ensure we always show the maximum frequency
                filtered[-1] = tick_positions[-1]

        return np.array(filtered)

    @staticmethod
    def _filter_ticks_with_priority(
        all_ticks: list, all_labels: list, priority_tick: float, n_mels: int
    ) -> Tuple[list, list]:
        """Filter overlapping ticks while ensuring a priority tick is always shown.

        Args:
            all_ticks: List of all tick positions
            all_labels: List of all tick labels
            priority_tick: Tick position that must be kept (e.g., max frequency)
            n_mels: Total number of mel bins

        Returns:
            Tuple of (filtered_ticks, filtered_labels)
        """
        min_separation = n_mels * FrequencyAxisManager.MIN_TICK_SEPARATION_RATIO
        filtered_ticks = []
        filtered_labels = []

        for i, tick in enumerate(all_ticks):
            if tick == priority_tick:
                # Always add the priority tick
                filtered_ticks.append(tick)
                filtered_labels.append(all_labels[i])
            else:
                # Check if this tick is far enough from priority tick and other filtered ticks
                too_close = False

                # Check distance from priority tick
                if abs(tick - priority_tick) < min_separation:
                    too_close = True

                # Check distance from already filtered ticks
                if not too_close:
                    for filtered_tick in filtered_ticks:
                        if abs(tick - filtered_tick) < min_separation:
                            too_close = True
                            break

                if not too_close:
                    filtered_ticks.append(tick)
                    filtered_labels.append(all_labels[i])

        # Sort ticks and labels together to maintain order
        if filtered_ticks:
            sorted_pairs = sorted(zip(filtered_ticks, filtered_labels))
            filtered_ticks, filtered_labels = zip(*sorted_pairs)
            filtered_ticks = list(filtered_ticks)
            filtered_labels = list(filtered_labels)

        return filtered_ticks, filtered_labels

    def _calculate_ticks_and_labels(
        self, mel_freqs: np.ndarray, fmax: float
    ) -> Tuple[np.ndarray, List[str]]:
        """Calculate tick positions and labels."""
        n_mels = len(mel_freqs)
        n_ticks = UIConstants.N_FREQUENCY_TICKS

        # Split ticks: more in lower frequencies
        lower_ticks = int(n_ticks * UIConstants.FREQ_TICKS_LOWER_FRACTION)
        upper_ticks = n_ticks - lower_ticks

        # Calculate indices
        lower_indices = np.linspace(0, n_mels // 3, lower_ticks, dtype=int)
        upper_indices = np.linspace(n_mels // 3 + 1, n_mels - 1, upper_ticks, dtype=int)

        log_indices = np.unique(np.concatenate([lower_indices, upper_indices]))
        log_indices[0] = 0
        log_indices[-1] = n_mels - 1

        # Filter out overlapping ticks
        log_indices = self._filter_overlapping_ticks(log_indices, n_mels)

        # Create labels
        labels = []
        for i, idx in enumerate(log_indices):
            freq = mel_freqs[idx]
            # Show exact Nyquist for last tick
            if i == len(log_indices) - 1:
                freq = fmax
            labels.append(self._format_frequency(freq))

        return log_indices, labels

    @staticmethod
    def _format_frequency(freq: float) -> str:
        """Format frequency value for display."""
        if freq < 1000:
            return f"{int(freq)}"
        elif freq == int(freq / 1000) * 1000:  # Round kHz
            return f"{int(freq/1000)}k"
        else:
            return f"{freq/1000:.1f}k"

    def _apply_ticks_and_labels(self, ticks: np.ndarray, labels: List[str]) -> None:
        """Apply ticks and labels to axis."""
        # Store base ticks and labels for later use
        self._base_ticks = list(ticks)
        self._base_labels = list(labels)

        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)

    def _reset_label_styles(self) -> None:
        """Reset all labels to default color and weight."""
        for label in self.ax.get_yticklabels():
            label.set_color(UIConstants.COLOR_TEXT_SECONDARY)
            label.set_weight("normal")

    def _highlight_specific_tick(self, ticks: list, target_tick: float) -> None:
        """Highlight a specific tick label with orange color and bold weight.

        Args:
            ticks: List of tick positions
            target_tick: The tick position to highlight
        """
        # First reset all labels to default style
        for label in self.ax.get_yticklabels():
            label.set_color(UIConstants.COLOR_TEXT_SECONDARY)
            label.set_weight("normal")

        # Then highlight the target tick
        for i, tick in enumerate(ticks):
            if tick == target_tick:
                labels = self.ax.get_yticklabels()
                if i < len(labels):
                    labels[i].set_color("orange")
                    labels[i].set_weight("bold")
                break
