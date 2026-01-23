"""Audio editing operations with cross-fade support.

This module provides audio editing functions for deleting, inserting,
and replacing audio segments with smooth cross-fade transitions.
"""

import numpy as np

from ..constants import AudioConstants


class AudioEditor:
    """Provides audio editing operations with cross-fade support.

    All operations use equal-power cross-fading to ensure smooth
    transitions without audible clicks or pops.
    """

    @staticmethod
    def delete_range(
        audio: np.ndarray, start_sample: int, end_sample: int, sample_rate: int
    ) -> np.ndarray:
        """Delete a range of audio samples with cross-fade.

        Args:
            audio: Input audio array
            start_sample: Start of deletion range
            end_sample: End of deletion range
            sample_rate: Audio sample rate in Hz

        Returns:
            New audio array with the range deleted and cross-faded
        """
        if start_sample >= end_sample:
            return audio.copy()

        if start_sample < 0:
            start_sample = 0
        if end_sample > len(audio):
            end_sample = len(audio)

        # Calculate fade samples
        selection_samples = end_sample - start_sample
        fade_samples = AudioEditor._calculate_fade_samples(
            sample_rate, selection_samples
        )

        # Get the parts before and after the deletion
        before = audio[:start_sample]
        after = audio[end_sample:]

        if fade_samples > 0 and len(before) >= fade_samples and len(after) >= fade_samples:
            # Apply cross-fade between the end of 'before' and start of 'after'
            before_fade = before[-fade_samples:]
            after_fade = after[:fade_samples]

            crossfaded = AudioEditor._equal_power_crossfade(
                before_fade, after_fade, fade_samples
            )

            # Construct result: before (minus fade region) + crossfade + after (minus fade region)
            result = np.concatenate([
                before[:-fade_samples],
                crossfaded,
                after[fade_samples:]
            ])
        else:
            # No cross-fade possible, just concatenate
            result = np.concatenate([before, after])

        return result

    @staticmethod
    def insert_at_position(
        original: np.ndarray,
        insert: np.ndarray,
        position: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Insert audio at a position with cross-fade.

        Args:
            original: Original audio array
            insert: Audio to insert
            position: Sample position to insert at
            sample_rate: Audio sample rate in Hz

        Returns:
            New audio array with the inserted content cross-faded
        """
        if len(insert) == 0:
            return original.copy()

        if position < 0:
            position = 0
        if position > len(original):
            position = len(original)

        # Calculate fade samples based on insert length
        fade_samples = AudioEditor._calculate_fade_samples(sample_rate, len(insert))

        before = original[:position]
        after = original[position:]

        if fade_samples > 0:
            # Cross-fade at insertion point
            result_parts = []

            # Before section
            if len(before) >= fade_samples:
                # Fade out the end of 'before' and fade in start of 'insert'
                before_fade = before[-fade_samples:]
                insert_start_fade = insert[:fade_samples] if len(insert) >= fade_samples else insert

                if len(insert_start_fade) == fade_samples:
                    crossfaded_start = AudioEditor._equal_power_crossfade(
                        before_fade, insert_start_fade, fade_samples
                    )
                    result_parts.append(before[:-fade_samples])
                    result_parts.append(crossfaded_start)
                else:
                    result_parts.append(before)
            else:
                result_parts.append(before)

            # Middle section of insert (if any)
            if len(insert) > 2 * fade_samples:
                result_parts.append(insert[fade_samples:-fade_samples])
            elif len(insert) > fade_samples:
                result_parts.append(insert[fade_samples:])

            # After section
            if len(after) >= fade_samples and len(insert) >= fade_samples:
                # Fade out end of 'insert' and fade in start of 'after'
                insert_end_fade = insert[-fade_samples:]
                after_fade = after[:fade_samples]

                crossfaded_end = AudioEditor._equal_power_crossfade(
                    insert_end_fade, after_fade, fade_samples
                )
                result_parts.append(crossfaded_end)
                result_parts.append(after[fade_samples:])
            else:
                result_parts.append(after)

            result = np.concatenate([p for p in result_parts if len(p) > 0])
        else:
            # No cross-fade, simple concatenation
            result = np.concatenate([before, insert, after])

        return result

    @staticmethod
    def replace_range(
        original: np.ndarray,
        replacement: np.ndarray,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Replace a range of audio with new content and cross-fade.

        Args:
            original: Original audio array
            replacement: Audio to replace the range with
            start_sample: Start of range to replace
            end_sample: End of range to replace
            sample_rate: Audio sample rate in Hz

        Returns:
            New audio array with the range replaced and cross-faded
        """
        if start_sample >= end_sample:
            return original.copy()

        if start_sample < 0:
            start_sample = 0
        if end_sample > len(original):
            end_sample = len(original)

        # Calculate fade samples
        selection_samples = end_sample - start_sample
        fade_samples = AudioEditor._calculate_fade_samples(
            sample_rate, min(selection_samples, len(replacement))
        )

        before = original[:start_sample]
        after = original[end_sample:]

        if fade_samples > 0:
            result_parts = []

            # Cross-fade at start of replacement
            if len(before) >= fade_samples and len(replacement) >= fade_samples:
                before_fade = before[-fade_samples:]
                replacement_start = replacement[:fade_samples]

                crossfaded_start = AudioEditor._equal_power_crossfade(
                    before_fade, replacement_start, fade_samples
                )
                result_parts.append(before[:-fade_samples])
                result_parts.append(crossfaded_start)
            else:
                result_parts.append(before)

            # Middle of replacement
            if len(replacement) > 2 * fade_samples:
                result_parts.append(replacement[fade_samples:-fade_samples])
            elif len(replacement) > fade_samples:
                result_parts.append(replacement[fade_samples:])

            # Cross-fade at end of replacement
            if len(after) >= fade_samples and len(replacement) >= fade_samples:
                replacement_end = replacement[-fade_samples:]
                after_start = after[:fade_samples]

                crossfaded_end = AudioEditor._equal_power_crossfade(
                    replacement_end, after_start, fade_samples
                )
                result_parts.append(crossfaded_end)
                result_parts.append(after[fade_samples:])
            else:
                result_parts.append(after)

            result = np.concatenate([p for p in result_parts if len(p) > 0])
        else:
            # No cross-fade, simple replacement
            result = np.concatenate([before, replacement, after])

        return result

    @staticmethod
    def _equal_power_crossfade(
        audio_a: np.ndarray, audio_b: np.ndarray, fade_samples: int
    ) -> np.ndarray:
        """Apply equal-power cross-fade between two audio segments.

        Equal-power cross-fade maintains constant perceived loudness
        during the transition by using sine/cosine curves for gain.

        Args:
            audio_a: First audio segment (fade out)
            audio_b: Second audio segment (fade in)
            fade_samples: Number of samples for the fade

        Returns:
            Cross-faded audio segment of length fade_samples
        """
        if fade_samples <= 0:
            return audio_b[:0] if len(audio_b) > 0 else np.array([])

        # Ensure we have enough samples
        actual_samples = min(fade_samples, len(audio_a), len(audio_b))
        if actual_samples <= 0:
            return np.array([])

        # Equal power cross-fade using sine/cosine curves
        t = np.linspace(0, np.pi / 2, actual_samples)
        gain_a = np.cos(t)  # Fade out
        gain_b = np.sin(t)  # Fade in

        result = audio_a[:actual_samples] * gain_a + audio_b[:actual_samples] * gain_b

        return result

    @staticmethod
    def _calculate_fade_samples(sample_rate: int, selection_samples: int) -> int:
        """Calculate the number of samples for cross-fade.

        The fade length is adaptive: it uses the configured cross-fade duration
        but is capped at half the selection length to ensure smooth transitions
        for short selections.

        Args:
            sample_rate: Audio sample rate in Hz
            selection_samples: Number of samples in the selection

        Returns:
            Number of samples to use for cross-fade
        """
        # Calculate fade samples from configured duration
        fade_from_config = int(AudioConstants.CROSSFADE_MS * sample_rate / 1000)

        # Cap at half the selection length
        max_fade = selection_samples // 2

        return min(fade_from_config, max_fade)
