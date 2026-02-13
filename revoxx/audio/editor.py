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

        if (
            fade_samples > 0
            and len(before) >= fade_samples
            and len(after) >= fade_samples
        ):
            # Apply cross-fade between the end of 'before' and start of 'after'
            before_fade = before[-fade_samples:]
            after_fade = after[:fade_samples]

            crossfaded = AudioEditor._equal_power_crossfade(
                before_fade, after_fade, fade_samples
            )

            # Construct result: before (minus fade region) + crossfade + after (minus fade region)
            result = np.concatenate(
                [before[:-fade_samples], crossfaded, after[fade_samples:]]
            )
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
            result = AudioEditor._splice_with_crossfade(
                before, insert, after, fade_samples
            )
        else:
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
            result = AudioEditor._splice_with_crossfade(
                before, replacement, after, fade_samples
            )
        else:
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
    def _splice_with_crossfade(
        before: np.ndarray,
        insert: np.ndarray,
        after: np.ndarray,
        fade_samples: int,
    ) -> np.ndarray:
        """Splice three audio segments with crossfade at boundaries.

        Combines before + insert + after with equal-power crossfade at
        the junction points to avoid clicks.

        Args:
            before: Audio segment before the insert point
            insert: Audio segment to insert
            after: Audio segment after the insert point
            fade_samples: Number of samples for crossfade at each boundary

        Returns:
            Combined audio array with crossfaded transitions
        """
        result_parts = []

        # Crossfade at start (before -> insert)
        if len(before) >= fade_samples and len(insert) >= fade_samples:
            crossfaded_start = AudioEditor._equal_power_crossfade(
                before[-fade_samples:], insert[:fade_samples], fade_samples
            )
            result_parts.append(before[:-fade_samples])
            result_parts.append(crossfaded_start)
        else:
            result_parts.append(before)

        # Middle section of insert
        AudioEditor._append_middle_section(result_parts, insert, fade_samples)

        # Crossfade at end (insert -> after)
        if len(after) >= fade_samples and len(insert) >= fade_samples:
            crossfaded_end = AudioEditor._equal_power_crossfade(
                insert[-fade_samples:], after[:fade_samples], fade_samples
            )
            result_parts.append(crossfaded_end)
            result_parts.append(after[fade_samples:])
        else:
            result_parts.append(after)

        return np.concatenate([p for p in result_parts if len(p) > 0])

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

    @staticmethod
    def _append_middle_section(
        result_parts: list, audio: np.ndarray, fade_samples: int
    ) -> None:
        """Append the middle section of audio, excluding fade regions.

        Used during insert/replace operations to add the portion of audio
        between the start and end crossfade regions.

        Args:
            result_parts: List to append the middle section to
            audio: Audio array to extract middle section from
            fade_samples: Number of samples used for crossfade at each end
        """
        if len(audio) > 2 * fade_samples:
            result_parts.append(audio[fade_samples:-fade_samples])
        elif len(audio) > fade_samples:
            result_parts.append(audio[fade_samples:])

    @staticmethod
    def loop_audio_for_duration(
        audio: np.ndarray, target_samples: int, sample_rate: int
    ) -> np.ndarray:
        """Loop audio to achieve target duration with cross-fade at splice points.

        If the source audio is longer than needed, it is trimmed.
        If shorter, it is looped with equal-power cross-fade at each loop point.

        Args:
            audio: Source audio to loop
            target_samples: Target number of samples
            sample_rate: Audio sample rate in Hz

        Returns:
            Audio array of exactly target_samples length
        """
        if len(audio) == 0:
            return np.zeros(target_samples, dtype=np.float32)

        if len(audio) >= target_samples:
            return audio[:target_samples].astype(np.float32)

        fade_samples = AudioEditor._calculate_fade_samples(sample_rate, len(audio))

        if fade_samples < 2:
            repeats = (target_samples // len(audio)) + 1
            return np.tile(audio, repeats)[:target_samples].astype(np.float32)

        # Build looped audio with crossfades
        result = []
        remaining = target_samples
        is_first = True

        while remaining > 0:
            if is_first:
                remaining = AudioEditor._append_first_segment(
                    result, audio, fade_samples, remaining
                )
                is_first = False
            elif remaining >= len(audio):
                remaining = AudioEditor._append_full_loop(
                    result, audio, fade_samples, remaining
                )
            else:
                remaining = AudioEditor._append_final_segment(
                    result, audio, fade_samples, remaining
                )

        output = np.concatenate(result)
        return AudioEditor._ensure_exact_length(output, target_samples)

    @staticmethod
    def _append_first_segment(
        result: list, audio: np.ndarray, fade_samples: int, remaining: int
    ) -> int:
        """Append first segment of looped audio.

        The first segment excludes the final fade_samples to leave room
        for crossfading with the next loop iteration.

        Args:
            result: List to append audio segments to
            audio: Source audio array to loop
            fade_samples: Number of samples reserved for crossfade
            remaining: Number of samples still needed

        Returns:
            Updated remaining sample count
        """
        if remaining >= len(audio):
            result.append(audio[:-fade_samples].copy())
            return remaining - (len(audio) - fade_samples)
        else:
            result.append(audio[:remaining].copy())
            return 0

    @staticmethod
    def _append_full_loop(
        result: list, audio: np.ndarray, fade_samples: int, remaining: int
    ) -> int:
        """Append a full loop iteration with crossfade.

        Creates a seamless loop by crossfading the end of the previous
        iteration with the beginning of this one, then appends the
        middle portion of the audio.

        Args:
            result: List to append audio segments to
            audio: Source audio array to loop
            fade_samples: Number of samples for crossfade region
            remaining: Number of samples still needed

        Returns:
            Updated remaining sample count
        """
        crossfade = AudioEditor._equal_power_crossfade(
            audio[-fade_samples:], audio[:fade_samples], fade_samples
        )
        result.append(crossfade)
        remaining -= fade_samples

        middle_samples = len(audio) - 2 * fade_samples
        if middle_samples > 0:
            if remaining >= middle_samples:
                result.append(audio[fade_samples:-fade_samples].copy())
                remaining -= middle_samples
            else:
                result.append(audio[fade_samples : fade_samples + remaining].copy())
                remaining = 0

        return remaining

    @staticmethod
    def _append_final_segment(
        result: list, audio: np.ndarray, fade_samples: int, remaining: int
    ) -> int:
        """Append the final partial segment with crossfade.

        Handles the case where the remaining samples are less than a full
        loop iteration. Crossfades at the loop point and takes only the
        needed portion.

        Args:
            result: List to append audio segments to
            audio: Source audio array to loop
            fade_samples: Number of samples for crossfade region
            remaining: Number of samples still needed

        Returns:
            Updated remaining sample count (always 0 after this)
        """
        crossfade = AudioEditor._equal_power_crossfade(
            audio[-fade_samples:], audio[:fade_samples], fade_samples
        )

        if remaining >= fade_samples:
            result.append(crossfade)
            remaining -= fade_samples
            if remaining > 0:
                take = min(remaining, len(audio) - fade_samples)
                result.append(audio[fade_samples : fade_samples + take].copy())
                remaining -= take
        else:
            result.append(crossfade[:remaining].copy())
            remaining = 0

        return remaining

    @staticmethod
    def _ensure_exact_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Ensure audio has exactly the target length.

        Trims if too long, zero-pads if too short.

        Args:
            audio: Audio array to adjust
            target_samples: Exact number of samples required

        Returns:
            Audio array with exactly target_samples length as float32
        """
        if len(audio) > target_samples:
            return audio[:target_samples].astype(np.float32)
        elif len(audio) < target_samples:
            return np.pad(audio, (0, target_samples - len(audio))).astype(np.float32)
        return audio.astype(np.float32)
