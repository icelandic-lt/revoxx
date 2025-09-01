"""Text import utilities for converting raw text to Festival script format."""

import re
import numpy as np
from pathlib import Path
from typing import List, Optional


class TextImporter:
    """Utilities for importing and processing raw text files."""

    @staticmethod
    def truncated_normal(
        size: int, mean: float, std_dev: float, min_val: float, max_val: float
    ) -> np.ndarray:
        """Generate truncated normal distribution without scipy.

        Args:
            size: Number of samples to generate
            mean: Mean of the distribution
            std_dev: Standard deviation
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Array of samples from truncated normal distribution
        """
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")

        # Generate more samples than needed to account for truncation
        oversample_factor = 10
        samples = []

        while len(samples) < size:
            # Generate batch of normal samples
            batch = np.random.normal(mean, std_dev, size * oversample_factor)

            # Keep only samples within bounds
            valid = batch[(batch >= min_val) & (batch <= max_val)]
            samples.extend(valid)

        # Return requested number of samples, rounded to integers
        return np.round(samples[:size]).astype(int)

    @staticmethod
    def calculate_truncated_normal_pdf(
        x: np.ndarray, mean: float, std_dev: float, min_val: float, max_val: float
    ) -> np.ndarray:
        """Calculate approximate PDF (Probability Density Function) for truncated normal distribution.

        Args:
            x: Points at which to evaluate PDF
            mean: Mean of the distribution
            std_dev: Standard deviation
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            PDF values at x
        """
        # Standard normal PDF
        pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std_dev) ** 2
        )

        # Zero out values outside bounds
        pdf[x < min_val] = 0
        pdf[x > max_val] = 0

        # Normalize to account for truncation
        # Approximate normalization constant using CDF (Cumulative Distribution Function) difference
        # For simplicity, use numerical integration
        x_norm = np.linspace(min_val, max_val, 1000)
        pdf_norm = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_norm - mean) / std_dev) ** 2
        )
        norm_constant = np.trapezoid(pdf_norm, x_norm)

        if norm_constant > 0:
            pdf = pdf / norm_constant

        return pdf

    @staticmethod
    def split_long_sentence(sentence: str, max_length: int) -> List[str]:
        """Split a long sentence into smaller chunks.

        Args:
            sentence: Sentence to split
            max_length: Maximum length for each chunk

        Returns:
            List of sentence chunks
        """
        if len(sentence) <= max_length:
            return [sentence]

        chunks = []

        # First try to split at punctuation
        punctuation_pattern = r"[,;:]"
        parts = re.split(f"({punctuation_pattern})", sentence)

        current_chunk = ""
        for i, part in enumerate(parts):
            # Check if this is punctuation (odd indices after split)
            if i % 2 == 1:
                # Add punctuation to current chunk
                current_chunk += part
            else:
                # This is text
                if len(current_chunk) + len(part) <= max_length:
                    current_chunk += part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # If part itself is too long, split by words
                    if len(part) > max_length:
                        words = part.split()
                        word_chunk = ""
                        for word in words:
                            if len(word_chunk) + len(word) + 1 <= max_length:
                                word_chunk = (word_chunk + " " + word).strip()
                            else:
                                if word_chunk:
                                    chunks.append(word_chunk)
                                word_chunk = word
                        current_chunk = word_chunk
                    else:
                        current_chunk = part

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def _split_into_sentences(text: str, max_length: int) -> List[str]:
        """Split text into individual sentences, respecting max length.

        Each sentence becomes a separate utterance. Long sentences are split
        at punctuation or word boundaries.

        Args:
            text: Text to split
            max_length: Maximum length for each utterance

        Returns:
            List of sentences
        """
        # Clean up whitespace and line breaks within text
        text = " ".join(text.split())

        # Split on sentence boundaries
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        utterances = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Each sentence becomes its own utterance
            if len(sentence) <= max_length:
                utterances.append(sentence)
            else:
                # Split long sentence at punctuation or words
                parts = TextImporter.split_long_sentence(sentence, max_length)
                utterances.extend(parts)

        return utterances

    @staticmethod
    def split_text(
        text: str, max_length: int, split_mode: str = "sentences"
    ) -> List[str]:
        """Split text into utterances respecting boundaries.

        Args:
            text: Text to split
            max_length: Maximum length for each utterance
            split_mode: "sentences", "lines", or "paragraphs"

        Returns:
            List of utterances
        """
        if not text:
            return []

        if split_mode == "paragraphs":
            # Split by paragraphs (double newlines or more)
            paragraphs = re.split(r"\n\s*\n", text)
            utterances = []
            for para in paragraphs:
                para = " ".join(para.split())  # Clean whitespace within paragraph
                if para:
                    if len(para) <= max_length:
                        utterances.append(para)
                    else:
                        # Split long paragraphs at sentence boundaries
                        utterances.extend(
                            TextImporter._split_into_sentences(para, max_length)
                        )
            return utterances

        elif split_mode == "lines":
            lines = text.split("\n")
            utterances = []
            for line in lines:
                line = line.strip()
                if line:
                    if len(line) <= max_length:
                        utterances.append(line)
                    else:
                        utterances.extend(
                            TextImporter.split_long_sentence(line, max_length)
                        )
            return utterances

        # Default: split on sentence boundaries
        return TextImporter._split_into_sentences(text, max_length)

    @staticmethod
    def generate_labels(count: int, prefix: str = "utt") -> List[str]:
        """Generate unique labels for utterances.

        Args:
            count: Number of labels to generate
            prefix: Prefix for labels

        Returns:
            List of labels
        """
        return [f"{prefix}_{i:04d}" for i in range(1, count + 1)]

    @staticmethod
    def add_emotion_levels(
        utterances: List[str],
        mode: str = "none",
        fixed_level: int = 0,
        distribution_params: Optional[dict] = None,
    ) -> List[str]:
        """Add emotion level prefixes to utterances.

        Args:
            utterances: List of utterances
            mode: "none", "fixed", or "distribution"
            fixed_level: Level to use for "fixed" mode
            distribution_params: Dict with mean, std_dev, min_val, max_val for "distribution" mode

        Returns:
            List of utterances with emotion level prefixes
        """
        if mode == "none":
            return utterances

        if mode == "fixed":
            if fixed_level == 0:
                return utterances
            return [f"{fixed_level}: {utt}" for utt in utterances]

        if mode == "distribution":
            if not distribution_params:
                raise ValueError("distribution_params required for distribution mode")

            levels = TextImporter.truncated_normal(
                size=len(utterances),
                mean=distribution_params["mean"],
                std_dev=distribution_params["std_dev"],
                min_val=distribution_params["min_val"],
                max_val=distribution_params["max_val"],
            )

            # Apply levels to utterances
            formatted = []
            for utt, level in zip(utterances, levels):
                if level == 0:
                    formatted.append(utt)
                else:
                    formatted.append(f"{level}: {utt}")

            return formatted

        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def write_festival_script(
        output_path: Path, labels: List[str], utterances: List[str]
    ) -> None:
        """Write Festival format script file.

        Args:
            output_path: Output file path
            labels: List of utterance labels
            utterances: List of utterance texts
        """
        if len(labels) != len(utterances):
            raise ValueError("labels and utterances must have same length")

        with open(output_path, "w", encoding="utf-8") as f:
            for label, text in zip(labels, utterances):
                # Escape quotes in text
                text = text.replace('"', '\\"')
                f.write(f'( {label} "{text}" )\n')

    @staticmethod
    def import_text_file(
        input_path: Path,
        output_path: Path,
        max_length: int = 80,
        split_mode: str = "sentences",
        emotion_mode: str = "none",
        fixed_level: int = 0,
        distribution_params: Optional[dict] = None,
        label_prefix: str = "utt",
    ) -> int:
        """Import a text file and convert to Festival script format.

        Args:
            input_path: Path to input text file
            output_path: Path for output script file
            max_length: Maximum utterance length
            split_mode: "sentences", "lines", or "paragraphs"
            emotion_mode: "none", "fixed", or "distribution"
            fixed_level: Level for fixed mode
            distribution_params: Parameters for distribution mode
            label_prefix: Prefix for utterance labels

        Returns:
            Number of utterances generated
        """
        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        utterances = TextImporter.split_text(text, max_length, split_mode)
        if not utterances:
            raise ValueError("No utterances generated from input text")

        labels = TextImporter.generate_labels(len(utterances), label_prefix)
        utterances = TextImporter.add_emotion_levels(
            utterances, emotion_mode, fixed_level, distribution_params
        )
        TextImporter.write_festival_script(output_path, labels, utterances)

        return len(utterances)
