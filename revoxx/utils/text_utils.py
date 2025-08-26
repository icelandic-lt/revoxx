"""Utility functions for text processing."""

import re
from typing import Tuple, Optional


def extract_emotion_level(text: str) -> Tuple[Optional[int], str]:
    """Extract emotion level from utterance text.

    Looks for pattern "N: " at the beginning of text where N is a number.

    Args:
        text: The utterance text possibly containing emotion label

    Returns:
        Tuple of (emotion_level, clean_text)
        - emotion_level: The extracted level as int, or None if not found
        - clean_text: The text with emotion label removed
    """
    if not text:
        return None, text

    # Pattern matches digit(s) followed by colon and space at start of string
    pattern = r"^(\d+):\s*"
    match = re.match(pattern, text)

    if match:
        emotion_level = int(match.group(1))
        clean_text = text[match.end() :]  # Text after the pattern
        return emotion_level, clean_text

    return None, text


def get_max_emotion_level(utterances: list) -> int:
    """Determine the maximum emotion level from all utterances.

    Args:
        utterances: List of utterance strings

    Returns:
        Maximum emotion level found, or 5 as default
    """
    max_level = 0

    for utterance in utterances:
        level, _ = extract_emotion_level(utterance)
        if level and level > max_level:
            max_level = level

    # Return found max or default to 5
    return max_level if max_level > 0 else 5
