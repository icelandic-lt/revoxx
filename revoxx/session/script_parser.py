"""Parser for Festival format script files."""

from pathlib import Path
from typing import Dict, List, Tuple


class FestivalScriptParser:
    """Parser for Festival format script files used in Revoxx."""

    @staticmethod
    def parse_script(script_path: Path) -> Dict[str, str]:
        """Parse a Festival format script file.

        Args:
            script_path: Path to script.txt file

        Returns:
            Dictionary mapping utterance_id to text
        """
        script_data = {}
        if not script_path.exists():
            return script_data

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        # Parse Festival format: (utterance_id "text")
                        parts = line[1:-1].strip().split(None, 1)
                        if len(parts) == 2:
                            utterance_id = parts[0]
                            text = parts[1].strip('"')
                            script_data[utterance_id] = text
        except (IOError, UnicodeDecodeError):
            pass

        return script_data

    @staticmethod
    def extract_intensity_and_text(text: str) -> Tuple[str, str]:
        """Extract intensity level and clean text from utterance.

        The text format can be:
        - "N: actual text" where N is intensity level 1-5
        - Just "actual text" for no intensity (defaults to "0")

        Args:
            text: Raw text that may contain intensity prefix

        Returns:
            Tuple of (intensity, clean_text)
        """
        # Check if text starts with intensity pattern "N: "
        if text and len(text) > 2 and text[0].isdigit() and text[1] == ":":
            intensity = text[0]
            clean_text = text[2:].strip()  # Remove "N: " prefix
            return intensity, clean_text
        return "0", text

    @staticmethod
    def get_utterance_list(script_path: Path) -> List[Tuple[str, str]]:
        """Get ordered list of utterances from script.

        Args:
            script_path: Path to script.txt file

        Returns:
            List of (utterance_id, text) tuples in order
        """
        utterances = []
        if not script_path.exists():
            return utterances

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        parts = line[1:-1].strip().split(None, 1)
                        if len(parts) == 2:
                            utterance_id = parts[0]
                            text = parts[1].strip('"')
                            utterances.append((utterance_id, text))
        except (IOError, UnicodeDecodeError):
            pass

        return utterances
