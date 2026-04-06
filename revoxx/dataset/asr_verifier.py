"""ASR verification for recorded utterances.

Transcribes recordings via an OpenAI-compatible ASR endpoint and compares
the result against the expected script text. Utterances where the ASR
transcription diverges beyond a configurable threshold are flagged.
"""

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import httpx


def _normalize_base_url(url: str) -> str:
    """Normalize a base URL for the ASR client.

    Adds http:// scheme if missing and strips /v1 suffix.
    """
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if url.endswith("/v1"):
        url = url[:-3]
    return url


class ASRClient:
    """Client for OpenAI-compatible speech-to-text endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        clean_url = _normalize_base_url(base_url)

        self.client = httpx.Client(
            base_url=clean_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to audio file
            language: Optional ISO language code (e.g. "is" for Icelandic)

        Returns:
            Transcribed text
        """
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/flac")}
            data = {"model": "whisper-1", "response_format": "text"}
            if language:
                data["language"] = language

            response = self.client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Extract transcription text from response.

        Handles both plain text and JSON responses from different servers.
        """
        text = response.text.strip()
        if text.startswith("{"):
            try:
                import json

                data = json.loads(text)
                return data.get("text", text).strip()
            except (json.JSONDecodeError, AttributeError):
                pass
        return text

    def close(self):
        self.client.close()


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Lowercases, removes punctuation, collapses whitespace,
    and applies unicode normalization.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def character_similarity(expected: str, actual: str) -> float:
    """Calculate character-level similarity between two strings.

    Uses Levenshtein distance normalized by the longer string length.

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    expected = normalize_text(expected)
    actual = normalize_text(actual)

    if expected == actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    # Levenshtein distance via dynamic programming
    m, n = len(expected), len(actual)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if expected[i - 1] == actual[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    distance = dp[n]
    max_len = max(m, n)
    return 1.0 - (distance / max_len)


def verify_utterances(
    session_dir: Path,
    labels: List[str],
    utterances: List[str],
    takes: Dict[str, int],
    base_url: str,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    similarity_threshold: float = 0.95,
    max_concurrent: int = 4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    rejected_labels: Optional[set] = None,
    existing_results: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Verify recordings against expected text using ASR.

    Args:
        session_dir: Path to session directory
        labels: List of utterance labels
        utterances: List of expected utterance texts
        takes: Dict of label -> highest take number
        base_url: ASR endpoint base URL
        api_key: Optional API key
        language: Optional ISO language code
        similarity_threshold: Minimum similarity to count as match (0.0-1.0)
        max_concurrent: Number of concurrent ASR requests
        progress_callback: Optional callback(completed, total, label)
        rejected_labels: Labels to skip
        existing_results: Already verified results to skip

    Returns:
        Dict of label -> {transcription, expected, match, similarity}
    """
    rejected = rejected_labels or set()
    verified = existing_results or {}

    # Collect utterances to verify
    to_verify: List[Tuple[str, str, Path]] = []
    recordings_dir = session_dir / "recordings"

    from ..controllers.session_controller import REFERENCE_SILENCE_LABEL

    for label, text in zip(labels, utterances):
        if label == REFERENCE_SILENCE_LABEL:
            continue
        if label in rejected:
            continue
        if label in verified:
            continue
        take_num = takes.get(label, 0)
        if take_num == 0:
            continue
        audio_path = recordings_dir / label / f"take_{take_num:03d}.flac"
        if audio_path.exists():
            to_verify.append((label, text, audio_path))

    if not to_verify:
        return {}

    total = len(to_verify)
    results = {}
    client = ASRClient(base_url, api_key)

    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_label = {
                executor.submit(client.transcribe, audio_path, language): (
                    label,
                    expected_text,
                )
                for label, expected_text, audio_path in to_verify
            }

            completed = 0
            for future in as_completed(future_to_label):
                label, expected_text = future_to_label[future]
                completed += 1

                try:
                    transcription = future.result()
                    similarity = character_similarity(expected_text, transcription)
                    results[label] = {
                        "transcription": transcription,
                        "expected": expected_text,
                        "match": similarity >= similarity_threshold,
                        "similarity": round(similarity, 4),
                    }
                except Exception as e:
                    results[label] = {
                        "transcription": "",
                        "expected": expected_text,
                        "match": False,
                        "similarity": 0.0,
                        "error": str(e),
                    }

                if progress_callback:
                    progress_callback(completed, total, label)
    finally:
        client.close()

    return results
