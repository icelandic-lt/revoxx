"""Automatic ASR verification after recording."""

import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import tkinter as tk

from ..constants import MsgType

if TYPE_CHECKING:
    from ..app import Revoxx


class ASRAutoController:
    """Runs ASR verification in the background after each recording.

    The transcription runs in a worker thread so it doesn't block the
    recording UI. Results are delivered back to the main thread via a
    queue that is polled periodically.
    """

    POLL_INTERVAL_MS = 150

    def __init__(self, app: "Revoxx"):
        self.app = app
        self._result_queue: queue.Queue = queue.Queue()
        self._polling = False
        self._no_endpoint_warned = False

    def is_enabled(self) -> bool:
        """Check if auto-verification is enabled and an endpoint is configured."""
        settings = self.app.settings_manager.settings
        if not getattr(settings, "asr_auto_verify", False):
            return False
        if not getattr(settings, "asr_base_url", None):
            # User enabled auto-verify but no endpoint - warn once
            if not self._no_endpoint_warned:
                self._no_endpoint_warned = True
                self.app.display_controller.set_status(
                    "ASR auto-verify enabled but no endpoint configured - "
                    "open File > ASR Verification to set URL",
                    MsgType.TEMPORARY,
                )
            return False
        # Reset warning flag once endpoint is available again
        self._no_endpoint_warned = False
        return True

    def has_endpoint(self) -> bool:
        """Check if an ASR endpoint is configured (for menu state)."""
        return bool(getattr(self.app.settings_manager.settings, "asr_base_url", None))

    def trigger(self, label: str, take: int, audio_path: Path) -> None:
        """Trigger background ASR verification for a freshly recorded take.

        Args:
            label: Utterance label
            take: Take number
            audio_path: Path to the audio file
        """
        if not self.is_enabled():
            return

        if not self.app.current_session:
            return

        # Find the expected text for this label
        expected_text = self._get_expected_text(label)
        if expected_text is None:
            return

        settings = self.app.settings_manager.settings
        base_url = settings.asr_base_url
        api_key = self._load_api_key()
        language = settings.asr_language
        threshold = settings.asr_similarity_threshold

        self.app.display_controller.set_status(
            f"ASR: transcribing {label}...", MsgType.TEMPORARY
        )

        thread = threading.Thread(
            target=self._worker,
            args=(
                label,
                take,
                audio_path,
                expected_text,
                base_url,
                api_key,
                language,
                threshold,
            ),
            daemon=True,
        )
        thread.start()

        if not self._polling:
            self._polling = True
            self.app.window.window.after(self.POLL_INTERVAL_MS, self._poll_results)

    def _worker(
        self,
        label: str,
        take: int,
        audio_path: Path,
        expected_text: str,
        base_url: str,
        api_key: Optional[str],
        language: Optional[str],
        threshold: float,
    ) -> None:
        """Worker thread: run ASR and push result to queue."""
        from ..dataset.asr_verifier import verify_single

        result = verify_single(
            audio_path=audio_path,
            expected_text=expected_text,
            base_url=base_url,
            api_key=api_key,
            language=language,
            similarity_threshold=threshold,
        )
        self._result_queue.put((label, take, result))

    def _poll_results(self) -> None:
        """Poll the result queue from the main thread and apply results."""
        drained = False
        try:
            while True:
                try:
                    label, take, result = self._result_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                self._apply_result(label, take, result)
        except tk.TclError:
            self._polling = False
            return

        # Keep polling if worker thread might still be running
        # Stop polling after queue is empty and we drained at least once
        if drained and self._result_queue.empty():
            self._polling = False
            return

        try:
            self.app.window.window.after(self.POLL_INTERVAL_MS, self._poll_results)
        except tk.TclError:
            self._polling = False

    def _apply_result(self, label: str, take: int, result: dict) -> None:
        """Apply a verification result to the session."""
        session = self.app.current_session
        if not session:
            return

        # Verify the take hasn't been replaced by a newer recording
        if self.app.active_recordings:
            current_take = self.app.active_recordings.get_highest_take(label)
            if current_take != take:
                # Newer recording exists, skip this result
                return

        session.asr_verification[label] = result
        session.save()

        if self.app.active_recordings:
            self.app.active_recordings.set_asr_verification(session.asr_verification)

        # Status feedback
        if "error" in result:
            msg = f"ASR error for {label}: {result['error'][:60]}"
        elif result["match"]:
            pct = int(result["similarity"] * 100)
            msg = f"ASR: match ({pct}%) {label}"
        else:
            pct = int(result["similarity"] * 100)
            msg = f"ASR: mismatch ({pct}%) {label}"

        self.app.display_controller.set_status(msg, MsgType.TEMPORARY)

        # Refresh flag indicator if this is the current utterance
        current_label = self.app.state.recording.current_label
        if current_label == label:
            self.app.display_controller.update_flag_indicator(label)

    def _get_expected_text(self, label: str) -> Optional[str]:
        """Get the expected script text for a label."""
        labels = self.app.state.recording.labels
        utterances = self.app.state.recording.utterances
        try:
            idx = labels.index(label)
            return utterances[idx]
        except (ValueError, IndexError):
            return None

    def _load_api_key(self) -> Optional[str]:
        """Load API key from .env file or environment."""
        import os

        env_file = Path.home() / ".revoxx" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ASR_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        return os.environ.get("ASR_API_KEY") or None
