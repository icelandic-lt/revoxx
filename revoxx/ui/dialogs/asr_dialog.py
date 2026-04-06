"""Dialog for ASR verification of recorded utterances."""

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import TYPE_CHECKING

from .dialog_utils import setup_dialog_window, create_tooltip
from .progress_dialog import ProgressDialog
from ...controllers.session_controller import REFERENCE_SILENCE_LABEL

if TYPE_CHECKING:
    from ...app import Revoxx

ENV_FILE = Path.home() / ".revoxx" / ".env"


def _load_api_key() -> str:
    """Load ASR API key from .env file."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("ASR_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return os.environ.get("ASR_API_KEY", "")


def _save_api_key(key: str) -> None:
    """Save ASR API key to .env file."""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    found = False
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith("ASR_API_KEY="):
                lines.append(f"ASR_API_KEY={key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"ASR_API_KEY={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


class ASRVerificationDialog:
    """Dialog for configuring and running ASR verification."""

    DIALOG_WIDTH = 600
    DIALOG_HEIGHT = 400

    def __init__(self, parent: tk.Tk, app: "Revoxx"):
        self.parent = parent
        self.app = app
        self.settings = app.settings_manager
        self.dialog = tk.Toplevel(parent)
        self._running = False
        self._connection_queue: queue.Queue = queue.Queue()

        self._create_widgets()

        setup_dialog_window(
            self.dialog,
            self.parent,
            title="ASR Verification",
            width=self.DIALOG_WIDTH,
            height=self.DIALOG_HEIGHT,
            center_on_parent=True,
        )

        self.dialog.bind("<Escape>", lambda e: self._on_close())
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-test connection in background if URL is configured
        if self.url_var.get().strip():
            self.dialog.after(100, self._auto_test_connection)

    def _create_widgets(self):
        main = ttk.Frame(self.dialog, padding="10")
        main.pack(fill=tk.BOTH, expand=True)

        # Endpoint settings
        endpoint_frame = ttk.LabelFrame(main, text="ASR Endpoint", padding="8")
        endpoint_frame.pack(fill=tk.X, pady=(0, 8))
        endpoint_frame.columnconfigure(1, weight=1)

        ttk.Label(endpoint_frame, text="Base URL:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2
        )
        self.url_var = tk.StringVar(
            value=self.settings.get_setting("asr_base_url", "") or ""
        )
        url_frame = ttk.Frame(endpoint_frame)
        url_frame.grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Entry(url_frame, textvariable=self.url_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Label(
            url_frame,
            text="/v1/audio/transcriptions",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(endpoint_frame, text="API Key:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2
        )
        self.key_var = tk.StringVar(value=_load_api_key())
        ttk.Entry(endpoint_frame, textvariable=self.key_var, show="*").grid(
            row=1, column=1, sticky=tk.EW, pady=2
        )

        ttk.Label(endpoint_frame, text="Language:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5), pady=2
        )
        self.lang_var = tk.StringVar(
            value=self.settings.get_setting("asr_language", "") or ""
        )
        lang_frame = ttk.Frame(endpoint_frame)
        lang_frame.grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Entry(lang_frame, textvariable=self.lang_var, width=8).pack(side=tk.LEFT)
        ttk.Label(lang_frame, text="(ISO code, e.g. 'is', 'en')").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Test connection button and status
        test_frame = ttk.Frame(endpoint_frame)
        test_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        ttk.Button(
            test_frame, text="Test Connection", command=self._test_connection
        ).pack(side=tk.LEFT)
        self.connection_status = ttk.Label(test_frame, text="")
        self.connection_status.pack(side=tk.LEFT, padx=(10, 0))

        # Verification settings
        verify_frame = ttk.LabelFrame(main, text="Verification", padding="8")
        verify_frame.pack(fill=tk.X, pady=(0, 8))
        verify_frame.columnconfigure(1, weight=1)

        ttk.Label(verify_frame, text="Similarity threshold:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2
        )
        thresh_frame = ttk.Frame(verify_frame)
        thresh_frame.grid(row=0, column=1, sticky=tk.W, pady=2)
        self.threshold_var = tk.StringVar(
            value=str(
                int(self.settings.get_setting("asr_similarity_threshold", 0.95) * 100)
            )
        )
        ttk.Entry(thresh_frame, textvariable=self.threshold_var, width=5).pack(
            side=tk.LEFT
        )
        ttk.Label(thresh_frame, text="%").pack(side=tk.LEFT)

        ttk.Label(verify_frame, text="Concurrent requests:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2
        )
        self.concurrent_var = tk.StringVar(
            value=str(self.settings.get_setting("asr_max_concurrent", 4))
        )
        ttk.Entry(verify_frame, textvariable=self.concurrent_var, width=5).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )

        self.force_var = tk.BooleanVar(value=False)
        force_cb = ttk.Checkbutton(
            verify_frame,
            text="Force",
            variable=self.force_var,
        )
        force_cb.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        create_tooltip(
            force_cb, "Re-transcribe all utterances, overwriting existing results"
        )

        # Status info
        self.status_label = ttk.Label(main, text=self._build_status_text())
        self.status_label.pack(fill=tk.X, pady=(0, 8))

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Close", command=self._on_close).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        self.run_btn = ttk.Button(
            btn_frame,
            text="Run Verification",
            command=self._run_verification,
            state="disabled",
        )
        self.run_btn.pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Clear Results", command=self._clear_results).pack(
            side=tk.LEFT
        )

    def _build_status_text(self) -> str:
        session = self.app.current_session
        if not session:
            return "No session loaded"

        labels = self.app.state.recording.labels
        takes = self.app.state.recording.takes
        rejected = {k for k, v in session.utterance_flags.items() if v == "rejected"}
        verified = session.asr_verification

        total = 0
        recorded = 0
        already_verified = 0
        to_verify = 0
        mismatches = 0

        for label in labels:
            if label == REFERENCE_SILENCE_LABEL:
                continue
            total += 1
            if label in rejected:
                continue
            if takes.get(label, 0) > 0:
                recorded += 1
                if label in verified:
                    already_verified += 1
                    if not verified[label].get("match", True):
                        mismatches += 1
                else:
                    to_verify += 1

        return (
            f"{total} utterances, {recorded} recorded, "
            f"{already_verified} verified ({mismatches} mismatches), "
            f"{to_verify} to verify"
        )

    def _test_connection(self):
        """Test the ASR endpoint with a GET to /v1/models."""
        url = self.url_var.get().strip()
        if not url:
            self.connection_status.config(text="Enter a URL first", foreground="red")
            return

        self.connection_status.config(text="Testing...", foreground="gray")
        self.dialog.update()

        import httpx as _httpx
        from ...dataset.asr_verifier import _normalize_base_url

        try:
            base = _normalize_base_url(url)
            _httpx.get(
                f"{base}/v1/models",
                timeout=_httpx.Timeout(5.0, connect=3.0),
            )
            self.connection_status.config(text="Connection OK", foreground="green")
            self.run_btn.configure(state="normal")
        except Exception as e:
            self.connection_status.config(
                text=f"Failed: {str(e)[:60]}", foreground="red"
            )

    def _auto_test_connection(self):
        """Test connection in background without blocking the dialog.

        Uses a thread to perform the HTTP request and a queue + polling
        to safely update the UI from the main thread.
        """
        url = self.url_var.get().strip()
        if not url:
            return

        self.connection_status.config(text="Testing...", foreground="gray")

        def worker():
            import httpx as _httpx
            from ...dataset.asr_verifier import _normalize_base_url

            try:
                base = _normalize_base_url(url)
                _httpx.get(
                    f"{base}/v1/models",
                    timeout=_httpx.Timeout(5.0, connect=3.0),
                )
                self._connection_queue.put(("ok", None))
            except Exception as e:
                self._connection_queue.put(("fail", str(e)[:60]))

        threading.Thread(target=worker, daemon=True).start()
        self.dialog.after(100, self._poll_connection_result)

    def _poll_connection_result(self):
        """Poll the connection test result queue from the main thread."""
        try:
            status, msg = self._connection_queue.get_nowait()
        except queue.Empty:
            # Still waiting - schedule next poll
            try:
                self.dialog.after(100, self._poll_connection_result)
            except tk.TclError:
                pass  # Dialog was destroyed
            return

        try:
            if status == "ok":
                self.connection_status.config(text="Connection OK", foreground="green")
                self.run_btn.configure(state="normal")
            else:
                self.connection_status.config(text=f"Failed: {msg}", foreground="red")
        except tk.TclError:
            pass  # Dialog was destroyed

    def _save_settings(self):
        url = self.url_var.get().strip()
        if url:
            self.settings.update_setting("asr_base_url", url)
        lang = self.lang_var.get().strip()
        self.settings.update_setting("asr_language", lang if lang else None)

        try:
            threshold = int(self.threshold_var.get()) / 100.0
            self.settings.update_setting("asr_similarity_threshold", threshold)
        except ValueError:
            pass
        try:
            concurrent = int(self.concurrent_var.get())
            self.settings.update_setting("asr_max_concurrent", max(1, concurrent))
        except ValueError:
            pass

        key = self.key_var.get().strip()
        if key:
            _save_api_key(key)

    def _run_verification(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning(
                "Missing URL", "Please enter the ASR endpoint URL.", parent=self.dialog
            )
            return

        session = self.app.current_session
        if not session:
            return

        self._save_settings()

        try:
            threshold = int(self.threshold_var.get()) / 100.0
        except ValueError:
            threshold = 0.95
        try:
            concurrent = max(1, int(self.concurrent_var.get()))
        except ValueError:
            concurrent = 4

        key = self.key_var.get().strip() or None
        lang = self.lang_var.get().strip() or None

        # Collect data for verification
        labels = self.app.state.recording.labels
        utterances = self.app.state.recording.utterances
        takes = {}
        for label in labels:
            if self.app.active_recordings:
                takes[label] = self.app.active_recordings.get_highest_take(label)
            else:
                takes[label] = self.app.state.recording.takes.get(label, 0)

        rejected = {k for k, v in session.utterance_flags.items() if v == "rejected"}

        self._running = True
        self.run_btn.configure(state="disabled")

        progress = ProgressDialog(self.dialog, "ASR Verification")

        def on_progress(completed, total, label):
            progress.update(completed, f"{completed}/{total}: {label}", maximum=total)

        try:
            from ...dataset.asr_verifier import verify_utterances

            results = verify_utterances(
                session_dir=session.session_dir,
                labels=labels,
                utterances=utterances,
                takes=takes,
                base_url=url,
                api_key=key,
                language=lang,
                similarity_threshold=threshold,
                max_concurrent=concurrent,
                progress_callback=on_progress,
                rejected_labels=rejected,
                existing_results=(
                    {} if self.force_var.get() else session.asr_verification
                ),
            )

            progress.close()

            if self.force_var.get():
                # Force mode: replace all results (clears stale entries)
                session.asr_verification = results
            else:
                session.asr_verification.update(results)
            session.save()

            # Refresh sort data in active_recordings
            if self.app.active_recordings:
                self.app.active_recordings.set_asr_verification(
                    session.asr_verification
                )

            mismatches = sum(1 for r in results.values() if not r.get("match", True))
            errors = sum(1 for r in results.values() if "error" in r)
            msg = f"Verified {len(results)} utterances: {mismatches} mismatches"
            if errors:
                msg += f", {errors} errors"

            self.status_label.config(text=self._build_status_text())
            messagebox.showinfo("Verification Complete", msg, parent=self.dialog)

        except Exception as e:
            progress.close()
            messagebox.showerror("Verification Error", str(e), parent=self.dialog)
        finally:
            self._running = False
            self.run_btn.configure(state="normal")

    def _clear_results(self):
        session = self.app.current_session
        if not session:
            return
        if not session.asr_verification:
            return

        count = len(session.asr_verification)
        if messagebox.askyesno(
            "Clear Results",
            f"Clear {count} ASR verification results?",
            parent=self.dialog,
        ):
            session.asr_verification.clear()
            session.save()
            if self.app.active_recordings:
                self.app.active_recordings.set_asr_verification({})
            self.status_label.config(text=self._build_status_text())

    def _on_close(self):
        if self._running:
            return
        self._save_settings()
        self.dialog.grab_release()
        self.dialog.destroy()
        self.parent.focus_force()

    def show(self):
        self.dialog.wait_window()
