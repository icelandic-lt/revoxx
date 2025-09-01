"""Queue manager for audio process communication.

This module provides a central interface for sending commands to
audio processing queues.
"""

import multiprocessing as mp
import queue
from typing import Optional, Any, Dict


class AudioQueueManager:
    """Manages communication with audio processing queues.

    This class provides high-level methods for sending commands to the
    recording and playback processes, as well as managing the audio
    visualization queue.

    Can be used in two modes:
    1. Main process mode: Creates queues (no parameters)
    2. Child process mode: Uses existing queues (pass queues as parameters)

    """

    def __init__(self, record_queue=None, playback_queue=None, audio_queue=None):
        """Initialize the audio queue manager.

        Args:
            record_queue: Optional existing record queue (for child processes)
            playback_queue: Optional existing playback queue (for child processes)
            audio_queue: Optional existing audio queue (for child processes)
        """
        if record_queue is None and playback_queue is None and audio_queue is None:
            # Main process mode - create queues
            self._record_queue = mp.Queue(maxsize=10)
            self._playback_queue = mp.Queue(maxsize=10)
            self._audio_queue = mp.Queue(maxsize=100)
        else:
            # Child process mode - use existing queues
            self._record_queue = record_queue
            self._playback_queue = playback_queue
            self._audio_queue = audio_queue

    # ========== Playback Control Methods ==========

    def start_playback(self, buffer_metadata: Dict[str, Any], sample_rate: int) -> bool:
        """Start audio playback with the given buffer.

        Args:
            buffer_metadata: Metadata for the shared memory buffer containing:
                - name: Name of the shared memory block
                - shape: Shape of the numpy array
                - dtype: Data type of the array
            sample_rate: Sample rate of the audio in Hz

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._playback_queue.put(
                {
                    "action": "play",
                    "buffer_metadata": buffer_metadata,
                    "sample_rate": sample_rate,
                },
                block=False,
            )
            return True
        except queue.Full:
            return False

    def stop_playback(self) -> None:
        """Stop audio playback immediately."""
        # Stop command should block to ensure it gets through
        self._playback_queue.put({"action": "stop"})

    def set_output_device(self, device_name: Optional[str]) -> bool:
        """Set the output device for playback.

        Args:
            device_name: Device name or None for system default

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._playback_queue.put(
                {"action": "set_output_device", "device_name": device_name}, block=False
            )
            return True
        except queue.Full:
            return False

    def set_output_channel_mapping(self, mapping: Optional[list]) -> bool:
        """Set output channel mapping for playback.

        Args:
            mapping: Channel mapping list or None for default

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._playback_queue.put(
                {"action": "set_output_channel_mapping", "mapping": mapping},
                block=False,
            )
            return True
        except queue.Full:
            return False

    def quit_playback_process(self) -> bool:
        """Send quit command to playback process.

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._playback_queue.put({"action": "quit"}, block=False)
            return True
        except queue.Full:
            return False

    def refresh_playback_devices(self) -> bool:
        """Send refresh devices command to playback process.

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._playback_queue.put({"action": "refresh_devices"}, block=False)
            return True
        except queue.Full:
            return False

    # ========== Recording Control Methods ==========

    def start_recording(self) -> bool:
        """Start audio recording.

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._record_queue.put({"action": "start"}, block=False)
            return True
        except queue.Full:
            return False

    def stop_recording(self) -> None:
        """Stop audio recording."""
        # Stop command should block to ensure it gets through
        self._record_queue.put({"action": "stop"})

    def set_input_device(self, device_name: Optional[str]) -> bool:
        """Set the input device for recording.

        Args:
            device_name: Device name or None for system default

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._record_queue.put(
                {"action": "set_input_device", "device_name": device_name}, block=False
            )
            return True
        except queue.Full:
            return False

    def set_input_channel_mapping(self, mapping: Optional[list]) -> bool:
        """Set input channel mapping for recording.

        Args:
            mapping: Channel mapping list or None for default

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._record_queue.put(
                {"action": "set_input_channel_mapping", "mapping": mapping}, block=False
            )
            return True
        except queue.Full:
            return False

    def quit_record_process(self) -> bool:
        """Send quit command to record process.

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._record_queue.put({"action": "quit"}, block=False)
            return True
        except queue.Full:
            return False

    def refresh_record_devices(self) -> bool:
        """Send refresh devices command to record process.

        Returns:
            True if command was queued, False if queue was full
        """
        try:
            self._record_queue.put({"action": "refresh_devices"}, block=False)
            return True
        except queue.Full:
            return False

    # ========== Audio Visualization Queue Methods ==========

    def get_audio_data(self, timeout: float = 0.1) -> Any:
        """Get audio data from the visualization queue.

        Args:
            timeout: Timeout in seconds for getting data

        Returns:
            Audio data from the queue

        Raises:
            queue.Empty: If no data is available within the timeout
        """
        return self._audio_queue.get(timeout=timeout)

    @property
    def audio_queue(self) -> mp.Queue:
        """Get direct access to audio queue for special cases.

        Returns:
            The audio visualization queue
        """
        return self._audio_queue

    @property
    def record_queue(self) -> mp.Queue:
        """Get record queue for process initialization.

        Returns:
            The record command queue
        """
        return self._record_queue

    @property
    def playback_queue(self) -> mp.Queue:
        """Get playback queue for process initialization.

        Returns:
            The playback command queue
        """
        return self._playback_queue

    # ========== Processing-side Methods ==========

    def get_record_command(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get next command from record queue (used by record process).

        Args:
            timeout: Timeout in seconds

        Returns:
            Command dictionary or None if timeout

        Raises:
            TypeError: If received message is not a dictionary
        """
        try:
            command = self._record_queue.get(timeout=timeout)
            if not isinstance(command, dict):
                raise TypeError(
                    f"Expected dict command, got {type(command).__name__}: {command}"
                )
            return command
        except queue.Empty:
            return None

    def get_playback_command(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get next command from playback queue (used by playback process).

        Args:
            timeout: Timeout in seconds

        Returns:
            Command dictionary or None if timeout

        Raises:
            TypeError: If received message is not a dictionary
        """
        try:
            command = self._playback_queue.get(timeout=timeout)
            if not isinstance(command, dict):
                raise TypeError(
                    f"Expected dict command, got {type(command).__name__}: {command}"
                )
            return command
        except queue.Empty:
            return None

    def put_audio_data(self, data: Any) -> bool:
        """Put audio data into visualization queue.

        Args:
            data: Audio data to send

        Returns:
            True if sent, False if queue full
        """
        try:
            self._audio_queue.put_nowait(data)
            return True
        except queue.Full:
            return False
