"""Wake-word detection.

Production uses openWakeWord or Picovoice. For the skeleton we provide
a `KeywordWakeWord` that scans transcribed text for the wake phrase
("Hey Ongo" by default). It's not a real wake-word detector — it sits
*after* ASR — but it's the right Protocol shape for tests, and lets the
rest of the pipeline be exercised end-to-end without weights.
"""

from __future__ import annotations

import re

from ongovoice.core import AudioFrame


class KeywordWakeWord:
    """Text-side wake-word stub for tests + dev.

    The real on-device detector runs *before* ASR — this is here so
    the pipeline manager can be tested with a transcript-level signal.
    """

    def __init__(self, *, wake_phrase: str = "Hey Ongo") -> None:
        self._wake_phrase = wake_phrase
        self._pattern = re.compile(rf"\b{re.escape(wake_phrase)}\b", re.IGNORECASE)
        self._armed = False

    @property
    def wake_phrase(self) -> str:
        return self._wake_phrase

    def detect(self, frame: AudioFrame) -> bool:  # noqa: ARG002
        """Audio-side stub. Always False — the real impl runs ONNX here."""
        return False

    def matches_text(self, text: str) -> bool:
        """Used by the text-driven pipeline tests."""
        return bool(self._pattern.search(text))

    def reset(self) -> None:
        self._armed = False
