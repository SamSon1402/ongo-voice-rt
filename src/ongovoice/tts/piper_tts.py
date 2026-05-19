"""Piper TTS adapter (scaffolded).

Piper is the right edge TTS for Ongo — it runs in ~50ms on a Raspberry
Pi 5 for short utterances, voices are tiny (20-50MB), and the output
sounds genuinely good. The streaming shape matches the Protocol almost
1:1.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from ongovoice.core import AudioFrame, Token


class PiperTTS:
    """Piper TTS adapter. Scaffolded."""

    def __init__(
        self,
        *,
        voice_path: Path,
        sample_rate: int = 22_050,
    ) -> None:
        self._voice_path = Path(voice_path)
        self._sample_rate = sample_rate
        # self._voice = PiperVoice.load(str(voice_path))
        raise NotImplementedError(
            "PiperTTS scaffolded — wire `piper_tts.PiperVoice.synthesize_stream_raw` "
            "into the loop once we settle on a default voice. en_GB-alan-low is "
            "the leading candidate for Ongo's persona."
        )

    async def stream(self, tokens: AsyncIterator[Token]) -> AsyncIterator[AudioFrame]:
        raise NotImplementedError
        if False:
            yield  # type: ignore[unreachable]
