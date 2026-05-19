"""Mock streaming TTS.

Accepts an async stream of `Token` and emits an async stream of
`AudioFrame`. We don't synthesize anything real — each token becomes
a short burst of silent PCM, sized realistically.

Critically: this implementation back-pressures correctly. If the
consumer (network / speaker) is slow, we don't run ahead. That's the
contract every real TTS adapter must honour.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from ongovoice.core import AudioFrame, Token


class MockTTS:
    """Emits one ~80ms silent audio frame per non-empty token."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: float = 80.0,
        synth_latency_ms: float = 25.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._synth_latency_ms = synth_latency_ms
        self._samples_per_frame = int(sample_rate * frame_ms / 1000)

    async def stream(self, tokens: AsyncIterator[Token]) -> AsyncIterator[AudioFrame]:
        async for token in tokens:
            if not token.text:
                continue
            # Realistic per-token synth latency.
            await asyncio.sleep(self._synth_latency_ms / 1000.0)
            pcm = np.zeros(self._samples_per_frame, dtype=np.int16)
            yield AudioFrame(sample_rate=self._sample_rate, pcm=pcm)
