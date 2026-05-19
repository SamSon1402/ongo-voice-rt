"""Mock ASR for tests + the text-driven CLI.

Pretends to transcribe by ignoring audio entirely and emitting a
pre-configured transcript with a realistic streaming pattern (partial,
partial, final).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ongovoice.core import AudioFrame, PartialTranscript


class MockASR:
    """Configurable mock ASR.

    Used by the CLI to send a pre-written transcript through the
    pipeline, and by tests to exercise the full async flow without
    needing weights.
    """

    def __init__(
        self,
        *,
        final_text: str = "Hey Ongo, what's the time?",
        final_confidence: float = 0.92,
        time_to_final_ms: float = 180.0,
    ) -> None:
        self._final_text = final_text
        self._final_confidence = final_confidence
        self._time_to_final_ms = time_to_final_ms

    @property
    def is_edge(self) -> bool:
        return True

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[PartialTranscript]:
        # We still consume the audio iterator so back-pressure is real.
        # But we ignore the content.
        async def _drain() -> None:
            async for _ in audio:
                pass

        drain_task = asyncio.create_task(_drain())
        try:
            # Two intermediate partials at ~⅓ and ⅔ of the way through.
            words = self._final_text.split()
            step_ms = self._time_to_final_ms / 3
            await asyncio.sleep(step_ms / 1000.0)
            yield PartialTranscript(
                text=" ".join(words[: max(1, len(words) // 3)]),
                is_final=False,
                confidence=0.45,
            )
            await asyncio.sleep(step_ms / 1000.0)
            yield PartialTranscript(
                text=" ".join(words[: max(2, 2 * len(words) // 3)]),
                is_final=False,
                confidence=0.70,
            )
            await asyncio.sleep(step_ms / 1000.0)
            yield PartialTranscript(
                text=self._final_text,
                is_final=True,
                confidence=self._final_confidence,
            )
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
