"""Protocols.

Every pluggable component declares its interface here. New ASR? New
TTS vendor? New LLM provider? Implement the Protocol, register it,
done.

We use `runtime_checkable` so `isinstance(thing, LLMProvider)` works
at runtime — useful in the API to validate config without booting
the whole pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ongovoice.core.types import (
    AudioFrame,
    Intent,
    LLMRequest,
    PartialTranscript,
    Provider,
    RoutingDecision,
    Token,
    Transcript,
)


@runtime_checkable
class VADProvider(Protocol):
    """Voice-activity detection. Stateful (tracks silence runs)."""

    def is_speech(self, frame: AudioFrame) -> bool: ...

    def reset(self) -> None: ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """Looks for the wake phrase in a rolling buffer of frames."""

    @property
    def wake_phrase(self) -> str: ...

    def detect(self, frame: AudioFrame) -> bool: ...

    def reset(self) -> None: ...


@runtime_checkable
class ASRProvider(Protocol):
    """Streaming speech-to-text.

    Yields partials as soon as the underlying recognizer emits them.
    The final partial has `is_final=True` and is followed by `aclose()`.
    """

    @property
    def is_edge(self) -> bool: ...

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[PartialTranscript]:
        ...


@runtime_checkable
class Router(Protocol):
    """Decides where each turn goes — edge or cloud."""

    async def route(self, transcript: Transcript, intent: Intent) -> RoutingDecision: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Streaming LLM. One method. Returns tokens as they arrive."""

    @property
    def provider_id(self) -> Provider: ...

    @property
    def model_name(self) -> str: ...

    def stream(self, req: LLMRequest) -> AsyncIterator[Token]: ...

    def cost_estimate(self, tokens_in: int, tokens_out: int) -> float:
        """Return USD cost. Pulled from a per-provider rate table."""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """Streaming text-to-speech.

    Consumes tokens, emits audio frames. Tokens may arrive faster than
    we can synthesize; the implementation must back-pressure correctly.
    """

    def stream(self, tokens: AsyncIterator[Token]) -> AsyncIterator[AudioFrame]: ...
