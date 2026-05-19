"""Conversation manager.

Owns one user's voice session. Wires together:

    audio_source ─► ring_buffer ─► ASR.stream() ─► transcript
                                                     │
                                              IntentClassifier
                                                     │
                                              RouterPolicy.route()
                                                     │
                                  ┌──────────────────┴────────────────┐
                                edge LLM                        cloud LLM
                                  └──────────────────┬────────────────┘
                                                  TTS.stream()
                                                     │
                                                  audio_sink

The manager measures TTFT end-to-end and surfaces it via TurnMetrics.

Barge-in: if the user starts speaking while we're streaming a response,
`interrupt()` cancels the LLM + TTS tasks. We catch `CancelledError`
to clean up, record `barged_in=True`, and the loop reopens for the
next utterance.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from ongovoice.core import (
    AudioFrame,
    Intent,
    LLMProvider,
    LLMRequest,
    Provider,
    RoutingDecision,
    RoutingTarget,
    Token,
    Transcript,
    Turn,
    TurnMetrics,
)
from ongovoice.core.protocols import ASRProvider, TTSProvider
from ongovoice.router.classifier import IntentClassifier
from ongovoice.router.policy import RouterPolicy

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class ManagerConfig:
    system_prompt: str = (
        "You are Ongo, a friendly desk-lamp companion robot. Keep replies "
        "short — one or two sentences — because you're speaking, not typing."
    )
    max_response_tokens: int = 160


@dataclass(slots=True)
class _TurnState:
    turn_id: str
    started_at: float
    asr_done_at: float = 0.0
    routing_done_at: float = 0.0
    llm_first_token_at: float = 0.0
    llm_done_at: float = 0.0
    tts_first_audio_at: float = 0.0
    barged_in: bool = False
    assistant_text: list[str] = field(default_factory=list)
    tokens_out: int = 0


class ConversationManager:
    """One per active user session."""

    def __init__(
        self,
        *,
        asr: ASRProvider,
        edge_llm: LLMProvider,
        cloud_llm: LLMProvider,
        tts: TTSProvider,
        router: RouterPolicy,
        classifier: IntentClassifier,
        config: ManagerConfig | None = None,
    ) -> None:
        self._asr = asr
        self._edge_llm = edge_llm
        self._cloud_llm = cloud_llm
        self._tts = tts
        self._router = router
        self._classifier = classifier
        self._config = config or ManagerConfig()

        # Active task handles, owned per-turn. Used by interrupt().
        self._active_tasks: set[asyncio.Task[object]] = set()

    # ── public API ──────────────────────────────────────────────────

    async def handle_turn(
        self,
        audio: AsyncIterator[AudioFrame],
        audio_sink: asyncio.Queue[AudioFrame] | None = None,
    ) -> Turn:
        """Drive one turn end-to-end. Returns the full Turn record."""
        state = _TurnState(turn_id=_new_turn_id(), started_at=time.perf_counter())
        started_at_dt = datetime.now(UTC)
        log.info("turn.start", turn_id=state.turn_id)

        # 1. ASR — collect partials, keep only the final.
        transcript = await self._run_asr(audio, state)

        # 2. Intent + routing.
        intent, conf = self._classifier.classify(transcript)
        # Boost the transcript's confidence with the classifier's view so
        # the router has a single number to gate on.
        gating_transcript = transcript.model_copy(
            update={"confidence": min(1.0, conf)}
        )
        decision = await self._router.route(gating_transcript, intent)
        state.routing_done_at = time.perf_counter()
        log.info(
            "turn.routed",
            turn_id=state.turn_id,
            target=decision.target.value,
            intent=intent.value,
            reason=decision.reason,
        )

        # 3. Choose provider.
        provider = self._cloud_llm if decision.target == RoutingTarget.CLOUD else self._edge_llm

        # 4. LLM stream → TTS stream → audio_sink.
        await self._stream_response(provider, transcript, audio_sink, state)

        # 5. Build the Turn record.
        finished_at_dt = datetime.now(UTC)
        metrics = self._metrics(state, decision, provider, started_at_dt, finished_at_dt)
        if decision.target == RoutingTarget.CLOUD:
            self._router.record_cost(metrics.cost_usd)

        return Turn(
            turn_id=state.turn_id,
            user_text=transcript.text,
            assistant_text="".join(state.assistant_text).strip(),
            metrics=metrics,
        )

    async def interrupt(self) -> None:
        """Cancel any in-flight LLM / TTS tasks. Used for barge-in."""
        for task in list(self._active_tasks):
            task.cancel()
        # Don't await here — let the turn's own try/finally clean up.
        log.info("manager.interrupted", n_cancelled=len(self._active_tasks))

    # ── internals ───────────────────────────────────────────────────

    async def _run_asr(
        self, audio: AsyncIterator[AudioFrame], state: _TurnState
    ) -> Transcript:
        final: Transcript | None = None
        async for partial in self._asr.stream(audio):
            if partial.is_final:
                final = Transcript(
                    text=partial.text,
                    confidence=partial.confidence,
                    duration_ms=(time.perf_counter() - state.started_at) * 1000.0,
                )
                break
        if final is None:
            raise RuntimeError("ASR stream ended without a final transcript")
        state.asr_done_at = time.perf_counter()
        log.info("turn.asr_done", turn_id=state.turn_id, text=final.text[:80])
        return final

    async def _stream_response(
        self,
        provider: LLMProvider,
        transcript: Transcript,
        audio_sink: asyncio.Queue[AudioFrame] | None,
        state: _TurnState,
    ) -> None:
        """Run LLM → TTS → sink, recording timing as it flows."""
        request = LLMRequest(
            transcript=transcript,
            system=self._config.system_prompt,
            max_tokens=self._config.max_response_tokens,
        )

        # We tee the LLM stream so we can both feed TTS and record text.
        token_queue: asyncio.Queue[Token | None] = asyncio.Queue(maxsize=64)

        producer = asyncio.create_task(
            self._produce_tokens(provider, request, token_queue, state),
            name=f"llm.{state.turn_id}",
        )
        consumer = asyncio.create_task(
            self._consume_tokens_to_tts(token_queue, audio_sink, state),
            name=f"tts.{state.turn_id}",
        )
        self._active_tasks.update({producer, consumer})

        try:
            await asyncio.gather(producer, consumer)
        except asyncio.CancelledError:
            state.barged_in = True
            log.info("turn.barged_in", turn_id=state.turn_id)
            # Make sure both tasks get torn down.
            for t in (producer, consumer):
                if not t.done():
                    t.cancel()
            raise
        finally:
            self._active_tasks.discard(producer)
            self._active_tasks.discard(consumer)

    async def _produce_tokens(
        self,
        provider: LLMProvider,
        request: LLMRequest,
        sink: asyncio.Queue[Token | None],
        state: _TurnState,
    ) -> None:
        try:
            async for token in provider.stream(request):
                if state.llm_first_token_at == 0.0:
                    state.llm_first_token_at = time.perf_counter()
                state.assistant_text.append(token.text)
                if token.text:
                    state.tokens_out += 1
                await sink.put(token)
            state.llm_done_at = time.perf_counter()
        finally:
            # Sentinel: tell the consumer we're done, even on failure.
            await sink.put(None)

    async def _consume_tokens_to_tts(
        self,
        source: asyncio.Queue[Token | None],
        audio_sink: asyncio.Queue[AudioFrame] | None,
        state: _TurnState,
    ) -> None:
        async def _token_iter() -> AsyncIterator[Token]:
            while True:
                tok = await source.get()
                if tok is None:
                    return
                yield tok

        async for frame in self._tts.stream(_token_iter()):
            if state.tts_first_audio_at == 0.0:
                state.tts_first_audio_at = time.perf_counter()
            if audio_sink is not None:
                await audio_sink.put(frame)

    def _metrics(
        self,
        state: _TurnState,
        decision: RoutingDecision,
        provider: LLMProvider,
        started_at_dt: datetime,
        finished_at_dt: datetime,
    ) -> TurnMetrics:
        ms = lambda a, b: max(0.0, (b - a) * 1000.0)  # noqa: E731
        ttft_ms = (
            ms(state.started_at, state.tts_first_audio_at)
            if state.tts_first_audio_at
            else ms(state.started_at, state.llm_first_token_at)
        )
        # Rough token-in estimate: 1 token per ~4 chars.
        tokens_in = max(1, len("".join(state.assistant_text)) // 4)
        cost = provider.cost_estimate(tokens_in, state.tokens_out)

        return TurnMetrics(
            turn_id=state.turn_id,
            started_at=started_at_dt,
            finished_at=finished_at_dt,
            routing=decision,
            asr_ms=ms(state.started_at, state.asr_done_at),
            routing_ms=ms(state.asr_done_at, state.routing_done_at),
            llm_first_token_ms=ms(state.routing_done_at, state.llm_first_token_at),
            llm_total_ms=ms(state.llm_first_token_at, state.llm_done_at or state.llm_first_token_at),
            tts_first_audio_ms=ms(state.llm_first_token_at, state.tts_first_audio_at),
            ttft_ms=ttft_ms,
            provider=provider.provider_id if decision.target == RoutingTarget.CLOUD else Provider.LOCAL,
            tokens_in=tokens_in,
            tokens_out=state.tokens_out,
            cost_usd=cost,
            edge_resolved=(decision.target == RoutingTarget.EDGE),
            barged_in=state.barged_in,
        )


def _new_turn_id() -> str:
    return uuid.uuid4().hex[:10]
