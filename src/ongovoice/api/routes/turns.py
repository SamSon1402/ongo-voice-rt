"""Turn endpoints — drive the pipeline from text or audio.

POST /turns           run a turn from a transcript string (fast, text-only path)
GET  /turns/{id}      fetch a turn record
GET  /metrics         rolling per-provider TTFT + cost stats
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ongovoice.asr.mock_asr import MockASR
from ongovoice.core import AudioFrame, Turn
from ongovoice.llm.mock_provider import MockProvider
from ongovoice.pipeline import ConversationManager
from ongovoice.router import IntentClassifier, RouterPolicy
from ongovoice.tts.mock_tts import MockTTS

router = APIRouter()
log = structlog.get_logger(__name__)


# ── request / response ─────────────────────────────────────────────


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    asr_confidence: float = Field(default=0.92, ge=0.0, le=1.0)


class TurnResponse(BaseModel):
    turn_id: str
    user_text: str
    assistant_text: str
    routing_target: str
    routing_reason: str
    intent: str
    confidence: float
    ttft_ms: float
    asr_ms: float
    llm_first_token_ms: float
    cost_usd: float
    edge_resolved: bool
    barged_in: bool


class MetricsResponse(BaseModel):
    turns_total: int
    edge_resolved_pct: float
    avg_ttft_ms: float
    avg_cost_per_turn_usd: float
    monthly_cost_estimate_usd: float


# ── in-memory turn log + shared router (state) ─────────────────────


class _Store:
    """Per-process store. Replace with redis in production."""

    def __init__(self) -> None:
        self._turns: dict[str, Turn] = {}
        self._lock = asyncio.Lock()
        self._router = RouterPolicy()

    async def add(self, turn: Turn) -> None:
        async with self._lock:
            self._turns[turn.turn_id] = turn

    async def get(self, turn_id: str) -> Turn | None:
        async with self._lock:
            return self._turns.get(turn_id)

    async def all(self) -> list[Turn]:
        async with self._lock:
            return list(self._turns.values())

    @property
    def router(self) -> RouterPolicy:
        return self._router


_STORE = _Store()


def get_store() -> _Store:
    return _STORE


# ── endpoints ──────────────────────────────────────────────────────


@router.post("/turns", response_model=TurnResponse)
async def run_turn(
    body: TurnRequest,
    store: Annotated[_Store, Depends(get_store)],
) -> TurnResponse:
    """Drive a single turn through the pipeline from text input.

    This is the fast path used by the dashboard demo. The full bidi-audio
    WebSocket lives at /sessions/{id}/ws.
    """
    # Compose providers per-request (cheap when they're mocks).
    asr = MockASR(final_text=body.text, final_confidence=body.asr_confidence)
    edge_llm = MockProvider(canned_response=_edge_canned_for(body.text))
    cloud_llm = MockProvider(
        canned_response=_cloud_canned_for(body.text),
        first_token_ms=380.0,  # cloud RTT
    )
    tts = MockTTS()

    manager = ConversationManager(
        asr=asr,
        edge_llm=edge_llm,
        cloud_llm=cloud_llm,
        tts=tts,
        router=store.router,
        classifier=IntentClassifier(),
    )

    audio = _silent_audio_stream(duration_ms=600.0)
    try:
        turn = await manager.handle_turn(audio)
    except Exception as exc:  # noqa: BLE001
        log.exception("turn.failed", error=str(exc))
        raise HTTPException(500, detail=f"turn failed: {exc}") from exc

    await store.add(turn)
    log.info(
        "api.turn_done",
        turn_id=turn.turn_id,
        target=turn.metrics.routing.target.value,
        ttft_ms=round(turn.metrics.ttft_ms, 1),
        cost=round(turn.metrics.cost_usd, 4),
    )
    return _serialize(turn)


@router.get("/turns/{turn_id}", response_model=TurnResponse)
async def get_turn(
    turn_id: str,
    store: Annotated[_Store, Depends(get_store)],
) -> TurnResponse:
    turn = await store.get(turn_id)
    if not turn:
        raise HTTPException(404, detail=f"turn {turn_id} not found")
    return _serialize(turn)


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(
    store: Annotated[_Store, Depends(get_store)],
) -> MetricsResponse:
    turns = await store.all()
    n = len(turns)
    if n == 0:
        return MetricsResponse(
            turns_total=0,
            edge_resolved_pct=0.0,
            avg_ttft_ms=0.0,
            avg_cost_per_turn_usd=0.0,
            monthly_cost_estimate_usd=0.0,
        )
    edge_pct = sum(1 for t in turns if t.metrics.edge_resolved) / n * 100
    avg_ttft = sum(t.metrics.ttft_ms for t in turns) / n
    avg_cost = sum(t.metrics.cost_usd for t in turns) / n
    # 30 turns / day × 30 days = 900 turns/month per device
    monthly = avg_cost * 900
    return MetricsResponse(
        turns_total=n,
        edge_resolved_pct=round(edge_pct, 1),
        avg_ttft_ms=round(avg_ttft, 1),
        avg_cost_per_turn_usd=round(avg_cost, 5),
        monthly_cost_estimate_usd=round(monthly, 3),
    )


# ── helpers ────────────────────────────────────────────────────────


def _edge_canned_for(text: str) -> str:
    """Plausible-sounding edge response for the demo."""
    lower = text.lower()
    if "timer" in lower:
        return "Timer set. I'll chirp when it's done."
    if "light" in lower or "lamp" in lower:
        return "Done — lights adjusted."
    if "next" in lower or "song" in lower or "music" in lower:
        return "Next track playing."
    if "time" in lower:
        now = datetime.now(UTC).strftime("%H:%M")
        return f"It's {now}."
    return "Got it."


def _cloud_canned_for(text: str) -> str:
    """Plausible-sounding cloud response for the demo."""
    lower = text.lower()
    if "calendar" in lower or "schedule" in lower:
        return "You have two things this afternoon — design sync at 2:30 and a call with Joonatan at 4."
    if "weather" in lower:
        return "Mild and overcast in Paris today — around 16 degrees."
    return "Let me think about that for a moment, then I'll get back to you."


async def _silent_audio_stream(duration_ms: float) -> "asyncio.AsyncIterator[AudioFrame]":
    """Yields ~30 ms silent frames for the requested duration."""
    frame_ms = 30.0
    samples = int(16000 * frame_ms / 1000)
    n_frames = int(duration_ms / frame_ms)
    for _ in range(n_frames):
        yield AudioFrame(sample_rate=16000, pcm=np.zeros(samples, dtype=np.int16))
        await asyncio.sleep(0)  # cooperative yield


def _serialize(turn: Turn) -> TurnResponse:
    m = turn.metrics
    return TurnResponse(
        turn_id=turn.turn_id,
        user_text=turn.user_text,
        assistant_text=turn.assistant_text,
        routing_target=m.routing.target.value,
        routing_reason=m.routing.reason,
        intent=m.routing.intent.value,
        confidence=round(m.routing.confidence, 3),
        ttft_ms=round(m.ttft_ms, 1),
        asr_ms=round(m.asr_ms, 1),
        llm_first_token_ms=round(m.llm_first_token_ms, 1),
        cost_usd=round(m.cost_usd, 5),
        edge_resolved=m.edge_resolved,
        barged_in=m.barged_in,
    )
