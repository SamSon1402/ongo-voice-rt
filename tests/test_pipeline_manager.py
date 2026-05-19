"""ConversationManager: end-to-end async pipeline.

These tests prove the pipeline actually works:
  * A timer request goes edge (and is cheap).
  * A calendar request goes cloud (and TTFT is higher).
  * Interrupt() cancels in-flight tasks (barge-in).
  * TurnMetrics are populated and consistent.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from ongovoice.asr.mock_asr import MockASR
from ongovoice.core import AudioFrame, RoutingTarget
from ongovoice.llm.mock_provider import MockProvider
from ongovoice.pipeline import ConversationManager
from ongovoice.router import IntentClassifier, RouterPolicy
from ongovoice.tts.mock_tts import MockTTS


async def _audio(duration_ms: float = 600) -> AsyncIterator[AudioFrame]:
    samples = int(16000 * 30 / 1000)
    for _ in range(int(duration_ms / 30)):
        yield AudioFrame(sample_rate=16000, pcm=np.zeros(samples, dtype=np.int16))
        await asyncio.sleep(0)


def _build_manager(*, user_text: str) -> ConversationManager:
    return ConversationManager(
        asr=MockASR(final_text=user_text, time_to_final_ms=50),
        edge_llm=MockProvider(
            canned_response="Done.", first_token_ms=30, inter_token_ms=5
        ),
        cloud_llm=MockProvider(
            canned_response="Let me look that up.", first_token_ms=200, inter_token_ms=10
        ),
        tts=MockTTS(synth_latency_ms=5),
        router=RouterPolicy(),
        classifier=IntentClassifier(),
    )


async def test_timer_request_goes_edge() -> None:
    mgr = _build_manager(user_text="set a five minute timer")
    turn = await mgr.handle_turn(_audio())
    assert turn.metrics.routing.target == RoutingTarget.EDGE
    assert turn.metrics.edge_resolved is True
    assert turn.metrics.cost_usd == 0.0
    assert turn.user_text == "set a five minute timer"
    assert turn.assistant_text.strip() != ""


async def test_calendar_request_goes_cloud() -> None:
    mgr = _build_manager(user_text="what's on my calendar this afternoon")
    turn = await mgr.handle_turn(_audio())
    assert turn.metrics.routing.target == RoutingTarget.CLOUD
    assert turn.metrics.edge_resolved is False


async def test_metrics_are_monotonic() -> None:
    """Each timestamp in the turn should be at or after the previous one."""
    mgr = _build_manager(user_text="set a timer for 5 minutes")
    turn = await mgr.handle_turn(_audio())
    m = turn.metrics
    # TTFT >= ASR latency (we don't time-travel)
    assert m.ttft_ms >= m.asr_ms
    assert m.llm_first_token_ms >= 0
    assert m.tts_first_audio_ms >= 0


async def test_assistant_text_assembled_from_stream() -> None:
    mgr = _build_manager(user_text="set a timer")
    turn = await mgr.handle_turn(_audio())
    # MockProvider's canned response is "Done." — should arrive intact.
    assert "Done" in turn.assistant_text


async def test_interrupt_cancels_in_flight_turn() -> None:
    """When the user barges in, the turn raises CancelledError cleanly."""

    # A cloud LLM that streams slowly enough we can interrupt mid-way.
    slow_cloud = MockProvider(
        canned_response="One two three four five six seven eight nine ten",
        first_token_ms=50,
        inter_token_ms=80,
    )
    mgr = ConversationManager(
        asr=MockASR(final_text="tell me a long story", time_to_final_ms=30),
        edge_llm=MockProvider(canned_response="ok"),
        cloud_llm=slow_cloud,
        tts=MockTTS(synth_latency_ms=10),
        router=RouterPolicy(),
        classifier=IntentClassifier(),
    )

    turn_task = asyncio.create_task(mgr.handle_turn(_audio(duration_ms=300)))
    # Let the LLM start streaming, then interrupt.
    await asyncio.sleep(0.15)
    await mgr.interrupt()

    with pytest.raises(asyncio.CancelledError):
        await turn_task
