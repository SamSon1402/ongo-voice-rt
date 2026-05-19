"""Router policy contracts.

These tests pin down the actual routing rules. If any of them break,
edge-resolved percentage will drop and cost will spike — exactly the
metrics we'd report to the CTO weekly. They're load-bearing.
"""

from __future__ import annotations

from ongovoice.core import Intent, RoutingTarget, Transcript
from ongovoice.router import RouterPolicy


def _tr(text: str, confidence: float = 0.9) -> Transcript:
    return Transcript(text=text, confidence=confidence, duration_ms=400)


async def test_timer_always_edge() -> None:
    pol = RouterPolicy()
    d = await pol.route(_tr("set a 5 min timer"), Intent.TIMER)
    assert d.target == RoutingTarget.EDGE


async def test_lights_always_edge() -> None:
    pol = RouterPolicy()
    d = await pol.route(_tr("turn off the lights"), Intent.LIGHTS)
    assert d.target == RoutingTarget.EDGE


async def test_open_qa_always_cloud() -> None:
    pol = RouterPolicy()
    d = await pol.route(_tr("explain entanglement"), Intent.OPEN_QA)
    assert d.target == RoutingTarget.CLOUD


async def test_low_confidence_goes_cloud() -> None:
    pol = RouterPolicy(confidence_threshold=0.65)
    d = await pol.route(_tr("hmm uhh maybe", confidence=0.45), Intent.SMALL_TALK)
    assert d.target == RoutingTarget.CLOUD


async def test_high_confidence_unknown_goes_edge() -> None:
    pol = RouterPolicy(confidence_threshold=0.65)
    d = await pol.route(_tr("hello there", confidence=0.92), Intent.SMALL_TALK)
    assert d.target == RoutingTarget.EDGE


async def test_budget_gate_forces_edge_when_over() -> None:
    pol = RouterPolicy(monthly_budget_usd=0.01)
    pol.record_cost(0.02)
    d = await pol.route(_tr("what's on my calendar"), Intent.CALENDAR)
    assert d.target == RoutingTarget.EDGE
    assert "budget" in d.reason


async def test_privacy_gate_blocks_sensitive_keywords() -> None:
    pol = RouterPolicy()
    d = await pol.route(
        _tr("my password is hunter2", confidence=0.3),
        Intent.UNKNOWN,
    )
    assert d.target == RoutingTarget.EDGE
    assert "privacy" in d.reason


async def test_reason_is_human_readable() -> None:
    pol = RouterPolicy()
    d = await pol.route(_tr("what time is it"), Intent.TIME)
    # Operators read these in logs — they should be intelligible.
    assert len(d.reason) > 5 and " " in d.reason
