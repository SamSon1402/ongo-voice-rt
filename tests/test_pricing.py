"""Pricing math sanity."""

from __future__ import annotations

from ongovoice.core import Provider
from ongovoice.llm.pricing import cost_usd


def test_local_provider_is_free() -> None:
    assert cost_usd(Provider.LOCAL, "llama-3.2-3b-int4", 1000, 1000) == 0.0


def test_unknown_model_returns_zero() -> None:
    # We log a warning in caller; missing pricing shouldn't blow up the turn.
    assert cost_usd(Provider.ANTHROPIC, "claude-imaginary-99", 1000, 1000) == 0.0


def test_haiku_cost_for_short_turn() -> None:
    # 200 in, 80 out → small but nonzero
    cost = cost_usd(Provider.ANTHROPIC, "claude-haiku-4-5", 200, 80)
    assert 0 < cost < 0.01


def test_sonnet_is_more_expensive_than_haiku() -> None:
    haiku = cost_usd(Provider.ANTHROPIC, "claude-haiku-4-5", 1000, 1000)
    sonnet = cost_usd(Provider.ANTHROPIC, "claude-sonnet-4-6", 1000, 1000)
    assert sonnet > haiku
