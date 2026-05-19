"""LLM pricing table.

USD per 1M tokens. Pulled from each vendor's pricing page (May 2026).
This file is intentionally a flat dict so it's trivially updatable
without touching code. Production: pull from a config service.
"""

from __future__ import annotations

from ongovoice.core import Provider


# (input_per_mtok, output_per_mtok) — USD
PRICING: dict[tuple[Provider, str], tuple[float, float]] = {
    # Anthropic
    (Provider.ANTHROPIC, "claude-haiku-4-5"): (1.00, 5.00),
    (Provider.ANTHROPIC, "claude-sonnet-4-6"): (3.00, 15.00),
    # OpenAI
    (Provider.OPENAI, "gpt-4o-mini"): (0.15, 0.60),
    (Provider.OPENAI, "gpt-4o"): (2.50, 10.00),
    # Mistral
    (Provider.MISTRAL, "mistral-small-latest"): (0.20, 0.60),
    # Local — free
    (Provider.LOCAL, "llama-3.2-3b-int4"): (0.0, 0.0),
    # Mock — free, for tests
    (Provider.MOCK, "mock"): (0.0, 0.0),
}


def cost_usd(provider: Provider, model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute USD cost for one call."""
    rates = PRICING.get((provider, model))
    if rates is None:
        # Unknown model — return 0 rather than blow up. We log a warning
        # in the caller; missing pricing data shouldn't break a turn.
        return 0.0
    in_rate, out_rate = rates
    return (tokens_in / 1_000_000) * in_rate + (tokens_out / 1_000_000) * out_rate
