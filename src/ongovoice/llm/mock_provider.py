"""Mock LLM provider for tests and the demo loop.

Streams a canned response token-by-token with a realistic TTFT and
inter-token delay. Deterministic given a seed.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from ongovoice.core import LLMRequest, Provider, Token


class MockProvider:
    """Pretend LLM that streams tokens with plausible timing."""

    def __init__(
        self,
        *,
        canned_response: str = (
            "Sure — I'll take care of that for you. Anything else you need?"
        ),
        first_token_ms: float = 80.0,
        inter_token_ms: float = 18.0,
        seed: int = 0,
    ) -> None:
        self._response = canned_response
        self._first_token_ms = first_token_ms
        self._inter_token_ms = inter_token_ms
        self._rng = random.Random(seed)

    @property
    def provider_id(self) -> Provider:
        return Provider.MOCK

    @property
    def model_name(self) -> str:
        return "mock"

    async def stream(self, req: LLMRequest) -> AsyncIterator[Token]:  # noqa: ARG002
        # Realistic TTFT.
        await asyncio.sleep(self._first_token_ms / 1000.0)
        tokens = self._response.split()
        for i, word in enumerate(tokens):
            jitter = self._rng.gauss(0, self._inter_token_ms * 0.15)
            await asyncio.sleep(max(0.0, (self._inter_token_ms + jitter) / 1000.0))
            yield Token(text=word + (" " if i < len(tokens) - 1 else ""), index=i)
        yield Token(text="", index=len(tokens), is_final=True)

    def cost_estimate(self, tokens_in: int, tokens_out: int) -> float:  # noqa: ARG002
        return 0.0

    # Helpers for tests
    def with_response(self, response: str) -> "MockProvider":
        return MockProvider(
            canned_response=response,
            first_token_ms=self._first_token_ms,
            inter_token_ms=self._inter_token_ms,
        )
