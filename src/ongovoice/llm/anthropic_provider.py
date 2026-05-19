"""Anthropic provider.

Streams from `messages.create` with `stream=True`. Lazy-imports the SDK
so the package works without it installed.

Why we like Anthropic for Ongo's cloud tier:
  * Fast TTFT on Haiku — typically 200-400ms from Paris.
  * Streaming is well-shaped: discrete `content_block_delta` events.
  * Sonnet 4.6 is our fallback for genuinely hard questions.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import structlog

from ongovoice.core import LLMRequest, Provider, Token
from ongovoice.core.errors import ConfigError, ProviderError
from ongovoice.llm.pricing import cost_usd

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic  # pragma: no cover

log = structlog.get_logger(__name__)


class AnthropicProvider:
    """Streaming Anthropic adapter."""

    DEFAULT_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key= explicitly."
            )
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._client: "AsyncAnthropic" | None = None  # lazy

    @property
    def provider_id(self) -> Provider:
        return Provider.ANTHROPIC

    @property
    def model_name(self) -> str:
        return self._model_name

    async def stream(self, req: LLMRequest) -> AsyncIterator[Token]:
        client = self._ensure_client()
        idx = 0
        try:
            async with client.messages.stream(
                model=self._model_name,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system=req.system,
                messages=[{"role": "user", "content": req.transcript.text}],
            ) as stream:
                async for delta in stream.text_stream:
                    if not delta:
                        continue
                    yield Token(text=delta, index=idx)
                    idx += 1
            yield Token(text="", index=idx, is_final=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("anthropic.stream_failed", error=str(exc))
            raise ProviderError(f"anthropic stream failed: {exc}") from exc

    def cost_estimate(self, tokens_in: int, tokens_out: int) -> float:
        return cost_usd(Provider.ANTHROPIC, self._model_name, tokens_in, tokens_out)

    # ── internal ───────────────────────────────────────────────────

    def _ensure_client(self) -> "AsyncAnthropic":
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ConfigError(
                "anthropic SDK not installed. `pip install 'ongovoice[anthropic]'`."
            ) from exc
        self._client = AsyncAnthropic(api_key=self._api_key, timeout=self._timeout_s)
        return self._client
