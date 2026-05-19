"""OpenAI provider.

Same shape as the Anthropic adapter. Streams from `chat.completions.create`
with `stream=True`. Yields tokens as `ChatCompletionChunk.choices[0].delta.content`
arrives.

We use this as the A/B comparison provider — comparing TTFT and cost
per-turn against Anthropic is a useful production signal.
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
    from openai import AsyncOpenAI  # pragma: no cover

log = structlog.get_logger(__name__)


class OpenAIProvider:
    """Streaming OpenAI adapter."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ConfigError(
                "OPENAI_API_KEY not set. Export it or pass api_key= explicitly."
            )
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._client: "AsyncOpenAI" | None = None

    @property
    def provider_id(self) -> Provider:
        return Provider.OPENAI

    @property
    def model_name(self) -> str:
        return self._model_name

    async def stream(self, req: LLMRequest) -> AsyncIterator[Token]:
        client = self._ensure_client()
        idx = 0
        try:
            stream = await client.chat.completions.create(
                model=self._model_name,
                stream=True,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                messages=[
                    {"role": "system", "content": req.system},
                    {"role": "user", "content": req.transcript.text},
                ],
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                yield Token(text=delta, index=idx)
                idx += 1
            yield Token(text="", index=idx, is_final=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("openai.stream_failed", error=str(exc))
            raise ProviderError(f"openai stream failed: {exc}") from exc

    def cost_estimate(self, tokens_in: int, tokens_out: int) -> float:
        return cost_usd(Provider.OPENAI, self._model_name, tokens_in, tokens_out)

    def _ensure_client(self) -> "AsyncOpenAI":
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ConfigError(
                "openai SDK not installed. `pip install 'ongovoice[openai]'`."
            ) from exc
        self._client = AsyncOpenAI(api_key=self._api_key, timeout=self._timeout_s)
        return self._client
