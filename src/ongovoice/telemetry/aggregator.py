"""Rolling aggregator over TurnMetrics.

Pulls turn records, returns derived stats. Thread-safe (asyncio.Lock).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from ongovoice.core import Provider, TurnMetrics


@dataclass(slots=True)
class AggregateView:
    turns: int
    edge_resolved_pct: float
    avg_ttft_ms: float
    p99_ttft_ms: float
    avg_cost_per_turn: float
    by_provider: dict[Provider, "ProviderStats"]


@dataclass(slots=True)
class ProviderStats:
    turns: int
    avg_ttft_ms: float
    avg_cost_per_turn: float


class TurnAggregator:
    """Rolling window of turn metrics. Bounded to `window` entries."""

    def __init__(self, *, window: int = 200) -> None:
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self._buf: deque[TurnMetrics] = deque(maxlen=window)
        self._lock = asyncio.Lock()

    async def record(self, metrics: TurnMetrics) -> None:
        async with self._lock:
            self._buf.append(metrics)

    async def view(self) -> AggregateView:
        async with self._lock:
            turns = list(self._buf)

        if not turns:
            return AggregateView(
                turns=0,
                edge_resolved_pct=0.0,
                avg_ttft_ms=0.0,
                p99_ttft_ms=0.0,
                avg_cost_per_turn=0.0,
                by_provider={},
            )

        n = len(turns)
        edge_pct = sum(1 for t in turns if t.edge_resolved) / n * 100
        ttfts = sorted(t.ttft_ms for t in turns)
        avg_ttft = sum(ttfts) / n
        p99 = ttfts[min(n - 1, int(0.99 * n))]
        avg_cost = sum(t.cost_usd for t in turns) / n

        # Per-provider rollup
        by_prov: dict[Provider, list[TurnMetrics]] = {}
        for t in turns:
            if t.provider is None:
                continue
            by_prov.setdefault(t.provider, []).append(t)

        prov_stats: dict[Provider, ProviderStats] = {}
        for prov, ts in by_prov.items():
            prov_stats[prov] = ProviderStats(
                turns=len(ts),
                avg_ttft_ms=sum(t.ttft_ms for t in ts) / len(ts),
                avg_cost_per_turn=sum(t.cost_usd for t in ts) / len(ts),
            )

        return AggregateView(
            turns=n,
            edge_resolved_pct=edge_pct,
            avg_ttft_ms=avg_ttft,
            p99_ttft_ms=p99,
            avg_cost_per_turn=avg_cost,
            by_provider=prov_stats,
        )
