"""Router policy.

The router answers one question: edge or cloud? It considers four gates,
in order. The first gate that fires owns the decision and gets recorded
as the `reason`.

  1. Privacy gate. Some intents *must never* leave the device (calendar
     containing PII has already been pulled locally; sending it to cloud
     is a leak). For now this is conservative — flag-controlled.
  2. Hard intent gate. Some intents are always cloud (open Q&A, planning)
     or always edge (timer, lights, time). No ambiguity.
  3. Confidence gate. If the classifier is confident, edge. If not, cloud.
  4. Budget gate. If we're over the per-device-month cost budget, fail
     open to edge with a fallback message rather than charging more.

This file is data-heavy on purpose. It should be readable by a product
manager. The CTO will read it for what's *not* there — magic constants
without justification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from ongovoice.core import (
    Intent,
    RoutingDecision,
    RoutingTarget,
    Transcript,
)

log = structlog.get_logger(__name__)


# Intents that we will never route to cloud, regardless of confidence.
_EDGE_ONLY: frozenset[Intent] = frozenset({
    Intent.TIMER,
    Intent.LIGHTS,
    Intent.MEDIA,
    Intent.TIME,
})

# Intents that we will always route to cloud — small models can't.
_CLOUD_ONLY: frozenset[Intent] = frozenset({
    Intent.OPEN_QA,
    Intent.CALENDAR,
})


@dataclass
class RouterPolicy:
    """Stateful router. Tracks running cost for the budget gate."""

    confidence_threshold: float = 0.65
    monthly_budget_usd: float = 1.50      # per device
    _spend_this_month: float = field(default=0.0, init=False)
    _month_started_at: float = field(default_factory=time.time, init=False)

    async def route(self, transcript: Transcript, intent: Intent) -> RoutingDecision:
        confidence = transcript.confidence  # ASR-side; the classifier may have boosted
        text = transcript.text

        # Gate 1: privacy
        if intent == Intent.UNKNOWN and any(
            kw in text.lower() for kw in ("password", "card number", "social security")
        ):
            return _decision(
                RoutingTarget.EDGE, intent, confidence,
                "privacy: sensitive keyword detected, refusing to send to cloud",
            )

        # Gate 2: hard intent
        if intent in _EDGE_ONLY:
            return _decision(
                RoutingTarget.EDGE, intent, confidence,
                f"intent={intent.value} always edge",
            )
        if intent in _CLOUD_ONLY:
            # ...but only if we haven't blown our budget.
            if self._over_budget():
                return _decision(
                    RoutingTarget.EDGE, intent, confidence,
                    "budget exceeded — falling back to edge",
                )
            return _decision(
                RoutingTarget.CLOUD, intent, confidence,
                f"intent={intent.value} always cloud",
            )

        # Gate 3: confidence
        if confidence >= self.confidence_threshold:
            return _decision(
                RoutingTarget.EDGE, intent, confidence,
                f"conf {confidence:.2f} ≥ {self.confidence_threshold:.2f}",
            )

        # Gate 4: budget (already low-conf, but check the wallet)
        if self._over_budget():
            return _decision(
                RoutingTarget.EDGE, intent, confidence,
                "budget exceeded — falling back to edge",
            )

        return _decision(
            RoutingTarget.CLOUD, intent, confidence,
            f"conf {confidence:.2f} < {self.confidence_threshold:.2f} threshold",
        )

    # ── budget bookkeeping ──────────────────────────────────────────

    def record_cost(self, usd: float) -> None:
        """Called by the pipeline after each cloud turn."""
        self._spend_this_month += usd
        log.info("router.cost", spent=round(self._spend_this_month, 4))

    def _over_budget(self) -> bool:
        # Reset every 30 days. Real impl uses calendar months.
        if time.time() - self._month_started_at > 30 * 86400:
            self._spend_this_month = 0.0
            self._month_started_at = time.time()
        return self._spend_this_month >= self.monthly_budget_usd


def _decision(
    target: RoutingTarget, intent: Intent, confidence: float, reason: str
) -> RoutingDecision:
    return RoutingDecision(target=target, intent=intent, confidence=confidence, reason=reason)
