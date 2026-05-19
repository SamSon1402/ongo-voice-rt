"""Intent classifier.

A small bag of regexes that map utterances to Intent + confidence.
Production: this gets replaced by a fine-tuned distilbert running on
the edge, but the *interface* (`(intent, confidence)`) stays the same.

The reason rules work surprisingly well here: Ongo's actual command
distribution is heavy-tailed — "set a timer", "play music", "what time
is it" cover >60% of all turns from public assistant datasets. The
hard cases go to cloud anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ongovoice.core import Intent, Transcript


@dataclass(frozen=True)
class _Rule:
    intent: Intent
    pattern: re.Pattern[str]
    confidence: float  # confidence we assign when this rule fires


# Ordered list — first match wins. Patterns are case-insensitive.
_RULES: list[_Rule] = [
    _Rule(Intent.TIMER, re.compile(
        r"\b(set|start|cancel|stop)\b.*\b(timer|alarm|countdown)\b", re.I), 0.95),
    _Rule(Intent.TIMER, re.compile(
        r"\btimer\s+(for|of)\s+\d+\b", re.I), 0.95),
    _Rule(Intent.LIGHTS, re.compile(
        r"\b(turn|dim|brighten|switch)\s+(on|off|up|down)?\s*(the\s+)?(light|lamp|lights)\b", re.I), 0.93),
    _Rule(Intent.LIGHTS, re.compile(
        r"\b(lights|lamp)\s+(on|off|dim|bright)\b", re.I), 0.92),
    _Rule(Intent.MEDIA, re.compile(
        r"\b(play|pause|stop|skip|next|previous|prev)\b.*\b(song|track|music|playlist)?\b", re.I), 0.90),
    _Rule(Intent.MEDIA, re.compile(
        r"\b(next|previous|skip|pause|resume)\b", re.I), 0.85),
    _Rule(Intent.TIME, re.compile(
        r"\bwhat(?:'s| is)\s+the\s+time\b", re.I), 0.96),
    _Rule(Intent.TIME, re.compile(
        r"\bwhat\s+time\s+is\s+it\b", re.I), 0.96),
    _Rule(Intent.WEATHER, re.compile(
        r"\b(weather|forecast|rain|temperature)\b", re.I), 0.78),
    _Rule(Intent.CALENDAR, re.compile(
        r"\b(calendar|meeting|schedule|appointment|what.{0,8}on\s+(my\s+)?(calendar|schedule))\b", re.I), 0.60),
    _Rule(Intent.SMALL_TALK, re.compile(
        r"^\s*(hi|hello|hey|good\s+(morning|evening)|how(?:'s| is) it going)\b", re.I), 0.80),
]


class IntentClassifier:
    """Maps a Transcript to an (Intent, confidence)."""

    def classify(self, transcript: Transcript) -> tuple[Intent, float]:
        text = transcript.text.strip()
        if not text:
            return Intent.UNKNOWN, 0.0

        for rule in _RULES:
            if rule.pattern.search(text):
                # Combine rule confidence with ASR confidence — the
                # latter caps the former. We don't trust a 95% rule on
                # a 50% ASR.
                conf = rule.confidence * (0.5 + 0.5 * transcript.confidence)
                return rule.intent, conf

        # Long, vague, or question-shaped utterances go to OPEN_QA.
        is_questiony = text.endswith("?") or text.lower().startswith(
            ("what", "why", "how", "tell me", "explain", "do you")
        )
        if is_questiony or len(text.split()) > 10:
            return Intent.OPEN_QA, 0.55 * transcript.confidence

        return Intent.UNKNOWN, 0.3 * transcript.confidence
