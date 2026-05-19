"""Intent classifier."""

from __future__ import annotations

import pytest

from ongovoice.core import Intent, Transcript
from ongovoice.router import IntentClassifier


def _tr(text: str, confidence: float = 0.92) -> Transcript:
    return Transcript(text=text, confidence=confidence, duration_ms=500.0)


@pytest.fixture
def clf() -> IntentClassifier:
    return IntentClassifier()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("set a timer for five minutes", Intent.TIMER),
        ("Hey Ongo, start a 10 minute timer", Intent.TIMER),
        ("turn off the lights", Intent.LIGHTS),
        ("lights on", Intent.LIGHTS),
        ("next song", Intent.MEDIA),
        ("skip", Intent.MEDIA),
        ("what time is it", Intent.TIME),
        ("what's the weather today", Intent.WEATHER),
        ("what's on my calendar this afternoon", Intent.CALENDAR),
        ("hello there", Intent.SMALL_TALK),
        ("explain quantum entanglement", Intent.OPEN_QA),
        ("why is the sky blue?", Intent.OPEN_QA),
    ],
)
def test_intents(clf: IntentClassifier, text: str, expected: Intent) -> None:
    intent, _ = clf.classify(_tr(text))
    assert intent == expected


def test_confidence_capped_by_asr(clf: IntentClassifier) -> None:
    """If ASR is shaky, classifier confidence should be too."""
    intent_high, conf_high = clf.classify(_tr("set a timer for 5 minutes", confidence=0.95))
    intent_low, conf_low = clf.classify(_tr("set a timer for 5 minutes", confidence=0.30))
    assert intent_high == intent_low == Intent.TIMER
    assert conf_low < conf_high


def test_empty_returns_unknown(clf: IntentClassifier) -> None:
    intent, conf = clf.classify(Transcript(text=" ", confidence=0.9, duration_ms=100))
    assert intent == Intent.UNKNOWN
    assert conf == 0.0
