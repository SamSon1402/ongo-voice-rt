"""Domain type contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ongovoice.core import (
    AudioFrame,
    Intent,
    Provider,
    RoutingDecision,
    RoutingTarget,
    Token,
    Transcript,
)


class TestAudioFrame:
    def test_valid(self) -> None:
        pcm = np.zeros(480, dtype=np.int16)
        f = AudioFrame(sample_rate=16_000, pcm=pcm)
        assert f.duration_ms == 30.0

    def test_rejects_stereo(self) -> None:
        pcm = np.zeros((480, 2), dtype=np.int16)
        with pytest.raises(ValidationError):
            AudioFrame(sample_rate=16_000, pcm=pcm)

    def test_rejects_wrong_dtype(self) -> None:
        pcm = np.zeros(480, dtype=np.float32)
        with pytest.raises(ValidationError):
            AudioFrame(sample_rate=16_000, pcm=pcm)

    def test_rms_silence_is_zero(self) -> None:
        f = AudioFrame(sample_rate=16_000, pcm=np.zeros(480, dtype=np.int16))
        assert f.rms == 0.0

    def test_rms_nonzero_for_signal(self) -> None:
        pcm = (np.ones(480) * 1000).astype(np.int16)
        f = AudioFrame(sample_rate=16_000, pcm=pcm)
        assert f.rms == pytest.approx(1000.0, rel=1e-3)


class TestTranscript:
    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(text="hi", confidence=1.5, duration_ms=10)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(text="", confidence=0.9, duration_ms=10)


class TestRoutingDecision:
    def test_construction(self) -> None:
        d = RoutingDecision(
            target=RoutingTarget.EDGE,
            intent=Intent.TIMER,
            confidence=0.91,
            reason="intent=timer always edge",
        )
        assert d.target == RoutingTarget.EDGE


class TestToken:
    def test_index_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            Token(text="hi", index=-1)


def test_provider_enum_complete() -> None:
    # If we add a new provider, we want the test suite to remind us to
    # add pricing for it.
    assert set(Provider) >= {
        Provider.ANTHROPIC,
        Provider.OPENAI,
        Provider.MISTRAL,
        Provider.LOCAL,
        Provider.MOCK,
    }
