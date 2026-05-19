"""Shared fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest

from ongovoice.core import AudioFrame


@pytest.fixture
def silent_frame() -> AudioFrame:
    return AudioFrame(sample_rate=16_000, pcm=np.zeros(480, dtype=np.int16))


@pytest.fixture
def loud_frame() -> AudioFrame:
    # ~1 kHz tone at high amplitude
    t = np.arange(480) / 16_000
    pcm = (np.sin(2 * np.pi * 1000 * t) * 16000).astype(np.int16)
    return AudioFrame(sample_rate=16_000, pcm=pcm)


@pytest.fixture
async def silent_audio_stream() -> AsyncIterator[AudioFrame]:
    """30 frames of 30ms silence ≈ 900ms of audio."""

    async def _gen() -> AsyncIterator[AudioFrame]:
        for _ in range(30):
            yield AudioFrame(sample_rate=16_000, pcm=np.zeros(480, dtype=np.int16))

    return _gen()
