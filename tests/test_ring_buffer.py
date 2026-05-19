"""Async ring buffer."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ongovoice.audio import AsyncRingBuffer
from ongovoice.core import AudioFrame
from ongovoice.core.errors import AudioError


def _frame(value: int = 0) -> AudioFrame:
    return AudioFrame(sample_rate=16_000, pcm=np.full(160, value, dtype=np.int16))


async def test_put_and_get() -> None:
    rb = AsyncRingBuffer(max_frames=4)
    await rb.put(_frame(1))
    await rb.put(_frame(2))
    f1 = await rb.get()
    f2 = await rb.get()
    assert f1 is not None and f1.pcm[0] == 1
    assert f2 is not None and f2.pcm[0] == 2


async def test_close_returns_none() -> None:
    rb = AsyncRingBuffer(max_frames=4)
    await rb.close()
    assert await rb.get() is None


async def test_rejects_invalid_capacity() -> None:
    with pytest.raises(AudioError):
        AsyncRingBuffer(max_frames=0)


async def test_overflow_drops_oldest() -> None:
    rb = AsyncRingBuffer(max_frames=2)
    await rb.put(_frame(1))
    await rb.put(_frame(2))
    await rb.put(_frame(3))  # should evict frame(1)
    assert rb.overflow_count == 1
    assert rb.depth == 2
    f = await rb.get()
    assert f is not None and f.pcm[0] == 2


async def test_async_iter() -> None:
    rb = AsyncRingBuffer(max_frames=4)
    for i in range(3):
        await rb.put(_frame(i + 1))
    await rb.close()
    received = [f.pcm[0] async for f in rb]
    assert received == [1, 2, 3]


async def test_get_waits_for_producer() -> None:
    """A consumer that arrives before any frame blocks until one shows up."""
    rb = AsyncRingBuffer(max_frames=4)

    async def producer() -> None:
        await asyncio.sleep(0.02)
        await rb.put(_frame(42))

    asyncio.create_task(producer())
    f = await asyncio.wait_for(rb.get(), timeout=1.0)
    assert f is not None and f.pcm[0] == 42
