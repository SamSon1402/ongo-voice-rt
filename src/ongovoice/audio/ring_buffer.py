"""Async ring buffer for audio frames.

Producer (mic / WS / file) pushes frames; consumers (VAD, ASR) read with
back-pressure. Bounded — drops the oldest frame on overflow rather than
blocking the producer, because a mic that pauses is worse than a 20ms
audio gap.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator

import structlog

from ongovoice.core import AudioFrame
from ongovoice.core.errors import AudioError

log = structlog.get_logger(__name__)


class AsyncRingBuffer:
    """Bounded async queue of AudioFrames with drop-oldest semantics."""

    def __init__(self, *, max_frames: int = 200) -> None:
        if max_frames <= 0:
            raise AudioError(f"max_frames must be > 0, got {max_frames}")
        self._buf: deque[AudioFrame] = deque(maxlen=max_frames)
        self._cv = asyncio.Condition()
        self._closed = False
        self._overflow_count = 0

    async def put(self, frame: AudioFrame) -> None:
        async with self._cv:
            if self._closed:
                raise AudioError("ring buffer closed")
            if len(self._buf) == self._buf.maxlen:
                self._overflow_count += 1
                if self._overflow_count % 50 == 1:
                    log.warning("ring_buffer.overflow", dropped=self._overflow_count)
            self._buf.append(frame)
            self._cv.notify_all()

    async def get(self) -> AudioFrame | None:
        """Returns next frame or None if buffer was closed."""
        async with self._cv:
            while not self._buf and not self._closed:
                await self._cv.wait()
            if self._buf:
                return self._buf.popleft()
            return None  # closed and empty

    async def close(self) -> None:
        async with self._cv:
            self._closed = True
            self._cv.notify_all()

    async def __aiter__(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self.get()
            if frame is None:
                return
            yield frame

    # ── introspection ───────────────────────────────────────────────

    @property
    def depth(self) -> int:
        return len(self._buf)

    @property
    def overflow_count(self) -> int:
        return self._overflow_count
