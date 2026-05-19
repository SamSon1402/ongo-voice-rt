"""Bidi audio WebSocket.

  client → server: binary PCM frames (16 kHz int16 little-endian)
  server → client: binary PCM frames (TTS output)
                   text JSON  (partial transcripts, routing, metrics)

The client speaks → we transcribe → route → stream the response back as
audio. Barge-in: if the client sends a frame while we're mid-response,
the manager cancels the in-flight LLM/TTS tasks.

This is the surface real devices talk to.
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ongovoice.asr.mock_asr import MockASR
from ongovoice.audio import AsyncRingBuffer
from ongovoice.core import AudioFrame
from ongovoice.llm.mock_provider import MockProvider
from ongovoice.pipeline import ConversationManager
from ongovoice.router import IntentClassifier, RouterPolicy
from ongovoice.tts.mock_tts import MockTTS

router = APIRouter()
log = structlog.get_logger(__name__)


@router.websocket("/sessions/{session_id}/ws")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    log.info("ws.connect", session_id=session_id)

    ring = AsyncRingBuffer(max_frames=400)
    audio_out: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=64)

    manager = ConversationManager(
        asr=MockASR(),
        edge_llm=MockProvider(canned_response="Got it."),
        cloud_llm=MockProvider(canned_response="Let me look that up for you.", first_token_ms=380),
        tts=MockTTS(),
        router=RouterPolicy(),
        classifier=IntentClassifier(),
    )

    receiver = asyncio.create_task(_receive_loop(websocket, ring, manager))
    sender = asyncio.create_task(_send_loop(websocket, audio_out))
    runner = asyncio.create_task(_turn_loop(manager, ring, audio_out, websocket))

    try:
        # Run until any of them finishes (typically receiver, on disconnect).
        done, pending = await asyncio.wait(
            {receiver, sender, runner}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        await ring.close()
        log.info("ws.disconnect", session_id=session_id)


async def _receive_loop(
    ws: WebSocket, ring: AsyncRingBuffer, manager: ConversationManager
) -> None:
    """Pull binary frames from the client; signal barge-in if we're talking."""
    while True:
        message = await ws.receive()
        if "bytes" in message and message["bytes"] is not None:
            pcm = np.frombuffer(message["bytes"], dtype=np.int16)
            frame = AudioFrame(sample_rate=16_000, pcm=pcm)
            await ring.put(frame)
            # Crude barge-in: if there's a turn in flight and we just got
            # loud audio, cancel.
            if frame.rms > 800 and manager._active_tasks:  # noqa: SLF001
                await manager.interrupt()
        elif "text" in message and message["text"]:
            # Control plane: client sends {"event": "end_of_utterance"} etc.
            # Not strictly necessary in the mock loop but the protocol is here.
            pass


async def _send_loop(ws: WebSocket, audio_out: asyncio.Queue[AudioFrame]) -> None:
    while True:
        frame = await audio_out.get()
        await ws.send_bytes(frame.pcm.tobytes())


async def _turn_loop(
    manager: ConversationManager,
    ring: AsyncRingBuffer,
    audio_out: asyncio.Queue[AudioFrame],
    ws: WebSocket,
) -> None:
    """Drive sequential turns. In production the trigger is end-of-utterance;
    here we run one full turn per drained ring fill."""
    while True:
        try:
            turn = await manager.handle_turn(ring.__aiter__(), audio_out)
            await ws.send_json(
                {
                    "event": "turn.done",
                    "turn_id": turn.turn_id,
                    "user_text": turn.user_text,
                    "assistant_text": turn.assistant_text,
                    "ttft_ms": round(turn.metrics.ttft_ms, 1),
                    "routing": turn.metrics.routing.target.value,
                    "reason": turn.metrics.routing.reason,
                    "cost_usd": round(turn.metrics.cost_usd, 4),
                }
            )
        except asyncio.CancelledError:
            await ws.send_json({"event": "barged_in"})
        except Exception as exc:  # noqa: BLE001
            log.exception("ws.turn_failed", error=str(exc))
            await ws.send_json({"event": "error", "detail": str(exc)})
            break
