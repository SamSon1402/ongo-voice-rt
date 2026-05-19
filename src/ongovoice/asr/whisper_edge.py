"""Edge ASR using faster-whisper (CTranslate2).

Scaffolded. Real impl loads `whisper-tiny.en` int8 once at startup,
keeps a buffer of audio samples, and flushes to `transcribe()` when
the VAD reports end-of-speech.

Why faster-whisper over openai-whisper:
  * CTranslate2 quantization → 4x lighter on ARM
  * Streaming-friendly API: returns segments incrementally
  * No CUDA required for the tiny model
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from ongovoice.core import AudioFrame, PartialTranscript


class WhisperEdgeASR:
    """Edge ASR. Scaffolded — loads on first use once weights are in place."""

    def __init__(
        self,
        *,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        weights_dir: Path | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._weights_dir = weights_dir
        # self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        raise NotImplementedError(
            "WhisperEdgeASR scaffolded — needs faster-whisper weights cached. "
            "Run `huggingface-cli download Systran/faster-whisper-tiny.en` once "
            "into the weights/ folder, then drop the `WhisperModel(...)` line in."
        )

    @property
    def is_edge(self) -> bool:
        return True

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[PartialTranscript]:
        raise NotImplementedError
        # NOTE for the real implementation:
        #   1. accumulate audio into a numpy buffer
        #   2. on every ~250ms boundary, call self._model.transcribe(buf, ...)
        #   3. yield the partial; on VAD end-of-speech, yield is_final=True
        if False:  # placate the type checker — keeps the AsyncIterator shape
            yield  # type: ignore[unreachable]
