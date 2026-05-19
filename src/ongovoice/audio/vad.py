"""Voice-activity detection.

Two implementations:
  * EnergyVAD — simple RMS threshold + hangover. Works, is dumb.
  * SileroVAD — wraps the Silero ONNX model. Scaffolded.

Both satisfy the same Protocol. The pipeline picks one at config time;
EnergyVAD is the default for tests and the dev loop.
"""

from __future__ import annotations

from ongovoice.core import AudioFrame


class EnergyVAD:
    """Threshold-based VAD with a hangover period.

    Tuned for 16 kHz int16 audio. The default threshold is intentionally
    permissive — false positives flow downstream to wake-word, which is
    the actual gate. False *negatives* (missed speech) would be fatal.
    """

    def __init__(
        self,
        *,
        rms_threshold: float = 300.0,
        hangover_ms: float = 200.0,
        frame_ms: float = 20.0,
    ) -> None:
        self._threshold = rms_threshold
        self._hangover_frames = int(hangover_ms / frame_ms)
        self._silent_frames = self._hangover_frames + 1  # start "in silence"

    def is_speech(self, frame: AudioFrame) -> bool:
        is_loud = frame.rms > self._threshold
        if is_loud:
            self._silent_frames = 0
            return True
        self._silent_frames += 1
        # Hangover: report speech for a bit after the last loud frame so
        # word-final consonants aren't clipped from ASR input.
        return self._silent_frames <= self._hangover_frames

    def reset(self) -> None:
        self._silent_frames = self._hangover_frames + 1


class SileroVAD:
    """Production VAD wrapping the Silero ONNX model.

    Scaffolded — loading the model requires a frame size of exactly
    512 samples at 16 kHz (32 ms windows). The ring buffer's frame
    size has to be compatible. Real impl drops in once we settle on
    the device frame size.
    """

    def __init__(self, *, model_path: str | None = None, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model_path = model_path
        # self._session = ort.InferenceSession(model_path, ...)
        raise NotImplementedError(
            "SileroVAD scaffolded — wiring the ONNX session lands when we "
            "finalize the device frame size. See module docstring."
        )

    def is_speech(self, frame: AudioFrame) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
