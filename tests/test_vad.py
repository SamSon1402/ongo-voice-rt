"""VAD contract."""

from __future__ import annotations

from ongovoice.audio import EnergyVAD
from ongovoice.core import AudioFrame


def test_silence_is_not_speech(silent_frame: AudioFrame) -> None:
    vad = EnergyVAD(rms_threshold=300.0)
    assert vad.is_speech(silent_frame) is False


def test_loud_frame_is_speech(loud_frame: AudioFrame) -> None:
    vad = EnergyVAD(rms_threshold=300.0)
    assert vad.is_speech(loud_frame) is True


def test_hangover_keeps_speech_for_a_while(loud_frame: AudioFrame, silent_frame: AudioFrame) -> None:
    """After a loud frame, the next few silent frames are still 'speech'."""
    vad = EnergyVAD(rms_threshold=300.0, hangover_ms=100.0, frame_ms=20.0)
    assert vad.is_speech(loud_frame) is True
    # 5 hangover frames at 20ms = 100ms — all reported as speech.
    for _ in range(5):
        assert vad.is_speech(silent_frame) is True
    # 6th silent frame falls past the hangover.
    assert vad.is_speech(silent_frame) is False


def test_reset_clears_hangover_state(loud_frame: AudioFrame, silent_frame: AudioFrame) -> None:
    vad = EnergyVAD(rms_threshold=300.0)
    vad.is_speech(loud_frame)
    vad.reset()
    assert vad.is_speech(silent_frame) is False
