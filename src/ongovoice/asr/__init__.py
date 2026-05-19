"""ASR providers."""

from ongovoice.asr.mock_asr import MockASR
from ongovoice.asr.whisper_edge import WhisperEdgeASR

__all__ = ["MockASR", "WhisperEdgeASR"]
