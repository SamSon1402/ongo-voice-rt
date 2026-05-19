"""Domain types for the voice pipeline.

Everything frozen, everything typed. A `Turn` is a single user-to-Ongo
exchange. A `Transcript` is one user utterance. A `Token` is one LLM
output unit (we don't care if it's a BPE chunk or a word — depends on
the provider).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── audio ───────────────────────────────────────────────────────────


class AudioFrame(BaseModel):
    """A chunk of mono PCM audio at a fixed sample rate.

    We standardise on 16 kHz, int16, mono everywhere internally. Resampling
    happens at the device boundary, not in the pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sample_rate: int = Field(default=16_000, gt=0)
    pcm: np.ndarray  # int16, shape (n,)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check_pcm(self) -> Self:
        if self.pcm.ndim != 1:
            raise ValueError(f"pcm must be mono (1-D), got shape {self.pcm.shape}")
        if self.pcm.dtype != np.int16:
            raise ValueError(f"pcm must be int16, got {self.pcm.dtype}")
        return self

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.pcm) / self.sample_rate

    @property
    def rms(self) -> float:
        """Root-mean-square energy. Cheap loudness proxy for VU meters."""
        if len(self.pcm) == 0:
            return 0.0
        # int16 → float to avoid overflow on square.
        return float(np.sqrt(np.mean(self.pcm.astype(np.float32) ** 2)))


# ── ASR ─────────────────────────────────────────────────────────────


class PartialTranscript(BaseModel):
    """A streaming ASR hypothesis, not yet finalized."""

    model_config = ConfigDict(frozen=True)

    text: str
    is_final: bool = False
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class Transcript(BaseModel):
    """A finalized user utterance."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    duration_ms: float = Field(ge=0)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── intent + routing ────────────────────────────────────────────────


class Intent(StrEnum):
    """Coarse intent taxonomy. The router uses this + confidence to gate."""

    TIMER = "timer"          # always edge
    LIGHTS = "lights"        # always edge
    MEDIA = "media"          # always edge ("next song", "pause")
    TIME = "time"            # always edge
    WEATHER = "weather"      # edge if cached, cloud otherwise
    CALENDAR = "calendar"    # always cloud (needs context)
    OPEN_QA = "open_qa"      # always cloud
    SMALL_TALK = "small_talk"  # edge if confident, else cloud
    UNKNOWN = "unknown"      # cloud


class RoutingTarget(StrEnum):
    """Where a turn was sent."""

    EDGE = "edge"
    CLOUD = "cloud"


class RoutingDecision(BaseModel):
    """Why a turn went where it went. Logged in full for audit."""

    model_config = ConfigDict(frozen=True)

    target: RoutingTarget
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str  # e.g. "intent=timer always edge", "conf 0.42 < 0.65 threshold"


# ── LLM ─────────────────────────────────────────────────────────────


class Provider(StrEnum):
    """LLM provider id."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MISTRAL = "mistral"
    LOCAL = "local"
    MOCK = "mock"


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    model_config = ConfigDict(frozen=True)

    transcript: Transcript
    system: str
    user_context: dict[str, str] = Field(default_factory=dict)
    max_tokens: int = Field(default=200, gt=0, le=4096)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)


class Token(BaseModel):
    """One streaming output unit from an LLM."""

    model_config = ConfigDict(frozen=True)

    text: str
    index: int = Field(ge=0)
    is_final: bool = False


# ── turn-level telemetry ────────────────────────────────────────────


class TurnMetrics(BaseModel):
    """Everything we measure about one user-to-Ongo exchange."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    started_at: datetime
    finished_at: datetime
    routing: RoutingDecision

    # timing — all milliseconds, all wall-clock from mic-end-of-utterance
    asr_ms: float = Field(ge=0)
    routing_ms: float = Field(ge=0)
    llm_first_token_ms: float = Field(ge=0)
    llm_total_ms: float = Field(ge=0)
    tts_first_audio_ms: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)  # end-to-end: ASR start → first TTS audio frame

    # business metrics
    provider: Provider | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    edge_resolved: bool = False
    barged_in: bool = False


class Turn(BaseModel):
    """The full record of one user-to-Ongo exchange."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    user_text: str
    assistant_text: str
    metrics: TurnMetrics
