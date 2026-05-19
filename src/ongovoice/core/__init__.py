"""Core domain types and protocols. No I/O. No SDK imports."""

from ongovoice.core.errors import (
    AudioError,
    BargedIn,
    ConfigError,
    ProviderError,
    RoutingError,
    VoiceError,
)
from ongovoice.core.protocols import (
    ASRProvider,
    LLMProvider,
    Router,
    TTSProvider,
    VADProvider,
    WakeWordDetector,
)
from ongovoice.core.types import (
    AudioFrame,
    Intent,
    LLMRequest,
    PartialTranscript,
    Provider,
    RoutingDecision,
    RoutingTarget,
    Token,
    Transcript,
    Turn,
    TurnMetrics,
)

__all__ = [
    # types
    "AudioFrame",
    "Intent",
    "LLMRequest",
    "PartialTranscript",
    "Provider",
    "RoutingDecision",
    "RoutingTarget",
    "Token",
    "Transcript",
    "Turn",
    "TurnMetrics",
    # protocols
    "ASRProvider",
    "LLMProvider",
    "Router",
    "TTSProvider",
    "VADProvider",
    "WakeWordDetector",
    # errors
    "AudioError",
    "BargedIn",
    "ConfigError",
    "ProviderError",
    "RoutingError",
    "VoiceError",
]
