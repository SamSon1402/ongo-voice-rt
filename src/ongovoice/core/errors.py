"""Error taxonomy."""

from __future__ import annotations


class VoiceError(Exception):
    """Base for everything raised inside ongovoice."""


class AudioError(VoiceError):
    """PCM shape, sample-rate mismatch, ring-buffer overrun."""


class ProviderError(VoiceError):
    """An LLM / ASR / TTS provider failed (timeout, 5xx, auth)."""


class RoutingError(VoiceError):
    """Router couldn't decide (no policy matched)."""


class ConfigError(VoiceError):
    """Bad pipeline config — missing API key, unknown provider, etc."""


class BargedIn(VoiceError):
    """Raised inside a streaming task when the user interrupts.

    This is *control flow*, not an error. The pipeline manager catches
    it and cleans up the orphaned LLM/TTS streams.
    """
