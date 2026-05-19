"""Pipeline manager — wires audio → ASR → router → LLM → TTS together."""

from ongovoice.pipeline.manager import ConversationManager, ManagerConfig

__all__ = ["ConversationManager", "ManagerConfig"]
