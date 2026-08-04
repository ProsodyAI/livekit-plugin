"""ProsodyAI acoustic speech analysis plugin for LiveKit Agents."""

from livekit.agents import Plugin

from .analyzer import ProsodyAnalyzer
from .conversation import Conversation
from .log import logger
from .models import (
    AcousticChange,
    AcousticDelta,
    AcousticEvent,
    AcousticFrames,
    AcousticState,
    AcousticWindow,
    FeaturePoint,
    FramePoint,
    Speaker,
    TranscriptTurn,
)
from .version import __version__

__all__ = [
    "AcousticChange",
    "AcousticDelta",
    "AcousticEvent",
    "AcousticFrames",
    "AcousticState",
    "AcousticWindow",
    "Conversation",
    "FeaturePoint",
    "FramePoint",
    "ProsodyAIPlugin",
    "ProsodyAnalyzer",
    "Speaker",
    "TranscriptTurn",
    "__version__",
]


class ProsodyAIPlugin(Plugin):
    """Register the ProsodyAI package with the LiveKit Agents runtime."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__ or __name__, logger)


Plugin.register_plugin(ProsodyAIPlugin())
