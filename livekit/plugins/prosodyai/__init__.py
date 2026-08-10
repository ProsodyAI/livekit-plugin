"""ProsodyAI plugin for LiveKit Agents.

Inference runs server-side; this plugin is pure I/O. ``RealtimeModel`` and
``FullDuplexBridge`` carry full-duplex speech with persistent speaker identity.
"""

from .full_duplex import (
    AgentThoughtEvent,
    AgentToolEvent,
    BargeInEvent,
    ConversationEvent,
    FullDuplexBridge,
    FullDuplexBridgeConfig,
    GatewayConnection,
    GatewayEnvError,
    GatewayEvent,
    IdentityEvent,
    IdentityResolvedEvent,
    ModelEvent,
    NewSpeakerEvent,
    ReadyEvent,
    SpeakerChangeEvent,
    StateDeltaEvent,
    TextEvent,
    TranscriptDelta,
    TranscriptEvent,
    TurnBoundaryEvent,
    gateway_ws_url,
    parse_control_event,
)
from .realtime import (
    RealtimeModel,
    RealtimeSession,
)
from .version import __version__

__all__ = [
    "AgentThoughtEvent",
    "AgentToolEvent",
    "BargeInEvent",
    "ConversationEvent",
    "FullDuplexBridge",
    "FullDuplexBridgeConfig",
    "GatewayConnection",
    "GatewayEnvError",
    "GatewayEvent",
    "IdentityEvent",
    "IdentityResolvedEvent",
    "ModelEvent",
    "NewSpeakerEvent",
    "ReadyEvent",
    "RealtimeModel",
    "RealtimeSession",
    "SpeakerChangeEvent",
    "StateDeltaEvent",
    "TextEvent",
    "TranscriptDelta",
    "TranscriptEvent",
    "TurnBoundaryEvent",
    "gateway_ws_url",
    "parse_control_event",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class ProsodyAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ProsodyAIPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
