"""Full-duplex bridge to the ProsodyAI gateway (``WS /v1/realtime``).

Browser and LiveKit room audio is PCM. The gateway advances it through Mimi,
ProsodySSM speaker state, and Moshi, then returns 24 kHz Opus. This module is
the reusable transport bridge for that continuous model path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import ClassVar
from urllib.parse import urlencode

import numpy as np
from websockets.asyncio.client import connect as ws_connect

from .audio_resample import (
    float32_to_pcm16_le,
    pcm16_le_to_float32,
    resample_float32,
)
from .wire import (
    KIND_AUDIO,
    KIND_EVENT,
    KIND_HANDSHAKE,
    KIND_IDENTITY,
    KIND_TEXT,
    KIND_TRANSCRIPT,
    IdentityEvent,
    RoomEventType,
    TranscriptDelta,
    TranscriptEvent,
    parse_conversation_event,
    parse_gateway_model_event,
    parse_identity_payload,
    parse_transcript_payload,
)
from .wire import (
    ConversationBargeInEvent as BargeInEvent,
)
from .wire import (
    ConversationStateDeltaEvent as StateDeltaEvent,
)
from .wire import (
    ConversationTurnBoundaryEvent as TurnBoundaryEvent,
)
from .wire import (
    ConversationWireEvent as ConversationEvent,
)
from .wire import (
    GatewayIdentityResolvedEvent as IdentityResolvedEvent,
)
from .wire import (
    GatewayModelEvent as ModelEvent,
)
from .wire import (
    GatewayNewSpeakerEvent as NewSpeakerEvent,
)
from .wire import (
    GatewaySpeakerChangeEvent as SpeakerChangeEvent,
)

__all__ = [
    "GATEWAY_FRAME_SAMPLES",
    "GATEWAY_SAMPLE_RATE",
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
    "SpeakerChangeEvent",
    "StateDeltaEvent",
    "TextEvent",
    "TranscriptDelta",
    "TranscriptEvent",
    "TurnBoundaryEvent",
    "gateway_ws_url",
    "parse_control_event",
]

logger = logging.getLogger("livekit.plugins.prosodyai.full_duplex")

# The gateway's audio wire format: Opus at 24 kHz in 80 ms frames.
GATEWAY_SAMPLE_RATE = 24_000
GATEWAY_FRAME_SAMPLES = 1_920


@dataclass
class ReadyEvent:
    """The gateway handshake completed: the model is live on the socket."""


@dataclass(frozen=True)
class TextEvent:
    """One token of the model's inner monologue, as it is spoken."""

    TYPE: ClassVar[RoomEventType] = RoomEventType.TEXT

    text: str

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


GatewayEvent = (
    ReadyEvent
    | TextEvent
    | IdentityEvent
    | TranscriptEvent
    | ModelEvent
    | ConversationEvent
)


def parse_control_event(kind: int, payload: bytes) -> GatewayEvent | None:
    """Parse one gateway control frame (non-audio) into a typed event.

    JSON decoding happens here; the event shapes, type strings, and field
    names come from the shared wire vocabulary (``wire.py``), which mirrors
    the gateway's senders. Returns ``None`` for unparseable payloads.
    """
    if kind == KIND_HANDSHAKE:
        return ReadyEvent()
    if kind == KIND_TEXT:
        return TextEvent(text=payload.decode("utf-8", errors="replace"))
    if kind not in (KIND_EVENT, KIND_TRANSCRIPT, KIND_IDENTITY):
        return None
    try:
        frame = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(frame, dict):
        return None
    if kind == KIND_EVENT:
        # The 0x06 channel carries two committed families: the gateway's
        # relabeled tracker events (``prosodyai.*``, ``timestamp_ms``) and
        # the learned deciders' conversation events relayed verbatim in the
        # model wire's own shape (``state_delta``, ``turn_boundary``,
        # ``barge_in``; ``frame_ms``/``commit_ms``).
        model_event = parse_gateway_model_event(frame)
        if model_event is not None:
            return model_event
        return parse_conversation_event(frame)
    if kind == KIND_TRANSCRIPT:
        return parse_transcript_payload(frame)
    return parse_identity_payload(frame)


class GatewayEnvError(RuntimeError):
    """The gateway connection settings are missing or contradictory."""


def gateway_ws_url(
    *,
    api_key: str,
    base_url: str = "https://api.prosodyai.app",
) -> str:
    """Authenticated gateway WebSocket URL. ``base_url`` accepts http(s) or ws(s)."""
    key = (api_key or "").strip()
    if not key:
        raise GatewayEnvError(
            "gateway_ws_url received an empty api_key; pass the ProsodyAI API key for your organization"
        )
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise GatewayEnvError(
            "gateway_ws_url received an empty base_url; pass the gateway origin, e.g. https://api.prosodyai.app"
        )
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    url = f"{base}/v1/realtime"
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode({'api_key': key})}"


@dataclass(frozen=True)
class GatewayConnection:
    """Resolved gateway endpoint. Reads only ``PROSODYAI_API_KEY``."""

    url: str

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "GatewayConnection":
        environ = os.environ if env is None else env
        key = (api_key or environ.get("PROSODYAI_API_KEY") or "").strip()
        if not key:
            raise GatewayEnvError(
                "No gateway API key was found: PROSODYAI_API_KEY is unset and no "
                "api_key argument was passed; set the environment variable or pass api_key"
            )
        return cls(
            url=gateway_ws_url(
                api_key=key,
                base_url=base_url or "https://api.prosodyai.app",
            )
        )


@dataclass(frozen=True)
class FullDuplexBridgeConfig:
    """Connection settings for one full-duplex session."""

    url: str
    room_sample_rate: int = 16_000
    publish_sample_rate: int = GATEWAY_SAMPLE_RATE


class FullDuplexBridge:
    """Mix-friendly duplex session: PCM16 uplink in, PCM16 downlink out + events."""

    def __init__(self, config: FullDuplexBridgeConfig) -> None:
        self._config = config
        self._ready = asyncio.Event()
        self._closed = False

    @property
    def ready(self) -> asyncio.Event:
        return self._ready

    async def run(
        self,
        uplink_pcm16: AsyncIterator[bytes],
        *,
        on_downlink_pcm16: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[GatewayEvent], Awaitable[None]] | None = None,
    ) -> None:
        """Pump room PCM into the gateway and publish model PCM + identity/text events.

        ``uplink_pcm16`` yields little-endian mono PCM16 at ``room_sample_rate``
        (typically 20 ms LiveKit frames). Downlink PCM is also LE mono PCM16 at
        ``publish_sample_rate``.
        """
        try:
            import sphn
        except ImportError as exc:
            raise RuntimeError(
                "sphn is required for Opus bridging "
                "(pip install 'livekit-plugins-prosodyai[duplex]')"
            ) from exc

        async with ws_connect(
            self._config.url,
            max_size=16 * 1024 * 1024,
            open_timeout=120.0,
        ) as websocket:
            writer = sphn.OpusStreamWriter(GATEWAY_SAMPLE_RATE)
            reader = sphn.OpusStreamReader(GATEWAY_SAMPLE_RATE)
            uplink_buf = np.zeros(0, dtype=np.float32)

            async def send_uplink() -> None:
                nonlocal uplink_buf
                async for frame in uplink_pcm16:
                    if self._closed or not frame:
                        continue
                    float_room = pcm16_le_to_float32(frame)
                    float_24k = resample_float32(
                        float_room,
                        self._config.room_sample_rate,
                        GATEWAY_SAMPLE_RATE,
                    )
                    uplink_buf = np.concatenate([uplink_buf, float_24k])
                    while uplink_buf.shape[0] >= GATEWAY_FRAME_SAMPLES:
                        block = uplink_buf[:GATEWAY_FRAME_SAMPLES]
                        uplink_buf = uplink_buf[GATEWAY_FRAME_SAMPLES:]
                        packet = writer.append_pcm(block)
                        if packet:
                            await websocket.send(bytes([KIND_AUDIO]) + packet)

            send_task = asyncio.create_task(send_uplink(), name="duplex-uplink")
            try:
                async for message in websocket:
                    if self._closed:
                        break
                    if isinstance(message, str) or not message:
                        continue
                    kind = message[0]
                    payload = message[1:]
                    if kind == KIND_AUDIO:
                        pcm = reader.append_bytes(payload)
                        if pcm is None or pcm.size == 0:
                            continue
                        flat = np.asarray(pcm, dtype=np.float32).reshape(-1)
                        if self._config.publish_sample_rate != GATEWAY_SAMPLE_RATE:
                            flat = resample_float32(
                                flat,
                                GATEWAY_SAMPLE_RATE,
                                self._config.publish_sample_rate,
                            )
                        await on_downlink_pcm16(float32_to_pcm16_le(flat))
                        continue
                    control = parse_control_event(kind, payload)
                    if control is None:
                        continue
                    if isinstance(control, ReadyEvent):
                        self._ready.set()
                    if on_event is not None:
                        await on_event(control)
            finally:
                self._closed = True
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)

    def close(self) -> None:
        self._closed = True
