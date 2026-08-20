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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect

if TYPE_CHECKING:
    import sphn

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
    TextEvent,
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
    GatewayAgentToolEvent as AgentToolEvent,
)
from .wire import (
    GatewayAgentToolStatusEvent as AgentToolStatusEvent,
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
    "AgentToolEvent",
    "AgentToolStatusEvent",
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

# The gateway origin, the socket it serves, and the header it authenticates on.
DEFAULT_BASE_URL = "https://api.prosodyai.app"
GATEWAY_PATH = "/v1/realtime"
API_KEY_HEADER = "x-api-key"

# What an origin may arrive as, and the transport scheme it connects on.
_WS_SCHEME = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}

# How long the uplink waits for the gateway to bind its model session before
# it calls the socket dead. Cold-starting a replica loads Mimi, ProsodySSM,
# Nemotron, Sortformer, and PersonaPlex, so the budget is generous; the point
# is that the wait ends in a raised error instead of a silent call.
GATEWAY_READY_TIMEOUT = 120.0


@dataclass
class ReadyEvent:
    """The gateway handshake completed: the model is live on the socket."""


GatewayEvent = (
    ReadyEvent | TextEvent | IdentityEvent | TranscriptEvent | ModelEvent | ConversationEvent
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


def gateway_ws_url(*, base_url: str = DEFAULT_BASE_URL) -> str:
    """The gateway socket for one origin. ``base_url`` is http(s) or ws(s).

    The URL carries no credential. A URL is the most copied string a networked
    system produces: proxies record it, transports log it, and every failed
    connect prints it in a traceback. The key travels as ``x-api-key`` on the
    handshake instead, which is what the gateway reads.
    """
    parsed = urlsplit((base_url or "").strip().rstrip("/"))
    scheme = _WS_SCHEME.get(parsed.scheme.lower())
    if scheme is None or not parsed.netloc:
        raise GatewayEnvError(
            "gateway base_url must be an http(s) or ws(s) origin such as "
            f"{DEFAULT_BASE_URL}, got {base_url!r}"
        )
    if parsed.query or parsed.fragment:
        raise GatewayEnvError(
            f"gateway base_url is an origin and carries no query or fragment, got {base_url!r}"
        )
    return urlunsplit((scheme, parsed.netloc, f"{parsed.path}{GATEWAY_PATH}", "", ""))


@dataclass(frozen=True)
class GatewayConnection:
    """Resolved gateway endpoint: the socket, and the key that opens it.

    The key is kept out of ``repr`` so the endpoint stays printable. Reads only
    ``PROSODYAI_API_KEY``.
    """

    url: str
    api_key: str = field(repr=False)

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
        return cls(url=gateway_ws_url(base_url=base_url or DEFAULT_BASE_URL), api_key=key)


@dataclass(frozen=True)
class FullDuplexBridgeConfig:
    """Connection settings for one full-duplex session.

    The key is kept out of ``repr`` so the config stays printable.
    """

    url: str
    api_key: str = field(repr=False)
    room_sample_rate: int = 16_000
    publish_sample_rate: int = GATEWAY_SAMPLE_RATE

    @property
    def headers(self) -> dict[str, str]:
        """The handshake credential, carried on the header so the URL stays loggable."""
        return {API_KEY_HEADER: self.api_key}


class FullDuplexBridge:
    """Mix-friendly duplex session: PCM16 uplink in, PCM16 downlink out + events."""

    def __init__(self, config: FullDuplexBridgeConfig) -> None:
        self._config = config
        self._ready = asyncio.Event()
        self._closed = False

    @property
    def ready(self) -> asyncio.Event:
        return self._ready

    async def _send_uplink(
        self,
        uplink_pcm16: AsyncIterator[bytes],
        websocket: ClientConnection,
        writer: sphn.OpusStreamWriter,
    ) -> None:
        """Resample room PCM onto the gateway's 80 ms grid and stream it as Opus.

        Hold the room's audio until the gateway binds its model session, then
        stream every frame. Testing readiness per frame instead discarded
        whatever the caller said during the handshake, and a handshake that
        never arrived discarded the whole call in silence.
        """
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=GATEWAY_READY_TIMEOUT)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "gateway never completed its handshake within "
                f"{GATEWAY_READY_TIMEOUT}s; no uplink audio was sent"
            ) from None
        pending = np.zeros(0, dtype=np.float32)
        async for frame in uplink_pcm16:
            if self._closed or not frame:
                continue
            float_24k = resample_float32(
                pcm16_le_to_float32(frame),
                self._config.room_sample_rate,
                GATEWAY_SAMPLE_RATE,
            )
            pending = np.concatenate([pending, float_24k])
            while pending.shape[0] >= GATEWAY_FRAME_SAMPLES:
                block = pending[:GATEWAY_FRAME_SAMPLES]
                pending = pending[GATEWAY_FRAME_SAMPLES:]
                packet = writer.append_pcm(block)
                if packet:
                    await websocket.send(bytes([KIND_AUDIO]) + packet)

    async def _receive_downlink(
        self,
        websocket: ClientConnection,
        reader: sphn.OpusStreamReader,
        *,
        on_downlink_pcm16: Callable[[bytes], Awaitable[None]],
        on_event: Callable[[GatewayEvent], Awaitable[None]] | None,
    ) -> None:
        """Split the gateway's frames into published model PCM and typed events."""
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
            additional_headers=self._config.headers,
            max_size=16 * 1024 * 1024,
            open_timeout=120.0,
            # The gateway queues uplink frames and steps them on a pump task,
            # so its receive loop keeps reading through GPU spikes. Client
            # pings stay disabled: uplink audio itself is the liveness signal
            # on this socket.
            ping_interval=None,
        ) as websocket:
            send_task = asyncio.create_task(
                self._send_uplink(
                    uplink_pcm16,
                    websocket,
                    sphn.OpusStreamWriter(GATEWAY_SAMPLE_RATE),
                ),
                name="duplex-uplink",
            )
            try:
                await self._receive_downlink(
                    websocket,
                    sphn.OpusStreamReader(GATEWAY_SAMPLE_RATE),
                    on_downlink_pcm16=on_downlink_pcm16,
                    on_event=on_event,
                )
            finally:
                self._closed = True
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)

    def close(self) -> None:
        self._closed = True
