"""The speech-backend protocol at both boundaries: PersonaPlex against a fake
gateway speaking the exact kind-tagged Opus wire, and the bridge against a
minimal fake backend that shares no code with PersonaPlex.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("sphn", reason="Opus bridging needs sphn")
import sphn
from livekit.plugins.prosodyai.full_duplex import (
    GATEWAY_FRAME_SAMPLES,
    GATEWAY_SAMPLE_RATE,
    KIND_AUDIO,
    KIND_EVENT,
    KIND_HANDSHAKE,
    KIND_IDENTITY,
    KIND_TEXT,
    KIND_TRANSCRIPT,
    FullDuplexBridge,
    FullDuplexBridgeConfig,
    GatewayEnvError,
    IdentityEvent,
    ReadyEvent,
    SpeakerChangeEvent,
    TextEvent,
    TranscriptEvent,
    parse_control_event,
)
from livekit.plugins.prosodyai.personaplex import (
    GatewayControlFrame,
    PersonaPlexBackend,
)
from livekit.plugins.prosodyai.speech_backend import (
    BackendCapabilityError,
    SessionOpened,
    SpeechAudio,
    SpeechBackendCapabilities,
    SpeechSessionConfig,
    SpeechText,
)
from websockets.asyncio.server import serve

ROOM_SAMPLE_RATE = 16_000
ROOM_FRAME_SAMPLES = ROOM_SAMPLE_RATE * 20 // 1000  # 20 ms room frames

IDENTITY = {
    "speaker_id": "speaker_0",
    "person_id": "person:158a2b2e",
    "display_name": "Ada",
    "resumed": True,
    "resolved_at_ms": 3000,
}

TRANSCRIPT = {
    "speaker_id": "speaker_0",
    "deltas": [
        {"text": "hello", "start_ms": 0, "end_ms": 320},
        {"text": "there", "start_ms": 320, "end_ms": 640},
    ],
}

SPEAKER_CHANGE = {
    "session_id": "sess-test",
    "timestamp_ms": 4000,
    "type": "prosodyai.speaker_change",
    "speaker_id": "speaker_0",
    "previous_speaker_id": None,
    "person_id": "person:158a2b2e",
    "display_name": "Ada",
    "is_agent": False,
}


async def _fake_gateway(websocket) -> None:
    """The production gateway's downlink in miniature: handshake first, then
    one of every frame family once uplink audio arrives."""
    await websocket.send(bytes([KIND_HANDSHAKE]))
    responded = False
    writer = sphn.OpusStreamWriter(GATEWAY_SAMPLE_RATE)
    async for message in websocket:
        if isinstance(message, str) or not message:
            continue
        if message[0] != KIND_AUDIO or responded:
            continue
        responded = True
        await websocket.send(bytes([KIND_TEXT]) + b"hello there")
        await websocket.send(bytes([KIND_IDENTITY]) + json.dumps(IDENTITY).encode("utf-8"))
        await websocket.send(bytes([KIND_TRANSCRIPT]) + json.dumps(TRANSCRIPT).encode("utf-8"))
        await websocket.send(bytes([KIND_EVENT]) + json.dumps(SPEAKER_CHANGE).encode("utf-8"))
        clock = np.arange(4 * GATEWAY_FRAME_SAMPLES, dtype=np.float32) / GATEWAY_SAMPLE_RATE
        tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * clock).astype(np.float32)
        for index in range(4):
            packet = writer.append_pcm(
                tone[index * GATEWAY_FRAME_SAMPLES : (index + 1) * GATEWAY_FRAME_SAMPLES]
            )
            if packet:
                await websocket.send(bytes([KIND_AUDIO]) + packet)


# ------------------------------------------------- the PersonaPlex backend


@pytest.mark.asyncio
async def test_personaplex_backend_speaks_the_gateway_wire() -> None:
    """Open, send audio, receive: every downlink family lands as its item."""
    async with serve(_fake_gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        backend = PersonaPlexBackend(url=f"ws://127.0.0.1:{port}", api_key="psk_test")

        capabilities = backend.capabilities
        assert capabilities == SpeechBackendCapabilities(
            full_duplex=True,
            accepts_voice_prompt=False,
            accepts_role_prompt=False,
            sample_rate=GATEWAY_SAMPLE_RATE,
        )

        await backend.open(SpeechSessionConfig())
        items: list = []
        heard_audio = asyncio.Event()

        async def _collect() -> None:
            async for item in backend.receive():
                items.append(item)
                if isinstance(item, SpeechAudio):
                    heard_audio.set()

        collector = asyncio.create_task(_collect())
        for _ in range(20):
            await backend.send_audio(np.zeros(GATEWAY_FRAME_SAMPLES, dtype=np.float32))
            await asyncio.sleep(0.005)
        await asyncio.wait_for(heard_audio.wait(), timeout=10.0)
        await backend.close()
        await asyncio.gather(collector, return_exceptions=True)

    assert isinstance(items[0], SessionOpened)

    texts = [item for item in items if isinstance(item, SpeechText)]
    assert [text.text for text in texts] == ["hello there"]

    audio = [item for item in items if isinstance(item, SpeechAudio)]
    assert audio and all(block.samples.dtype == np.float32 for block in audio)
    assert sum(block.samples.size for block in audio) > 0

    frames = {item.kind: item.payload for item in items if isinstance(item, GatewayControlFrame)}
    assert set(frames) == {KIND_IDENTITY, KIND_TRANSCRIPT, KIND_EVENT}
    assert json.loads(frames[KIND_IDENTITY]) == IDENTITY
    assert json.loads(frames[KIND_TRANSCRIPT]) == TRANSCRIPT
    assert json.loads(frames[KIND_EVENT]) == SPEAKER_CHANGE

    identity = parse_control_event(KIND_IDENTITY, frames[KIND_IDENTITY])
    assert isinstance(identity, IdentityEvent)
    assert identity.person_id == IDENTITY["person_id"]
    change = parse_control_event(KIND_EVENT, frames[KIND_EVENT])
    assert isinstance(change, SpeakerChangeEvent)
    assert change.timestamp_ms == 4000


@pytest.mark.asyncio
async def test_the_default_bridge_still_speaks_the_gateway() -> None:
    """The bridge with no backend argument is PersonaPlex on the gateway:
    the same typed events and published PCM as before the protocol existed."""
    async with serve(_fake_gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        bridge = FullDuplexBridge(
            FullDuplexBridgeConfig(
                url=f"ws://127.0.0.1:{port}",
                api_key="psk_test",
                room_sample_rate=ROOM_SAMPLE_RATE,
            )
        )
        assert isinstance(bridge.backend, PersonaPlexBackend)

        events: list = []
        downlink: list[bytes] = []
        complete = asyncio.Event()

        async def _on_pcm(pcm: bytes) -> None:
            downlink.append(pcm)
            if len(events) >= 5:
                complete.set()

        async def _on_event(event) -> None:
            events.append(event)

        async def _uplink():
            silence = np.zeros(ROOM_FRAME_SAMPLES, dtype=np.int16).tobytes()
            for _ in range(40):
                yield silence
                await asyncio.sleep(0.005)
            await asyncio.sleep(3600)

        runner = asyncio.create_task(
            bridge.run(_uplink(), on_downlink_pcm16=_on_pcm, on_event=_on_event)
        )
        await asyncio.wait_for(complete.wait(), timeout=10.0)
        bridge.close()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)

    assert isinstance(events[0], ReadyEvent)
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["hello there"]
    identity = next(e for e in events if isinstance(e, IdentityEvent))
    assert identity.display_name == "Ada"
    transcript = next(e for e in events if isinstance(e, TranscriptEvent))
    assert [d.text for d in transcript.deltas] == ["hello", "there"]
    change = next(e for e in events if isinstance(e, SpeakerChangeEvent))
    assert change.speaker_id == "speaker_0"
    assert downlink and all(len(pcm) % 2 == 0 for pcm in downlink)


# ----------------------------------------------------- any backend at all


ECHO_RATE = 8_000


class EchoBackend:
    """A minimal turn-based speaking loop that shares nothing with PersonaPlex:
    half-duplex, a role prompt channel, an 8 kHz clock."""

    def __init__(self) -> None:
        self.opened_with: SpeechSessionConfig | None = None
        self.closed = False
        self.heard: list[np.ndarray] = []
        self._items: asyncio.Queue = asyncio.Queue()

    @property
    def capabilities(self) -> SpeechBackendCapabilities:
        return SpeechBackendCapabilities(
            full_duplex=False,
            accepts_voice_prompt=False,
            accepts_role_prompt=True,
            sample_rate=ECHO_RATE,
        )

    async def open(self, config: SpeechSessionConfig) -> None:
        self.opened_with = config
        await self._items.put(SessionOpened())

    async def send_audio(self, samples: np.ndarray) -> None:
        self.heard.append(samples)
        if len(self.heard) == 3:
            await self._items.put(SpeechText(text="heard you"))
            await self._items.put(
                SpeechAudio(samples=np.full(ECHO_RATE // 100, 0.25, dtype=np.float32))
            )
            await self._items.put(None)

    async def receive(self):
        while True:
            item = await self._items.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_the_bridge_drives_any_backend_through_the_protocol() -> None:
    backend = EchoBackend()
    bridge = FullDuplexBridge(
        FullDuplexBridgeConfig(
            room_sample_rate=ROOM_SAMPLE_RATE,
            publish_sample_rate=ROOM_SAMPLE_RATE,
            role_prompt="answer briefly",
        ),
        backend=backend,
    )
    assert bridge.backend.capabilities.full_duplex is False

    events: list = []
    downlink: list[bytes] = []

    async def _on_pcm(pcm: bytes) -> None:
        downlink.append(pcm)

    async def _on_event(event) -> None:
        events.append(event)

    async def _uplink():
        silence = np.zeros(ROOM_FRAME_SAMPLES, dtype=np.int16).tobytes()
        for _ in range(6):
            yield silence
        await asyncio.sleep(3600)

    await asyncio.wait_for(
        bridge.run(_uplink(), on_downlink_pcm16=_on_pcm, on_event=_on_event),
        timeout=10.0,
    )

    # The declared role prompt channel carried the prompt to the session open.
    assert backend.opened_with == SpeechSessionConfig(
        voice_prompt=None, role_prompt="answer briefly"
    )
    assert backend.closed is True

    # Uplink landed on the backend's own 8 kHz clock: 20 ms is 160 samples.
    assert backend.heard and all(block.shape[0] == 160 for block in backend.heard)

    assert isinstance(events[0], ReadyEvent)
    assert [e.text for e in events if isinstance(e, TextEvent)] == ["heard you"]

    # 10 ms of backend audio published as PCM16 on the 16 kHz room clock.
    assert len(downlink) == 1
    assert len(downlink[0]) == (ECHO_RATE // 100) * 2 * 2


def test_a_prompt_outside_the_declared_capabilities_is_refused() -> None:
    with pytest.raises(BackendCapabilityError):
        FullDuplexBridge(FullDuplexBridgeConfig(voice_prompt="warm"), backend=EchoBackend())
    # PersonaPlex declares no prompt channel: the gateway primes server-side.
    with pytest.raises(BackendCapabilityError):
        FullDuplexBridge(
            FullDuplexBridgeConfig(url="ws://host", api_key="key", role_prompt="brief")
        )


def test_the_default_backend_requires_the_gateway_settings() -> None:
    with pytest.raises(GatewayEnvError):
        FullDuplexBridge(FullDuplexBridgeConfig())
