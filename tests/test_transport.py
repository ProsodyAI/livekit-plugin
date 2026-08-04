from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.plugins.prosodyai._transport import RealtimeTransport


class FakeWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages = iter(messages)
        self.sent: list[str | bytes] = []

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration from None

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self._websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self._websocket

    async def __aexit__(self, *_: object) -> None:
        return None


class ReceiveOnlyTransport(RealtimeTransport):
    async def _send_audio(self, websocket: Any, track: object) -> None:
        del websocket, track
        await asyncio.Event().wait()


class ClosableAudioStream:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_transport_delivers_session_end_before_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_end = {
        "type": "session_end",
        "session_id": "session-test",
        "transcript": {"turns": []},
    }
    websocket = FakeWebSocket([json.dumps(session_end)])

    def fake_connect(*_args: object, **_kwargs: object) -> FakeConnection:
        return FakeConnection(websocket)

    monkeypatch.setattr("websockets.asyncio.client.connect", fake_connect)
    transport = ReceiveOnlyTransport(
        api_key="test-key",
        base_url="https://api.prosodyai.app",
        session_id="session-test",
        sample_rate=16_000,
        source="livekit",
    )

    messages = [message async for message in transport.messages(object())]

    assert messages == [session_end]
    config = json.loads(str(websocket.sent[0]))
    assert config["api_key"] == "test-key"
    assert config["session_id"] == "session-test"


@pytest.mark.asyncio
async def test_audio_stream_closes_when_encoder_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_stream = ClosableAudioStream()

    def fake_audio_stream(*_args: object, **_kwargs: object) -> ClosableAudioStream:
        return audio_stream

    async def missing_ffmpeg(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("livekit.rtc.AudioStream", fake_audio_stream)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_ffmpeg)
    transport = RealtimeTransport(
        api_key="test-key",
        base_url="https://api.prosodyai.app",
        session_id="session-test",
        sample_rate=16_000,
        source="livekit",
    )

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        await transport._send_audio(FakeWebSocket([]), object())

    assert audio_stream.closed is True
