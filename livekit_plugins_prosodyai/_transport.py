"""Internal LiveKit audio capture and ProsodyAI streaming transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast


class _StreamDone:
    pass


_STREAM_DONE = _StreamDone()


def _stream_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base.removeprefix("https://")
    elif base.startswith("http://"):
        base = "ws://" + base.removeprefix("http://")
    if base.endswith("/v1/stream/realtime"):
        return base
    return base + "/v1/stream/realtime"


class RealtimeTransport:
    """Private transport used by :class:`ProsodyAnalyzer`."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        session_id: str,
        sample_rate: int,
        source: str,
    ) -> None:
        self._api_key = api_key
        self._url = _stream_url(base_url)
        self._session_id = session_id
        self._sample_rate = sample_rate
        self._source = source

    async def messages(self, track: object) -> AsyncIterator[Mapping[str, object]]:
        from websockets.asyncio.client import connect

        async with connect(
            self._url,
            ping_interval=20,
            max_size=8 * 1024 * 1024,
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "config",
                        "api_key": self._api_key,
                        "session_id": self._session_id,
                        "encoding": "opus",
                        "container": "ogg",
                        "source": self._source,
                        "source_offset_ms": 0,
                    },
                    separators=(",", ":"),
                )
            )
            queue: asyncio.Queue[Mapping[str, object] | Exception | _StreamDone] = asyncio.Queue()

            async def send_audio() -> None:
                try:
                    await self._send_audio(websocket, track)
                except Exception as exc:
                    await queue.put(exc)
                    await websocket.close()

            async def receive_messages() -> None:
                try:
                    async for raw_message in websocket:
                        if not isinstance(raw_message, str):
                            continue
                        decoded: object = json.loads(raw_message)
                        if not isinstance(decoded, dict):
                            continue
                        message = cast(dict[str, object], decoded)
                        if message.get("type") == "session_end":
                            return
                        if message.get("type") == "error":
                            detail = message.get("message")
                            safe_detail = detail if isinstance(detail, str) else "analysis failed"
                            await queue.put(RuntimeError(safe_detail))
                            return
                        await queue.put(message)
                except Exception as exc:
                    await queue.put(exc)
                finally:
                    await queue.put(_STREAM_DONE)

            sender = asyncio.create_task(
                send_audio(),
                name="prosodyai-livekit-audio",
            )
            receiver = asyncio.create_task(
                receive_messages(),
                name="prosodyai-analysis-events",
            )
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, _StreamDone):
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _send_audio(self, websocket: Any, track: object) -> None:
        from livekit import rtc

        process: asyncio.subprocess.Process | None = None
        stream = rtc.AudioStream(
            cast(Any, track),
            sample_rate=self._sample_rate,
            num_channels=1,
        )
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    str(self._sample_rate),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "libopus",
                    "-application",
                    "voip",
                    "-b:a",
                    "32k",
                    "-f",
                    "ogg",
                    "pipe:1",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg is required to analyze a LiveKit track") from exc

            if process.stdin is None or process.stdout is None:
                raise RuntimeError("ffmpeg did not expose its audio pipes")

            async def write_pcm() -> None:
                assert process is not None and process.stdin is not None
                async for frame_event in stream:
                    frame = cast(Any, frame_event).frame
                    raw = bytes(frame.data)
                    if raw:
                        process.stdin.write(raw)
                        await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

            async def send_encoded() -> None:
                assert process is not None and process.stdout is not None
                while True:
                    packet = await process.stdout.read(4096)
                    if not packet:
                        break
                    await websocket.send(packet)

            await asyncio.gather(write_pcm(), send_encoded())
            return_code = await process.wait()
            if return_code:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")
            await websocket.send(json.dumps({"type": "end"}))
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                await process.wait()


__all__ = ["RealtimeTransport"]
