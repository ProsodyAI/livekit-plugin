"""Developer-facing LiveKit analyzer."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from ._transport import RealtimeTransport
from .conversation import Conversation
from .models import AcousticEvent

DEFAULT_BASE_URL = "https://api.prosodyai.app"


class _MessageTransport(Protocol):
    def messages(self, track: object) -> AsyncIterator[Mapping[str, object]]: ...


class ProsodyAnalyzer:
    """Analyze a LiveKit track and maintain its :class:`Conversation`."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        session_id: str | None = None,
        sample_rate: int = 16_000,
        source: str = "livekit",
        max_reconnects: int = 3,
        _transport: _MessageTransport | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if max_reconnects < 1:
            raise ValueError("max_reconnects must be at least 1")
        self._api_key = api_key or os.environ.get("PROSODY_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._session_id = session_id or f"livekit-{uuid.uuid4().hex}"
        self._sample_rate = sample_rate
        self._source = source
        self._max_reconnects = max_reconnects
        self._transport = _transport
        self.conversation = Conversation(session_id=self._session_id)

    def __repr__(self) -> str:
        key_state = "<redacted>" if self._api_key else "<missing>"
        return (
            f"ProsodyAnalyzer(api_key={key_state}, base_url={self._base_url!r}, "
            f"session_id={self._session_id!r}, sample_rate={self._sample_rate})"
        )

    def apply_message(self, message: Mapping[str, object]) -> AcousticEvent | None:
        """Apply a decoded API message to this analyzer's conversation."""

        return self.conversation.apply_message(message)

    async def analyze_track(self, track: object) -> AsyncIterator[AcousticEvent]:
        """Analyze one LiveKit audio track and yield measured acoustic windows."""

        if not self._api_key:
            raise RuntimeError("PROSODY_API_KEY is required")

        for attempt in range(self._max_reconnects):
            transport = self._transport or self._make_transport()
            try:
                async for message in transport.messages(track):
                    event = self.apply_message(message)
                    if event is not None:
                        yield event
                return
            except (ConnectionError, OSError):
                if attempt + 1 >= self._max_reconnects:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))

    def _make_transport(self) -> RealtimeTransport:
        return RealtimeTransport(
            api_key=self._api_key,
            base_url=self._base_url,
            session_id=self._session_id,
            sample_rate=self._sample_rate,
            source=self._source,
        )


__all__ = ["ProsodyAnalyzer"]
