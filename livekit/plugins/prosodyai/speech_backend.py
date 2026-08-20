"""The speech-backend protocol the full-duplex bridge runs against.

The bridge owns everything ProsodyAI owns on the duplex path: the room-clock
resampling, the uplink hold for the session handshake, and the ProsodySSM
readout plumbing (identity lanes, transcripts, committed model events, state
resume). This module names only the speaking loop: open a session, stream
audio in, stream audio and text out, close. PersonaPlex over the gateway
socket (``personaplex.PersonaPlexBackend``) is the first-party implementation
and the bridge's default; an OpenAI Realtime-style, Gemini Live-style, or
Moshi-style loop plugs in by implementing :class:`SpeechBackend`.

Capabilities are declared facts. A turn-based (half-duplex) loop declares
``full_duplex=False`` and the declaration travels as-is: the bridge contains
no turn detector and synthesizes nothing on a backend's behalf.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "BackendCapabilityError",
    "SessionOpened",
    "SpeechAudio",
    "SpeechBackend",
    "SpeechBackendCapabilities",
    "SpeechItem",
    "SpeechSessionConfig",
    "SpeechText",
    "require_capabilities",
]


class BackendCapabilityError(RuntimeError):
    """A session config asked for a channel the backend's declaration excludes."""


@dataclass(frozen=True)
class SpeechBackendCapabilities:
    """What one speech backend declares about itself.

    ``full_duplex`` states whether the loop listens while it speaks; a
    turn-based backend declares ``False`` here and consumers read the fact.
    ``accepts_voice_prompt`` and ``accepts_role_prompt`` state whether the
    session open carries those prompts to the loop. ``sample_rate`` is the
    backend's native mono PCM rate; the bridge resamples to and from it.
    """

    full_duplex: bool
    accepts_voice_prompt: bool
    accepts_role_prompt: bool
    sample_rate: int


@dataclass(frozen=True)
class SpeechSessionConfig:
    """What the bridge asks of one speaking-loop session at open.

    ``voice_prompt`` names or seeds the voice the loop speaks with;
    ``role_prompt`` sets its role. Each is honored only by a backend whose
    capabilities declare the channel; :func:`require_capabilities` enforces
    the match before a session opens.
    """

    voice_prompt: str | None = None
    role_prompt: str | None = None


def require_capabilities(
    capabilities: SpeechBackendCapabilities, config: SpeechSessionConfig
) -> None:
    """Raise :class:`BackendCapabilityError` when the session config asks for
    a prompt channel the backend's declared capabilities exclude."""
    if config.voice_prompt is not None and not capabilities.accepts_voice_prompt:
        raise BackendCapabilityError(
            "the session config carries a voice prompt and the backend "
            "declares no voice prompt channel"
        )
    if config.role_prompt is not None and not capabilities.accepts_role_prompt:
        raise BackendCapabilityError(
            "the session config carries a role prompt and the backend "
            "declares no role prompt channel"
        )


@dataclass(frozen=True)
class SessionOpened:
    """The backend bound its session: audio may flow."""


@dataclass(frozen=True)
class SpeechAudio:
    """One block of the loop's spoken audio: mono float32 at the backend's
    declared ``sample_rate``."""

    samples: np.ndarray


@dataclass(frozen=True)
class SpeechText:
    """One text span of the loop's speech, as it is spoken."""

    text: str


SpeechItem = SessionOpened | SpeechAudio | SpeechText
"""One downlink item from the speaking loop, in the order the loop produced it."""


class SpeechBackend(Protocol):
    """One speech-to-speech speaking loop, as the bridge drives it.

    ``open`` binds a session, ``send_audio`` streams caller audio into the
    loop, ``receive`` yields the loop's downlink in production order, and
    ``close`` releases the session. Audio at this boundary is mono float32
    at ``capabilities.sample_rate`` in blocks of any length; framing and
    codec are the implementation's business.
    """

    @property
    def capabilities(self) -> SpeechBackendCapabilities: ...

    async def open(self, config: SpeechSessionConfig) -> None: ...

    async def send_audio(self, samples: np.ndarray) -> None: ...

    def receive(self) -> AsyncIterator[SpeechItem]: ...

    async def close(self) -> None: ...
