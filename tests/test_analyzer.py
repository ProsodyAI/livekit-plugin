from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import pytest
from livekit.agents import Plugin

from livekit.plugins import prosodyai

from .test_conversation import directive, transcript_update


class FakeTransport:
    def __init__(self, messages: list[Mapping[str, object]]) -> None:
        self._messages = messages

    async def messages(self, track: object) -> AsyncIterator[Mapping[str, object]]:
        del track
        for message in self._messages:
            yield message


class FailingTransport:
    async def messages(self, track: object) -> AsyncIterator[Mapping[str, object]]:
        if track is self:
            yield {}
        raise ConnectionError("stream disconnected")


def test_repr_redacts_api_key() -> None:
    secret = "test-key-that-must-not-appear"
    analyzer = prosodyai.ProsodyAnalyzer(api_key=secret, session_id="session-safe")

    rendered = repr(analyzer)
    assert secret not in rendered
    assert "<redacted>" in rendered


def test_analyzer_owns_and_updates_a_conversation() -> None:
    analyzer = prosodyai.ProsodyAnalyzer(api_key="test-key", session_id="session-test")

    analyzer.apply_message(transcript_update(final=True))
    event = analyzer.apply_message(directive())

    assert event is not None
    assert analyzer.conversation.get_transcript(final_only=True) == "A complete turn"
    assert len(analyzer.conversation.get_acoustics()) == 1


@pytest.mark.asyncio
async def test_analyze_track_applies_all_messages_and_yields_only_acoustics() -> None:
    transport = FakeTransport(
        [
            transcript_update(final=True),
            directive(),
            {"type": "warning", "code": "synthetic_warning"},
        ]
    )
    analyzer = prosodyai.ProsodyAnalyzer(
        api_key="test-key",
        session_id="session-test",
        _transport=transport,
    )

    events = [event async for event in analyzer.analyze_track(object())]

    assert len(events) == 1
    assert events[0].window.get_feature("rms_dbfs") == -21.5
    assert analyzer.conversation.get_turns(final_only=True)[0].speaker_id == "speaker_1"


@pytest.mark.asyncio
async def test_analyze_track_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROSODY_API_KEY", raising=False)
    analyzer = prosodyai.ProsodyAnalyzer(_transport=FakeTransport([]))

    with pytest.raises(RuntimeError, match="PROSODY_API_KEY is required"):
        async for _event in analyzer.analyze_track(object()):
            pass


@pytest.mark.asyncio
async def test_analyze_track_surfaces_transport_failure() -> None:
    analyzer = prosodyai.ProsodyAnalyzer(
        api_key="test-key",
        _transport=FailingTransport(),
    )

    with pytest.raises(ConnectionError, match="stream disconnected"):
        async for _event in analyzer.analyze_track(object()):
            pass


@pytest.mark.asyncio
async def test_analyze_track_applies_the_terminal_session_result() -> None:
    transport = FakeTransport(
        [
            transcript_update(final=False),
            {
                "type": "session_end",
                "session_id": "session-test",
                "transcript": {
                    "turns": [
                        {
                            "start_ms": 100,
                            "end_ms": 900,
                            "speaker_id": "speaker_4",
                            "text": "Authoritative final turn",
                        }
                    ]
                },
            },
        ]
    )
    analyzer = prosodyai.ProsodyAnalyzer(
        api_key="test-key",
        session_id="session-test",
        _transport=transport,
    )

    events = [event async for event in analyzer.analyze_track(object())]

    assert events == []
    assert analyzer.conversation.get_transcript(final_only=True) == "Authoritative final turn"
    assert analyzer.conversation.get_turn(0).speaker_id == "speaker_4"


def test_import_registers_the_livekit_plugin() -> None:
    registered = [
        plugin
        for plugin in Plugin.registered_plugins
        if plugin.package == "livekit.plugins.prosodyai"
    ]
    assert len(registered) == 1
    assert registered[0].title == "livekit.plugins.prosodyai"
    assert registered[0].version == prosodyai.__version__
