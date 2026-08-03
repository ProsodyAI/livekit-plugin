from __future__ import annotations

from dataclasses import fields

import pytest

from livekit_plugins_prosodyai import (
    AcousticEvent,
    Conversation,
    Speaker,
    TranscriptTurn,
)


def directive(
    *,
    timestamp_ms: int = 1_000,
    speaker_id: str = "speaker_0",
    level: float = -21.5,
    change: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "directive",
        "session_id": "session-test",
        "timestamp_ms": timestamp_ms,
        "speaker_id": speaker_id,
        "acoustic_state": {
            "values": {
                "rms_dbfs": level,
                "peak_dbfs": -8.0,
                "f0_median_hz": 184.0,
                "voiced_ratio": 0.72,
            },
            "masks": {
                "f0_available": True,
                "voiced_mask": [True, True, False],
            },
            "frames": {
                "frame_rate_hz": 12.5,
                "rms_dbfs": [-22.0, -21.0, -21.5],
                "f0_hz": [180.0, 188.0, None],
                "spectral_tilt_db_per_octave": [-7.0, -6.8, None],
                "voiced_probability": [0.95, 0.92, 0.08],
                "voice_activity_boundary_probability": [0.02, 0.04, 0.83],
            },
            "provenance": {
                "kind": "trained_head",
                "feature_version": "acoustic-v1",
            },
        },
    }
    if change is not None:
        payload["acoustic_change"] = {
            "values": {
                "rms_db_change": change,
                "f0_median_semitone_change": 0.7,
            },
            "reference": "previous_chunk_same_speaker",
            "provenance": {"kind": "trained_head"},
        }
    return payload


def transcript_update(*, final: bool) -> dict[str, object]:
    return {
        "type": "transcript_update",
        "session_id": "session-test",
        "result_id": "result-1",
        "is_final": final,
        "speech_final": final,
        "segments": [
            {
                "start_ms": 100,
                "end_ms": 900 if final else 700,
                "speaker_id": "speaker_1",
                "text": "A complete turn" if final else "A partial",
            }
        ],
    }


def test_conversation_exposes_state_change_and_frame_timing() -> None:
    conversation = Conversation()

    first = conversation.apply_message(directive())
    second = conversation.apply_message(directive(timestamp_ms=2_000, level=-18.0, change=3.5))

    assert isinstance(first, AcousticEvent)
    assert isinstance(second, AcousticEvent)
    assert conversation.session_id == "session-test"
    assert first.window.get_feature("rms_dbfs") == -21.5
    assert first.window.get_delta("rms_db_change") is None
    assert second.window.get_delta("rms_db_change") == 3.5

    frames = first.window.get_frame_series("f0_hz")
    assert [point.timestamp_ms for point in frames] == [1_000, 1_080, 1_160]
    assert [point.value for point in frames] == [180.0, 188.0, None]
    assert first.window.end_ms == 1_240

    series = conversation.get_feature_series("rms_dbfs", "speaker_0")
    assert [point.value for point in series] == [-21.5, -18.0]
    assert len(conversation.get_frame_series("voiced_probability")) == 6

    deltas = conversation.get_deltas("speaker_0")
    assert len(deltas) == 1
    assert deltas[0].reference == "previous_chunk_same_speaker"
    assert deltas[0].get_delta("rms_db_change") == 3.5


def test_transcript_revision_replaces_the_interim_turn() -> None:
    conversation = Conversation()

    assert conversation.apply_message(transcript_update(final=False)) is None
    assert conversation.get_transcript() == "A partial"
    assert conversation.get_turn(0).is_final is False

    conversation.apply_message(transcript_update(final=True))

    turns = conversation.get_turns(final_only=True)
    assert len(turns) == 1
    assert turns[0].text == "A complete turn"
    assert turns[0].end_ms == 900


def test_speakers_are_recording_local_activity_summaries() -> None:
    conversation = Conversation()
    conversation.apply_message(directive(speaker_id="speaker_0"))
    conversation.apply_message(transcript_update(final=True))

    speakers = conversation.get_speakers()
    assert speakers == (
        Speaker(
            speaker_id="speaker_0",
            talk_ms=0,
            turn_count=0,
            window_count=1,
        ),
        Speaker(
            speaker_id="speaker_1",
            talk_ms=800,
            turn_count=1,
            window_count=0,
        ),
    )


def test_replayed_directive_replaces_the_same_window() -> None:
    conversation = Conversation()
    conversation.apply_message(directive(level=-20.0))
    conversation.apply_message(directive(level=-17.0))

    assert len(conversation.get_acoustics()) == 1
    assert conversation.get_acoustic_window(0).get_feature("rms_dbfs") == -17.0


def test_unknown_features_fail_loudly() -> None:
    conversation = Conversation()
    event = conversation.apply_message(directive())
    assert event is not None

    with pytest.raises(KeyError, match="unknown acoustic feature"):
        event.window.get_feature("not_a_feature")
    with pytest.raises(KeyError, match="unknown frame feature"):
        event.window.get_frame_series("not_a_frame")


def test_public_models_have_no_durable_identity_field() -> None:
    for model in (TranscriptTurn, Speaker):
        assert all("person" not in field.name for field in fields(model))
