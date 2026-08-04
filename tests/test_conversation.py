from __future__ import annotations

from dataclasses import fields

import pytest

from livekit.plugins.prosodyai import (
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


def test_transcript_revision_removes_siblings_and_cannot_regress_a_final_result() -> None:
    conversation = Conversation()
    initial = {
        "type": "transcript_update",
        "result_id": "result-family",
        "is_final": False,
        "speech_final": False,
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 300,
                "speaker_id": "speaker_0",
                "text": "Stale first",
            },
            {
                "start_ms": 300,
                "end_ms": 600,
                "speaker_id": "speaker_0",
                "text": "Stale sibling",
            },
        ],
    }
    conversation.apply_message(initial)
    assert len(conversation.get_turns()) == 2

    revised = {
        **initial,
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 700,
                "speaker_id": "speaker_0",
                "text": "Only current revision",
            }
        ],
    }
    conversation.apply_message(revised)
    assert conversation.get_transcript() == "Only current revision"
    assert len(conversation.get_turns()) == 1

    final = {
        **revised,
        "is_final": True,
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 800,
                "speaker_id": "speaker_0",
                "text": "Committed revision",
            }
        ],
    }
    conversation.apply_message(final)
    conversation.apply_message(revised)

    assert conversation.get_transcript() == "Committed revision"
    assert conversation.get_turn(0).is_final is True


def test_final_only_means_is_final_without_requiring_speech_final() -> None:
    conversation = Conversation()
    update = transcript_update(final=True)
    update["speech_final"] = False

    conversation.apply_message(update)
    speech_endpoint = transcript_update(final=False)
    speech_endpoint["result_id"] = "result-2"
    speech_endpoint["speech_final"] = True
    speech_endpoint["segments"][0]["text"] = "Not final"
    conversation.apply_message(speech_endpoint)

    assert conversation.get_transcript(final_only=True) == "A complete turn"
    assert conversation.get_turns(final_only=True)[0].speech_final is False


def test_speaker_update_attributes_overlapping_unknown_turns_and_windows() -> None:
    conversation = Conversation()
    conversation.apply_message(directive(speaker_id="unknown"))
    conversation.apply_message(
        {
            "type": "transcript_update",
            "result_id": "unknown-result",
            "is_final": True,
            "speech_final": False,
            "segments": [
                {
                    "start_ms": 900,
                    "end_ms": 1_400,
                    "speaker_id": "unknown",
                    "text": "Attributed after diarization",
                }
            ],
        }
    )

    conversation.apply_message(
        {
            "type": "speaker_update",
            "start_ms": 1_000,
            "end_ms": 1_250,
            "speaker_id": "unknown",
            "dominant_speaker_id": "speaker_9",
            "speaker_merges": [],
        }
    )

    assert conversation.get_turn(0).speaker_id == "speaker_9"
    assert conversation.get_acoustic_window(0).speaker_id == "speaker_9"
    assert [speaker.speaker_id for speaker in conversation.get_speakers()] == ["speaker_9"]

    # Interval attribution must move the replay index to the resolved speaker.
    conversation.apply_message(directive(speaker_id="speaker_9", level=-15.0))
    assert len(conversation.get_acoustics()) == 1
    assert conversation.get_acoustic_window(0).get_feature("rms_dbfs") == -15.0


def test_speaker_cluster_update_applies_transitive_merges_to_stored_state() -> None:
    conversation = Conversation()
    conversation.apply_message(directive(speaker_id="speaker_0"))
    conversation.apply_message(directive(timestamp_ms=2_000, speaker_id="speaker_1", level=-19.0))
    conversation.apply_message(transcript_update(final=True))

    conversation.apply_message(
        {
            "type": "speaker_cluster_update",
            "speaker_merges": [
                {
                    "source_speaker_id": "speaker_0",
                    "target_speaker_id": "speaker_1",
                },
                {
                    "source_speaker_id": "speaker_1",
                    "target_speaker_id": "speaker_2",
                },
            ],
        }
    )

    assert {turn.speaker_id for turn in conversation.get_turns()} == {"speaker_2"}
    assert {window.speaker_id for window in conversation.get_acoustics()} == {"speaker_2"}
    assert [speaker.speaker_id for speaker in conversation.get_speakers()] == ["speaker_2"]

    # Relabeling must rebuild indexes used by later directive revisions.
    conversation.apply_message(directive(speaker_id="speaker_2", level=-16.0))
    assert len(conversation.get_acoustics()) == 2
    assert conversation.get_acoustic_window(0).get_feature("rms_dbfs") == -16.0


def test_session_end_replaces_turns_and_uses_terminal_timeline_as_fallback() -> None:
    conversation = Conversation()
    conversation.apply_message(transcript_update(final=False))

    conversation.apply_message(
        {
            "type": "session_end",
            "session_id": "session-test",
            "prosody_timeline": [],
            "transcript": {
                "turns": [
                    {
                        "start_ms": 0,
                        "end_ms": 700,
                        "speaker_id": "speaker_0",
                        "text": "Terminal first turn",
                    },
                    {
                        "start_ms": 800,
                        "end_ms": 1_500,
                        "speaker": "speaker_1",
                        "text": "Terminal second turn",
                    },
                ],
                "prosody_timeline": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "speaker_id": "speaker_0",
                        "acoustic_state": {
                            "values": {"rms_dbfs": -17.5},
                            "masks": {},
                        },
                        "acoustic_change": {
                            "values": {"rms_db_change": 2.0},
                            "reference": "previous_chunk_same_speaker",
                        },
                    }
                ],
            },
        }
    )

    assert conversation.get_transcript() == "Terminal first turn Terminal second turn"
    assert all(turn.is_final and turn.speech_final for turn in conversation.get_turns())
    assert [turn.result_id for turn in conversation.get_turns()] == [
        "session_end:0",
        "session_end:1",
    ]
    assert len(conversation.get_acoustics()) == 1
    window = conversation.get_acoustic_window(0)
    assert window.speaker_id == "speaker_0"
    assert window.end_ms == 1_000
    assert window.get_feature("rms_dbfs") == -17.5
    assert window.get_delta("rms_db_change") == 2.0


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
