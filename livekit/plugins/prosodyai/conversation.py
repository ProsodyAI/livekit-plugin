"""Ordered conversation state built from ProsodyAI analysis messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .models import (
    AcousticChange,
    AcousticDelta,
    AcousticEvent,
    AcousticState,
    AcousticWindow,
    FeaturePoint,
    FramePoint,
    Speaker,
    TranscriptTurn,
    _as_mapping,
    _boolean,
    _integer,
    _optional_float,
    _string,
)


class Conversation:
    """Transcript and acoustic state for one analyzed recording or call."""

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or ""
        self._windows: list[AcousticWindow] = []
        self._window_indexes: dict[tuple[str, int], int] = {}
        self._turns: dict[str, TranscriptTurn] = {}
        self._speaker_order: dict[str, int] = {}

    def __repr__(self) -> str:
        return (
            f"Conversation(session_id={self._session_id!r}, "
            f"turns={len(self._turns)}, windows={len(self._windows)})"
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    def apply_message(self, message: Mapping[str, object]) -> AcousticEvent | None:
        """Apply one decoded API message and return a new acoustic event, if any."""

        payload = dict(message)
        session_id = payload.get("session_id")
        if not self._session_id and isinstance(session_id, str):
            self._session_id = session_id

        message_type = payload.get("type")
        if message_type == "directive":
            return self._apply_directive(payload)
        if message_type == "transcript_update":
            self._apply_transcript_update(payload)
        return None

    def get_transcript(self, *, final_only: bool = False) -> str:
        return " ".join(turn.text for turn in self.get_turns(final_only=final_only) if turn.text)

    def get_turns(self, *, final_only: bool = False) -> tuple[TranscriptTurn, ...]:
        turns = sorted(
            self._turns.values(),
            key=lambda turn: (turn.start_ms, turn.end_ms, turn.result_id),
        )
        if final_only:
            turns = [turn for turn in turns if turn.is_final and turn.speech_final]
        return tuple(turns)

    def get_turn(self, index: int) -> TranscriptTurn:
        return self.get_turns()[index]

    def get_speakers(self) -> tuple[Speaker, ...]:
        speaker_ids = set(self._speaker_order)
        turns = self.get_turns()
        windows = self.get_acoustics()
        summaries = []
        for speaker_id in speaker_ids:
            speaker_turns = [turn for turn in turns if turn.speaker_id == speaker_id]
            speaker_windows = [window for window in windows if window.speaker_id == speaker_id]
            summaries.append(
                Speaker(
                    speaker_id=speaker_id,
                    talk_ms=sum(max(0, turn.end_ms - turn.start_ms) for turn in speaker_turns),
                    turn_count=len(speaker_turns),
                    window_count=len(speaker_windows),
                )
            )
        summaries.sort(key=lambda item: self._speaker_order[item.speaker_id])
        return tuple(summaries)

    def get_acoustics(self, speaker_id: str | None = None) -> tuple[AcousticWindow, ...]:
        windows = sorted(
            self._windows,
            key=lambda window: (window.start_ms, window.end_ms, window.speaker_id),
        )
        if speaker_id is not None:
            windows = [window for window in windows if window.speaker_id == speaker_id]
        return tuple(windows)

    def get_acoustic_window(self, index: int) -> AcousticWindow:
        return self.get_acoustics()[index]

    def get_feature_series(
        self,
        feature: str,
        speaker_id: str | None = None,
    ) -> tuple[FeaturePoint, ...]:
        return tuple(
            FeaturePoint(
                timestamp_ms=window.start_ms,
                speaker_id=window.speaker_id,
                value=window.get_feature(feature),
            )
            for window in self.get_acoustics(speaker_id)
        )

    def get_frame_series(
        self,
        feature: str,
        speaker_id: str | None = None,
    ) -> tuple[FramePoint, ...]:
        return tuple(
            point
            for window in self.get_acoustics(speaker_id)
            for point in window.get_frame_series(feature)
        )

    def get_deltas(self, speaker_id: str | None = None) -> tuple[AcousticDelta, ...]:
        return tuple(
            AcousticDelta(
                start_ms=window.start_ms,
                end_ms=window.end_ms,
                speaker_id=window.speaker_id,
                change=window.change,
            )
            for window in self.get_acoustics(speaker_id)
            if window.change is not None
        )

    def _apply_directive(self, payload: Mapping[str, object]) -> AcousticEvent | None:
        state_payload = payload.get("acoustic_state")
        if not isinstance(state_payload, dict):
            return None

        state = AcousticState.from_payload(state_payload)
        change_payload = payload.get("acoustic_change")
        change = (
            AcousticChange.from_payload(change_payload)
            if isinstance(change_payload, dict)
            else None
        )
        start_ms = _integer(payload.get("timestamp_ms"))
        speaker_id = _string(payload.get("speaker_id"), "unknown") or "unknown"
        end_ms = self._window_end_ms(payload, start_ms, state)
        window = AcousticWindow(
            start_ms=start_ms,
            end_ms=end_ms,
            speaker_id=speaker_id,
            state=state,
            change=change,
        )
        key = (speaker_id, start_ms)
        existing_index = self._window_indexes.get(key)
        if existing_index is None:
            self._window_indexes[key] = len(self._windows)
            self._windows.append(window)
        else:
            self._windows[existing_index] = window
        self._note_speaker(speaker_id)
        return AcousticEvent(window=window)

    def _apply_transcript_update(self, payload: Mapping[str, object]) -> None:
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            return
        segments = cast(list[object], raw_segments)
        event_result_id = _string(payload.get("result_id"))
        event_is_final = _boolean(payload.get("is_final"))
        event_speech_final = _boolean(payload.get("speech_final"))

        for index, raw_segment in enumerate(segments):
            segment = _as_mapping(raw_segment)
            if not segment:
                continue
            start_ms = _integer(segment.get("start_ms"), _integer(payload.get("start_ms")))
            end_ms = _integer(segment.get("end_ms"), _integer(payload.get("end_ms")))
            speaker_id = _string(segment.get("speaker_id"), "unknown") or "unknown"
            segment_result_id = _string(segment.get("result_id"))
            result_id = segment_result_id or event_result_id
            if result_id and len(segments) > 1 and not segment_result_id:
                result_id = f"{result_id}:{index}"
            if not result_id:
                result_id = f"{speaker_id}:{start_ms}:{end_ms}"

            is_final = (
                _boolean(segment.get("is_final"))
                if isinstance(segment.get("is_final"), bool)
                else event_is_final
            )
            speech_final = (
                _boolean(segment.get("speech_final"))
                if isinstance(segment.get("speech_final"), bool)
                else event_speech_final
            )
            self._turns[result_id] = TranscriptTurn(
                result_id=result_id,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id=speaker_id,
                text=_string(segment.get("text")).strip(),
                is_final=is_final,
                speech_final=speech_final,
            )
            self._note_speaker(speaker_id)

    @staticmethod
    def _window_end_ms(
        payload: Mapping[str, object],
        start_ms: int,
        state: AcousticState,
    ) -> int:
        explicit_end = _integer(payload.get("end_ms"), -1)
        if explicit_end >= start_ms:
            return explicit_end

        duration_ms = _optional_float(payload.get("duration_ms"))
        if duration_ms is None:
            duration_seconds = _optional_float(payload.get("duration_sec"))
            if duration_seconds is not None:
                duration_ms = duration_seconds * 1000

        frames = state.frames
        if duration_ms is None and frames is not None and frames.frame_count:
            duration_ms = frames.frame_count * 1000 / frames.frame_rate_hz
        if duration_ms is None:
            duration_ms = 1000
        return start_ms + max(0, round(duration_ms))

    def _note_speaker(self, speaker_id: str) -> None:
        if speaker_id not in self._speaker_order:
            self._speaker_order[speaker_id] = len(self._speaker_order)


__all__ = ["Conversation"]
