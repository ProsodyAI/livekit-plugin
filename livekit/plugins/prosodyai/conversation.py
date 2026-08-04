"""Ordered conversation state built from ProsodyAI analysis messages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
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
        self._result_turn_ids: dict[str, set[str]] = {}
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
        elif message_type == "speaker_update":
            self._apply_speaker_update(payload)
        elif message_type == "speaker_cluster_update":
            self._apply_speaker_merges(payload.get("speaker_merges"))
        elif message_type == "session_end":
            self._apply_session_end(payload)
        return None

    def get_transcript(self, *, final_only: bool = False) -> str:
        return " ".join(turn.text for turn in self.get_turns(final_only=final_only) if turn.text)

    def get_turns(self, *, final_only: bool = False) -> tuple[TranscriptTurn, ...]:
        turns = sorted(
            self._turns.values(),
            key=lambda turn: (turn.start_ms, turn.end_ms, turn.result_id),
        )
        if final_only:
            turns = [turn for turn in turns if turn.is_final]
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
        speaker_id = self._speaker_id(payload.get("speaker_id"))
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

        prior_result_ids: set[str] = set()
        if event_result_id:
            prior_result_ids = self._result_turn_ids.get(event_result_id, set())
            if not prior_result_ids:
                prior_result_ids = {
                    result_id
                    for result_id in self._turns
                    if result_id == event_result_id or result_id.startswith(f"{event_result_id}:")
                }
            if not event_is_final and any(
                self._turns[result_id].is_final
                for result_id in prior_result_ids
                if result_id in self._turns
            ):
                return
            for result_id in prior_result_ids:
                self._turns.pop(result_id, None)

        next_result_ids: set[str] = set()

        for index, raw_segment in enumerate(segments):
            segment = _as_mapping(raw_segment)
            if not segment:
                continue
            start_ms = _integer(segment.get("start_ms"), _integer(payload.get("start_ms")))
            end_ms = _integer(segment.get("end_ms"), _integer(payload.get("end_ms")))
            speaker_id = self._speaker_id(segment.get("speaker_id"))
            segment_result_id = _string(segment.get("result_id"))
            result_id = segment_result_id or event_result_id
            if result_id and len(segments) > 1 and not segment_result_id:
                result_id = f"{result_id}:{index}"
            if not result_id:
                result_id = f"{speaker_id}:{start_ms}:{end_ms}"
            if result_id in next_result_ids:
                result_id = f"{result_id}:{index}"

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
            next_result_ids.add(result_id)

        if event_result_id:
            self._result_turn_ids[event_result_id] = next_result_ids
        self._refresh_speaker_order()

    def _apply_speaker_update(self, payload: Mapping[str, object]) -> None:
        start_ms = _integer(payload.get("start_ms"))
        end_ms = max(start_ms + 1, _integer(payload.get("end_ms"), start_ms))
        speaker_id = self._speaker_id(payload.get("speaker_id"))
        if speaker_id == "unknown":
            speaker_id = self._speaker_id(payload.get("dominant_speaker_id"))

        if speaker_id != "unknown":
            for result_id, turn in tuple(self._turns.items()):
                if turn.speaker_id != "unknown":
                    continue
                turn_end_ms = max(turn.start_ms + 1, turn.end_ms)
                overlap_ms = self._overlap_ms(
                    turn.start_ms,
                    turn_end_ms,
                    start_ms,
                    end_ms,
                )
                duration_ms = max(1, turn_end_ms - turn.start_ms)
                if overlap_ms >= duration_ms * 0.25 or overlap_ms >= 200:
                    self._turns[result_id] = replace(turn, speaker_id=speaker_id)

            windows_changed = False
            for index, window in enumerate(self._windows):
                if window.speaker_id != "unknown":
                    continue
                if (
                    self._overlap_ms(
                        window.start_ms,
                        max(window.start_ms + 1, window.end_ms),
                        start_ms,
                        end_ms,
                    )
                    <= 0
                ):
                    continue
                self._windows[index] = replace(window, speaker_id=speaker_id)
                windows_changed = True
            if windows_changed:
                self._rebuild_window_indexes()
            self._refresh_speaker_order()

        self._apply_speaker_merges(payload.get("speaker_merges"))

    def _apply_speaker_merges(self, raw_merges: object) -> None:
        if not isinstance(raw_merges, list):
            return

        merges: dict[str, str] = {}
        for raw_merge in cast(list[object], raw_merges):
            merge = _as_mapping(raw_merge)
            source = self._speaker_id(merge.get("source_speaker_id"))
            target = self._speaker_id(merge.get("target_speaker_id"))
            if source == "unknown" or target == "unknown" or source == target:
                continue
            merges[source] = target
        if not merges:
            return

        self._turns = {
            result_id: replace(
                turn,
                speaker_id=self._speaker_after_merges(turn.speaker_id, merges),
            )
            for result_id, turn in self._turns.items()
        }
        self._windows = [
            replace(
                window,
                speaker_id=self._speaker_after_merges(window.speaker_id, merges),
            )
            for window in self._windows
        ]
        self._rebuild_window_indexes()
        self._refresh_speaker_order(merges)

    def _apply_session_end(self, payload: Mapping[str, object]) -> None:
        transcript = _as_mapping(payload.get("transcript"))
        raw_turns = transcript.get("turns")
        if isinstance(raw_turns, list) and raw_turns:
            terminal_turns: dict[str, TranscriptTurn] = {}
            for index, raw_turn in enumerate(cast(list[object], raw_turns)):
                turn = _as_mapping(raw_turn)
                if not turn:
                    continue
                start_ms = _integer(turn.get("start_ms"))
                end_ms = _integer(turn.get("end_ms"), start_ms)
                speaker_id = self._speaker_id(turn.get("speaker_id", turn.get("speaker")))
                result_id = _string(turn.get("result_id")).strip()
                if not result_id:
                    result_id = f"session_end:{index}"
                elif result_id in terminal_turns:
                    result_id = f"{result_id}:{index}"
                terminal_turns[result_id] = TranscriptTurn(
                    result_id=result_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_id=speaker_id,
                    text=_string(turn.get("text")).strip(),
                    is_final=True,
                    speech_final=True,
                )
            self._turns = terminal_turns
            self._result_turn_ids = {}

        if not self._windows:
            raw_timeline = payload.get("prosody_timeline")
            if not isinstance(raw_timeline, list) or not raw_timeline:
                raw_timeline = transcript.get("prosody_timeline")
            if isinstance(raw_timeline, list):
                for raw_point in cast(list[object], raw_timeline):
                    point = _as_mapping(raw_point)
                    state_payload = point.get("acoustic_state")
                    if not isinstance(state_payload, dict):
                        continue
                    state = AcousticState.from_payload(state_payload)
                    change_payload = point.get("acoustic_change")
                    change = (
                        AcousticChange.from_payload(change_payload)
                        if isinstance(change_payload, dict)
                        else None
                    )
                    start_ms = _integer(point.get("start_ms"))
                    window = AcousticWindow(
                        start_ms=start_ms,
                        end_ms=self._window_end_ms(point, start_ms, state),
                        speaker_id=self._speaker_id(point.get("speaker_id")),
                        state=state,
                        change=change,
                    )
                    key = (window.speaker_id, window.start_ms)
                    existing_index = self._window_indexes.get(key)
                    if existing_index is None:
                        self._window_indexes[key] = len(self._windows)
                        self._windows.append(window)
                    else:
                        self._windows[existing_index] = window

        self._refresh_speaker_order()

    def _rebuild_window_indexes(self) -> None:
        windows: list[AcousticWindow] = []
        indexes: dict[tuple[str, int], int] = {}
        for window in self._windows:
            key = (window.speaker_id, window.start_ms)
            existing_index = indexes.get(key)
            if existing_index is None:
                indexes[key] = len(windows)
                windows.append(window)
            else:
                windows[existing_index] = window
        self._windows = windows
        self._window_indexes = indexes

    def _refresh_speaker_order(self, merges: Mapping[str, str] | None = None) -> None:
        active_ids = {
            *(turn.speaker_id for turn in self._turns.values()),
            *(window.speaker_id for window in self._windows),
        }
        ordered_ids: list[str] = []
        for speaker_id, _ in sorted(self._speaker_order.items(), key=lambda item: item[1]):
            resolved = (
                self._speaker_after_merges(speaker_id, merges) if merges is not None else speaker_id
            )
            if resolved in active_ids and resolved not in ordered_ids:
                ordered_ids.append(resolved)
        for turn in self.get_turns():
            if turn.speaker_id not in ordered_ids:
                ordered_ids.append(turn.speaker_id)
        for window in self.get_acoustics():
            if window.speaker_id not in ordered_ids:
                ordered_ids.append(window.speaker_id)
        self._speaker_order = {speaker_id: index for index, speaker_id in enumerate(ordered_ids)}

    @classmethod
    def _speaker_after_merges(
        cls,
        speaker_id: str,
        merges: Mapping[str, str],
    ) -> str:
        current = cls._speaker_id(speaker_id)
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            target = merges.get(current)
            if target is None:
                break
            current = cls._speaker_id(target)
        return current

    @staticmethod
    def _speaker_id(value: object) -> str:
        speaker_id = _string(value).strip()
        return speaker_id or "unknown"

    @staticmethod
    def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
        return max(0, min(end_a, end_b) - max(start_a, start_b))

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
