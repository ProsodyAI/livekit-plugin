"""Typed acoustic and transcript values exposed by the public plugin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

DEFAULT_FRAME_RATE_HZ: Final = 12.5


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    raw = cast(dict[object, object], value)
    return {key: item for key, item in raw.items() if isinstance(key, str)}


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _string(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _boolean(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _numeric_mapping(value: object) -> Mapping[str, float | None]:
    mapped = _as_mapping(value)
    parsed = {key: _optional_float(item) for key, item in mapped.items()}
    return MappingProxyType(parsed)


def _string_mapping(value: object) -> Mapping[str, str]:
    mapped = _as_mapping(value)
    parsed = {key: item for key, item in mapped.items() if isinstance(item, str)}
    return MappingProxyType(parsed)


def _numeric_series(value: object) -> tuple[float | None, ...]:
    if not isinstance(value, list):
        return ()
    raw = cast(list[object], value)
    return tuple(_optional_float(item) for item in raw)


def _bool_series(value: object) -> tuple[bool, ...]:
    if not isinstance(value, list):
        return ()
    raw = cast(list[object], value)
    return tuple(item for item in raw if isinstance(item, bool))


@dataclass(frozen=True)
class AcousticFrames:
    """Frame-level acoustic trajectories for one analysis window."""

    frame_rate_hz: float
    rms_dbfs: tuple[float | None, ...]
    f0_hz: tuple[float | None, ...]
    spectral_tilt_db_per_octave: tuple[float | None, ...]
    voiced_probability: tuple[float | None, ...]
    voice_activity_boundary_probability: tuple[float | None, ...]

    @classmethod
    def from_payload(cls, payload: object) -> AcousticFrames:
        data = _as_mapping(payload)
        frame_rate_hz = _optional_float(data.get("frame_rate_hz"))
        if frame_rate_hz is None or frame_rate_hz <= 0:
            frame_rate_hz = DEFAULT_FRAME_RATE_HZ
        return cls(
            frame_rate_hz=frame_rate_hz,
            rms_dbfs=_numeric_series(data.get("rms_dbfs")),
            f0_hz=_numeric_series(data.get("f0_hz")),
            spectral_tilt_db_per_octave=_numeric_series(data.get("spectral_tilt_db_per_octave")),
            voiced_probability=_numeric_series(data.get("voiced_probability")),
            voice_activity_boundary_probability=_numeric_series(
                data.get("voice_activity_boundary_probability")
            ),
        )

    @property
    def frame_count(self) -> int:
        return max(
            (
                len(self.rms_dbfs),
                len(self.f0_hz),
                len(self.spectral_tilt_db_per_octave),
                len(self.voiced_probability),
                len(self.voice_activity_boundary_probability),
            )
        )

    def get_series(self, feature: str) -> tuple[float | None, ...]:
        series = {
            "rms_dbfs": self.rms_dbfs,
            "f0_hz": self.f0_hz,
            "spectral_tilt_db_per_octave": self.spectral_tilt_db_per_octave,
            "voiced_probability": self.voiced_probability,
            "voice_activity_boundary_probability": (self.voice_activity_boundary_probability),
        }
        try:
            return series[feature]
        except KeyError as exc:
            raise KeyError(f"unknown frame feature: {feature}") from exc


@dataclass(frozen=True)
class AcousticState:
    """Measured acoustic state for one analysis window."""

    values: Mapping[str, float | None]
    masks: Mapping[str, bool]
    voiced_mask: tuple[bool, ...]
    frames: AcousticFrames | None
    provenance: Mapping[str, str]

    @classmethod
    def from_payload(cls, payload: object) -> AcousticState:
        data = _as_mapping(payload)
        masks_data = _as_mapping(data.get("masks"))
        masks = {
            key: item
            for key, item in masks_data.items()
            if key != "voiced_mask" and isinstance(item, bool)
        }
        frames_payload = data.get("frames")
        frames = (
            AcousticFrames.from_payload(frames_payload)
            if isinstance(frames_payload, dict)
            else None
        )
        return cls(
            values=_numeric_mapping(data.get("values")),
            masks=MappingProxyType(masks),
            voiced_mask=_bool_series(masks_data.get("voiced_mask")),
            frames=frames,
            provenance=_string_mapping(data.get("provenance")),
        )

    def get_feature(self, feature: str) -> float | None:
        try:
            return self.values[feature]
        except KeyError as exc:
            raise KeyError(f"unknown acoustic feature: {feature}") from exc


@dataclass(frozen=True)
class AcousticChange:
    """Signed acoustic deltas against the model-declared reference window."""

    values: Mapping[str, float | None]
    reference: str | None
    provenance: Mapping[str, str]

    @classmethod
    def from_payload(cls, payload: object) -> AcousticChange:
        data = _as_mapping(payload)
        reference_value = data.get("reference")
        reference = reference_value if isinstance(reference_value, str) else None
        return cls(
            values=_numeric_mapping(data.get("values")),
            reference=reference,
            provenance=_string_mapping(data.get("provenance")),
        )

    def get_delta(self, feature: str) -> float | None:
        try:
            return self.values[feature]
        except KeyError as exc:
            raise KeyError(f"unknown acoustic delta: {feature}") from exc


@dataclass(frozen=True)
class FramePoint:
    """One timestamped value from a frame trajectory."""

    timestamp_ms: int
    speaker_id: str
    value: float | None


@dataclass(frozen=True)
class FeaturePoint:
    """One timestamped window-level acoustic value."""

    timestamp_ms: int
    speaker_id: str
    value: float | None


@dataclass(frozen=True)
class AcousticWindow:
    """Acoustic state and change aligned to one recording interval."""

    start_ms: int
    end_ms: int
    speaker_id: str
    state: AcousticState
    change: AcousticChange | None = None

    def get_feature(self, feature: str) -> float | None:
        return self.state.get_feature(feature)

    def get_delta(self, feature: str) -> float | None:
        if self.change is None:
            return None
        return self.change.get_delta(feature)

    def get_frame_series(self, feature: str) -> tuple[FramePoint, ...]:
        frames = self.state.frames
        if frames is None:
            return ()
        values = frames.get_series(feature)
        return tuple(
            FramePoint(
                timestamp_ms=self.start_ms + round(index * 1000 / frames.frame_rate_hz),
                speaker_id=self.speaker_id,
                value=value,
            )
            for index, value in enumerate(values)
        )


@dataclass(frozen=True)
class AcousticDelta:
    """A timestamped same-speaker change measurement."""

    start_ms: int
    end_ms: int
    speaker_id: str
    change: AcousticChange

    @property
    def reference(self) -> str | None:
        return self.change.reference

    @property
    def values(self) -> Mapping[str, float | None]:
        return self.change.values

    def get_delta(self, feature: str) -> float | None:
        return self.change.get_delta(feature)


@dataclass(frozen=True)
class TranscriptTurn:
    """The latest revision of one recording-local transcript turn."""

    result_id: str
    start_ms: int
    end_ms: int
    speaker_id: str
    text: str
    is_final: bool
    speech_final: bool


@dataclass(frozen=True)
class Speaker:
    """Recording-local speaker activity summarized from turns and windows."""

    speaker_id: str
    talk_ms: int
    turn_count: int
    window_count: int


@dataclass(frozen=True)
class AcousticEvent:
    """One newly applied acoustic window."""

    window: AcousticWindow

    @property
    def speaker_id(self) -> str:
        return self.window.speaker_id

    @property
    def acoustic_state(self) -> AcousticState:
        return self.window.state

    @property
    def acoustic_change(self) -> AcousticChange | None:
        return self.window.change


__all__ = [
    "AcousticChange",
    "AcousticDelta",
    "AcousticEvent",
    "AcousticFrames",
    "AcousticState",
    "AcousticWindow",
    "FeaturePoint",
    "FramePoint",
    "Speaker",
    "TranscriptTurn",
]
