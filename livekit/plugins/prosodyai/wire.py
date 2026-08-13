"""The ProsodyAI event wire, declared once.

This module is the canonical declaration of the model-to-consumer event
vocabulary: the event type strings, the field names, and the typed shapes
that every parse and serialize site derives from.

The module imports only the standard library. Two frozen wires are declared
here.

1. The model wire: committed tracker events on the prediction envelope's
   ``events`` list, sent by the deployment and parsed by the API. Fields
   speak the model's clock: ``frame_ms`` and the integer ``lane``.
2. The gateway wire: kind-tagged frames on the caller socket, serialized by
   the API gateway and parsed by consumers (the LiveKit plugin, the website,
   the CLI probes). Fields speak the caller's vocabulary: ``timestamp_ms``,
   ``session_id``, and the lane label ``speaker_id`` (``speaker_<lane>``).

The bytes of both wires are frozen. Consumers align to the wire, and a
rename lands here before it lands anywhere else. Every field is a committed
fact; deliberation numbers never join these shapes
(``ci/validate_architecture_contract.py`` enforces the ban).

Parsers here are strict. A recognized event missing a required field raises
``ValueError`` naming the event and the field, because a producer that
omits a required field broke the contract and a fabricated default would
turn the breakage into a fake committed fact. Optional fields are the ones
today's producers legitimately send as null; they parse to ``None`` and
nothing else. Unrecognized event types parse to ``None`` so the vocabulary
can grow model-side first without breaking a consumer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Mapping, Optional, Union


class WireEventType(str, Enum):
    """Base for the wire's event-type families.

    Members are the wire bytes: equality, hashing, and formatting all speak
    the wire string, so a member drops into JSON payloads and mapping keys
    unchanged.
    """

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Gateway socket frame kinds: byte 0 of every caller frame.

KIND_HANDSHAKE = 0x00
KIND_AUDIO = 0x01
KIND_TEXT = 0x02
KIND_CONTROL = 0x03
KIND_IDENTITY = 0x04
KIND_TRANSCRIPT = 0x05
KIND_EVENT = 0x06


def speaker_label(lane: int) -> str:
    """The gateway's public label for one identity lane."""
    return f"speaker_{lane}"


def _required(entry: Mapping[str, Any], key: str, owner: str) -> Any:
    """Index a required field and fail loud when the producer broke contract.

    Absence and an explicit null are the same violation: the emit sites
    always send these fields with real values.
    """
    value = entry.get(key)
    if value is None:
        raise ValueError(f"{owner} is missing required field {key!r}")
    return value


def _optional_int(entry: Mapping[str, Any], key: str) -> Optional[int]:
    value = entry.get(key)
    return None if value is None else int(value)


def _optional_str(entry: Mapping[str, Any], key: str) -> Optional[str]:
    value = entry.get(key)
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# The model wire: committed tracker events on the prediction envelope.


class TrackerEventType(WireEventType):
    """The identity-state commitments the deployment puts on the envelope."""

    SPEAKER_CHANGE = "speaker_change"
    NEW_SPEAKER = "new_speaker"
    IDENTITY_RESOLVED = "identity_resolved"


@dataclass(frozen=True)
class TrackerSpeakerChangeEvent:
    """The outer recurrence committed the floor to a different lane.

    ``frame_ms`` is retrodictive: it points at the segment's onset on the
    model's frame clock, while the commit itself lands when the segment
    closes (or when a held segment resolves late). The latency between the
    two is the model's business.
    """

    TYPE: ClassVar[TrackerEventType] = TrackerEventType.SPEAKER_CHANGE

    frame_ms: int
    lane: int
    previous_lane: Optional[int]
    known_id: Optional[str] = None
    previous_known_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class TrackerNewSpeakerEvent:
    """The outer recurrence opened a state lane for a new voice."""

    TYPE: ClassVar[TrackerEventType] = TrackerEventType.NEW_SPEAKER

    frame_ms: int
    lane: int
    evidence_seconds: float

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class TrackerIdentityResolvedEvent:
    """A lane matched a stored person, once per lane, at its first commit.

    ``verified`` is true on every committed decision: the decoder's absolute
    membership test is the only thing that writes a lane.
    """

    TYPE: ClassVar[TrackerEventType] = TrackerEventType.IDENTITY_RESOLVED

    frame_ms: int
    lane: int
    person_id: str
    verified: bool

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


TrackerEvent = Union[
    TrackerSpeakerChangeEvent,
    TrackerNewSpeakerEvent,
    TrackerIdentityResolvedEvent,
]
"""One committed identity-state event off the prediction envelope. The model
is the only author: nothing downstream detects, thresholds, or reconstructs
these."""


# ---------------------------------------------------------------------------
# The model wire: committed conversation events decoded by the learned event
# deciders. They share the prediction envelope's ``events`` list with the
# tracker events and share nothing else. Every field is a committed fact on
# the model's frame clock; the decision bands that produced the commit stay
# inside the model.


class ConversationEventType(WireEventType):
    """The learned event deciders' commitments, on the same envelope list."""

    STATE_DELTA = "state_delta"
    TURN_BOUNDARY = "turn_boundary"
    BARGE_IN = "barge_in"


@dataclass(frozen=True)
class ConversationStateDeltaEvent:
    """``state_delta``: the lane's state moved decisively against its own
    baseline.

    This is the significance signal: δ_t committed by the model that has been
    carrying the person's state. ``magnitude`` is the L2 norm of the mean
    per-frame departure from baseline over the excursion, in state units.
    Emitted at commit (``resolved`` false, duration so far) and once more when
    the return to baseline is observed (``resolved`` true, full duration).
    ``frame_ms`` is retrodictive to the excursion's onset.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.STATE_DELTA

    frame_ms: int
    commit_ms: int
    duration_ms: int
    magnitude: float
    resolved: bool

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class ConversationTurnBoundaryEvent:
    """``turn_boundary``: the model committed the floor passed between voices.

    An instantaneous committed fact. ``frame_ms`` is retrodictive: it points
    at where the boundary evidence began on the model's frame clock, while
    ``commit_ms`` is where the decision landed.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.TURN_BOUNDARY

    frame_ms: int
    commit_ms: int

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class ConversationBargeInEvent:
    """``barge_in``: the model committed a second voice entered against held
    speech.

    Emitted at commit (``resolved`` false, duration so far) and once more
    when the end of the overlap is observed (``resolved`` true, full
    duration). ``frame_ms`` is retrodictive to the overlap's onset.
    """

    TYPE: ClassVar[ConversationEventType] = ConversationEventType.BARGE_IN

    frame_ms: int
    commit_ms: int
    duration_ms: int
    resolved: bool

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


ConversationWireEvent = Union[
    ConversationStateDeltaEvent,
    ConversationTurnBoundaryEvent,
    ConversationBargeInEvent,
]
"""One committed conversation event off the prediction envelope, decoded by
the learned event deciders with carried state. The model is the only author."""


# ---------------------------------------------------------------------------
# The identity timeline: one model-owned history of who spoke when.
#
# Diarization is the identity-state timeline. The deployment's tracker commits
# each frame to a lane, a new lane, or hold, and the ordered sequence of those
# commitments is the product's diarization. These shapes are the canonical
# readout: every surface (live events, session export, batch response) derives
# from them, and nothing downstream reconstructs a timeline from heuristics.


IDENTITY_TIMELINE_SCHEMA_VERSION = 1


class IdentityDecision(WireEventType):
    """The tracker's verdict for one audio span."""

    COMMIT = "commit"  # the span belongs to ``lane``
    MINT = "mint"  # the span opened a fresh lane for a new voice
    HOLD = "hold"  # evidence stayed unresolved; the span remains unattributed


@dataclass(frozen=True)
class IdentitySpan:
    """One committed lane decision over a span of audio-clock time.

    ``start_ms`` and ``end_ms`` bound the audio the decision covers, and are
    retrodictive when the segment closed (or a hold resolved) after the fact;
    ``commit_ms`` is where the verdict landed on the model's frame clock. A
    HOLD span carries ``lane`` for the candidate under test while keeping
    ``speaker_id`` null, because a hold attributes nothing. ``late_resolved``
    marks a span whose commitment arrived after its audio window, and
    ``unique_voice`` marks a minted lane the tracker will not merge into an
    existing one.
    """

    start_ms: int
    end_ms: int
    commit_ms: int
    lane: int
    speaker_id: Optional[str]
    decision: IdentityDecision
    late_resolved: bool = False
    unique_voice: bool = False

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            **{k: v for k, v in asdict(self).items() if k != "decision"},
        }


@dataclass(frozen=True)
class IdentityLane:
    """One lane in the session's lane book.

    ``person_id`` is the durable cross-session lineage identity, present only
    after a committed identity resolution; ``display_name`` labels it.
    ``is_returning`` marks a lane that resumed a persisted person, and
    ``is_agent`` marks the org's declared agent identity for self-recognition
    filtering.
    """

    lane: int
    speaker_id: str
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    is_returning: bool = False
    is_agent: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class IdentityTimeline:
    """The session's whole identity history, in commit order.

    ``spans`` carries every model assignment in the order the tracker
    committed them; ``lanes`` is the lane book those spans reference.
    ``events`` carries the committed tracker events verbatim so the durable
    record keeps the raw commit stream beside the derived spans.
    """

    schema_version: int
    model_provenance: Mapping[str, Any]
    spans: tuple[IdentitySpan, ...]
    lanes: tuple[IdentityLane, ...]
    events: tuple[TrackerEvent, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model_provenance": dict(self.model_provenance),
            "spans": [span.to_dict() for span in self.spans],
            "lanes": [lane.to_dict() for lane in self.lanes],
            "events": [event.to_dict() for event in self.events],
        }


def parse_identity_span(entry: Mapping[str, Any]) -> IdentitySpan:
    """Parse one identity span. Strict: every required field must be present."""
    owner = "identity span"
    return IdentitySpan(
        start_ms=int(_required(entry, "start_ms", owner)),
        end_ms=int(_required(entry, "end_ms", owner)),
        commit_ms=int(_required(entry, "commit_ms", owner)),
        lane=int(_required(entry, "lane", owner)),
        speaker_id=_optional_str(entry, "speaker_id"),
        decision=IdentityDecision(str(_required(entry, "decision", owner))),
        late_resolved=bool(entry.get("late_resolved") or False),
        unique_voice=bool(entry.get("unique_voice") or False),
    )


def parse_identity_lane(entry: Mapping[str, Any]) -> IdentityLane:
    """Parse one identity lane-book entry."""
    owner = "identity lane"
    return IdentityLane(
        lane=int(_required(entry, "lane", owner)),
        speaker_id=str(_required(entry, "speaker_id", owner)),
        person_id=_optional_str(entry, "person_id"),
        display_name=_optional_str(entry, "display_name"),
        is_returning=bool(entry.get("is_returning") or False),
        is_agent=bool(entry.get("is_agent") or False),
    )


def parse_identity_timeline(entry: Mapping[str, Any]) -> IdentityTimeline:
    """Parse a canonical identity timeline payload."""
    owner = "identity timeline"
    raw_spans = _required(entry, "spans", owner)
    raw_lanes = _required(entry, "lanes", owner)
    if not isinstance(raw_spans, list) or not isinstance(raw_lanes, list):
        raise ValueError(f"{owner} spans and lanes must be lists")
    raw_events = entry.get("events") or []
    return IdentityTimeline(
        schema_version=int(_required(entry, "schema_version", owner)),
        model_provenance=dict(_required(entry, "model_provenance", owner)),
        spans=tuple(parse_identity_span(span) for span in raw_spans),
        lanes=tuple(parse_identity_lane(lane) for lane in raw_lanes),
        events=tuple(
            event
            for item in raw_events
            if isinstance(item, Mapping) and (event := parse_tracker_event(item)) is not None
        ),
    )


def parse_conversation_event(
    entry: Mapping[str, Any],
) -> Optional[ConversationWireEvent]:
    """Parse one committed conversation event off the prediction envelope.

    Strict on recognized types: a missing required field raises
    ``ValueError`` naming the event and the field. An unrecognized ``type``
    returns ``None`` because the vocabulary grows model-side first.
    """
    try:
        event_type = ConversationEventType(str(entry.get("type")))
    except ValueError:
        return None
    owner = f"{event_type.value} event"
    if event_type is ConversationEventType.STATE_DELTA:
        return ConversationStateDeltaEvent(
            frame_ms=int(_required(entry, "frame_ms", owner)),
            commit_ms=int(_required(entry, "commit_ms", owner)),
            duration_ms=int(_required(entry, "duration_ms", owner)),
            magnitude=float(_required(entry, "magnitude", owner)),
            resolved=bool(_required(entry, "resolved", owner)),
        )
    if event_type is ConversationEventType.TURN_BOUNDARY:
        return ConversationTurnBoundaryEvent(
            frame_ms=int(_required(entry, "frame_ms", owner)),
            commit_ms=int(_required(entry, "commit_ms", owner)),
        )
    return ConversationBargeInEvent(
        frame_ms=int(_required(entry, "frame_ms", owner)),
        commit_ms=int(_required(entry, "commit_ms", owner)),
        duration_ms=int(_required(entry, "duration_ms", owner)),
        resolved=bool(_required(entry, "resolved", owner)),
    )


def parse_tracker_event(entry: Mapping[str, Any]) -> Optional[TrackerEvent]:
    """Parse one committed tracker event off the prediction envelope.

    The model wire's single parse site. Strict on recognized types: a
    missing required field raises ``ValueError`` naming the event and the
    field. An unrecognized ``type`` returns ``None`` because the vocabulary
    grows model-side first.
    """
    try:
        event_type = TrackerEventType(str(entry.get("type")))
    except ValueError:
        return None
    owner = f"{event_type.value} event"
    if event_type is TrackerEventType.SPEAKER_CHANGE:
        return TrackerSpeakerChangeEvent(
            frame_ms=int(_required(entry, "frame_ms", owner)),
            lane=int(_required(entry, "lane", owner)),
            previous_lane=_optional_int(entry, "previous_lane"),
            known_id=_optional_str(entry, "known_id"),
            previous_known_id=_optional_str(entry, "previous_known_id"),
        )
    if event_type is TrackerEventType.NEW_SPEAKER:
        return TrackerNewSpeakerEvent(
            frame_ms=int(_required(entry, "frame_ms", owner)),
            lane=int(_required(entry, "lane", owner)),
            evidence_seconds=float(_required(entry, "evidence_seconds", owner)),
        )
    return TrackerIdentityResolvedEvent(
        frame_ms=int(_required(entry, "frame_ms", owner)),
        lane=int(_required(entry, "lane", owner)),
        person_id=str(_required(entry, "person_id", owner)),
        verified=bool(_required(entry, "verified", owner)),
    )


# ---------------------------------------------------------------------------
# The gateway wire: committed model events on the 0x06 caller frame.


class GatewayEventType(WireEventType):
    """The committed model events the gateway serializes onto 0x06 frames."""

    SPEAKER_CHANGE = "prosodyai.speaker_change"
    NEW_SPEAKER = "prosodyai.new_speaker"
    IDENTITY_RESOLVED = "prosodyai.identity_resolved"
    AGENT_TOOL = "prosodyai.agent_tool"
    AGENT_THOUGHT = "prosodyai.agent_thought"
    AGENT_TOOL_STATUS = "prosodyai.agent_tool_status"


@dataclass(frozen=True)
class GatewaySpeakerChangeEvent:
    """``prosodyai.speaker_change``: the model committed the floor moved lanes.

    ``timestamp_ms`` is retrodictive: it points at the turn's onset on the
    model's frame clock; the commit itself landed when the segment closed
    (or when a held segment resolved late).
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.SPEAKER_CHANGE

    timestamp_ms: int
    session_id: str
    speaker_id: str
    previous_speaker_id: Optional[str]
    person_id: Optional[str]
    display_name: Optional[str]
    is_agent: bool

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class GatewayNewSpeakerEvent:
    """``prosodyai.new_speaker``: a lane opened for a voice never heard here."""

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.NEW_SPEAKER

    timestamp_ms: int
    session_id: str
    speaker_id: str
    evidence_seconds: float

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class GatewayIdentityResolvedEvent:
    """``prosodyai.identity_resolved``: a lane matched a stored person.

    Fires once per lane, at its first committed segment. ``verified`` is true
    when the decision came from the decoder's absolute membership test.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.IDENTITY_RESOLVED

    timestamp_ms: int
    session_id: str
    speaker_id: str
    person_id: Optional[str]
    display_name: Optional[str]
    verified: bool

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class GatewayAgentToolEvent:
    """``prosodyai.agent_tool``: one completed capability, shown to the caller.

    The reasoner chose the tool, the gateway ran it, and ``result`` is what
    the speech model was told. This is the acting half of the reasoner's
    deliberation; ``prosodyai.agent_thought`` is the reading half.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.AGENT_TOOL

    session_id: str
    name: str
    arguments: dict[str, Any]
    result: str

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class GatewayAgentThoughtEvent:
    """``prosodyai.agent_thought``: the reasoner's own read of the moment.

    One deliberation line per reasoning pass, in the reasoner's own words,
    written before it decides whether any capability is warranted. It is
    Jarvis thinking, and it is never spoken: the speech model never receives
    it and it carries no measurement, verdict, or score.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.AGENT_THOUGHT

    session_id: str
    text: str

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


class ToolCallStatus(WireEventType):
    """The lifecycle stages of one capability invocation."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class GatewayAgentToolStatusEvent:
    """``prosodyai.agent_tool_status``: one stage of a capability's lifecycle.

    ``started`` fires the moment the reasoner commits to running the tool;
    ``completed`` carries the result; ``failed`` carries the error. PersonaPlex
    only ever receives the completed exchange (it cannot await a round trip), so
    these events exist for the caller surface and never join the control
    channel. ``call_id`` ties the stages together.
    """

    TYPE: ClassVar[GatewayEventType] = GatewayEventType.AGENT_TOOL_STATUS

    session_id: str
    call_id: str
    name: str
    status: ToolCallStatus
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.TYPE.value,
            "status": self.status.value,
            **{k: v for k, v in asdict(self).items() if k != "status"},
        }


GatewayModelEvent = Union[
    GatewaySpeakerChangeEvent,
    GatewayNewSpeakerEvent,
    GatewayIdentityResolvedEvent,
    GatewayAgentToolEvent,
    GatewayAgentThoughtEvent,
    GatewayAgentToolStatusEvent,
]
"""A committed model decision off the gateway's 0x06 event channel."""


def parse_gateway_model_event(
    frame: Mapping[str, Any],
) -> Optional[GatewayModelEvent]:
    """Parse one 0x06 committed-model-event payload into its typed form.

    The gateway wire's single parse site for model events. Strict on
    recognized types: a missing required field raises ``ValueError`` naming
    the event and the field. An unrecognized ``type`` returns ``None`` so an
    unknown event never breaks the socket.
    """
    try:
        event_type = GatewayEventType(str(frame.get("type")))
    except ValueError:
        return None
    owner = f"{event_type.value} event"
    if event_type is GatewayEventType.SPEAKER_CHANGE:
        return GatewaySpeakerChangeEvent(
            timestamp_ms=int(_required(frame, "timestamp_ms", owner)),
            session_id=str(_required(frame, "session_id", owner)),
            speaker_id=str(_required(frame, "speaker_id", owner)),
            previous_speaker_id=_optional_str(frame, "previous_speaker_id"),
            person_id=_optional_str(frame, "person_id"),
            display_name=_optional_str(frame, "display_name"),
            is_agent=bool(_required(frame, "is_agent", owner)),
        )
    if event_type is GatewayEventType.NEW_SPEAKER:
        return GatewayNewSpeakerEvent(
            timestamp_ms=int(_required(frame, "timestamp_ms", owner)),
            session_id=str(_required(frame, "session_id", owner)),
            speaker_id=str(_required(frame, "speaker_id", owner)),
            evidence_seconds=float(_required(frame, "evidence_seconds", owner)),
        )
    if event_type is GatewayEventType.IDENTITY_RESOLVED:
        return GatewayIdentityResolvedEvent(
            timestamp_ms=int(_required(frame, "timestamp_ms", owner)),
            session_id=str(_required(frame, "session_id", owner)),
            speaker_id=str(_required(frame, "speaker_id", owner)),
            person_id=_optional_str(frame, "person_id"),
            display_name=_optional_str(frame, "display_name"),
            verified=bool(_required(frame, "verified", owner)),
        )
    if event_type is GatewayEventType.AGENT_TOOL:
        arguments = frame.get("arguments")
        return GatewayAgentToolEvent(
            session_id=str(_required(frame, "session_id", owner)),
            name=str(_required(frame, "name", owner)),
            arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
            result=str(frame.get("result") or ""),
        )
    if event_type is GatewayEventType.AGENT_TOOL_STATUS:
        arguments = frame.get("arguments")
        return GatewayAgentToolStatusEvent(
            session_id=str(_required(frame, "session_id", owner)),
            call_id=str(_required(frame, "call_id", owner)),
            name=str(_required(frame, "name", owner)),
            status=ToolCallStatus(str(_required(frame, "status", owner))),
            arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
            result=_optional_str(frame, "result"),
            error=_optional_str(frame, "error"),
        )
    return GatewayAgentThoughtEvent(
        session_id=str(_required(frame, "session_id", owner)),
        text=str(_required(frame, "text", owner)),
    )


# ---------------------------------------------------------------------------
# The gateway wire: identity (0x04) and transcript (0x05) caller frames.
# ``to_payload`` is the gateway serialize site (the frame body carries no
# ``type`` key). ``to_dict`` adds the room topic type for consumers that
# republish the event, and the type strings below name those republished
# events on the LiveKit data topic.


class RoomEventType(WireEventType):
    """The republished event names on the LiveKit data topic."""

    TEXT = "prosodyai.text"
    IDENTITY = "prosodyai.identity"
    TRANSCRIPT = "prosodyai.transcript"


@dataclass(frozen=True)
class TranscriptDelta:
    """One committed span of a speaker's words, times in their own audio."""

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class IdentityEvent:
    """A committed identity resolution, announced mid-conversation.

    ``recognized_at_ms`` is the audio position (ms into the call) of the
    frame whose tracker assignment resolved the person. The model owns this
    clock, so it is the recognition time; consumers report it and derive
    nothing from it.
    """

    TYPE: ClassVar[RoomEventType] = RoomEventType.IDENTITY

    speaker_id: str
    person_id: str
    display_name: Optional[str]
    is_returning: bool
    recognized_at_ms: int

    def to_payload(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


@dataclass(frozen=True)
class TranscriptEvent:
    """Words for one tracked frame, attributed by the model's tracker.

    Subtitles only. Speaker and identity decisions arrive on the committed
    model-event channel and are never derived from this text.
    """

    TYPE: ClassVar[RoomEventType] = RoomEventType.TRANSCRIPT

    speaker_id: str
    deltas: tuple[TranscriptDelta, ...]

    def to_payload(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, **asdict(self)}


def parse_identity_payload(frame: Mapping[str, Any]) -> IdentityEvent:
    """Parse one 0x04 committed-identity payload.

    Strict on required fields. Fields outside the declared shape are
    ignored so a grown resolution never breaks the socket.
    """
    owner = f"{RoomEventType.IDENTITY.value} event"
    return IdentityEvent(
        speaker_id=str(_required(frame, "speaker_id", owner)),
        person_id=str(_required(frame, "person_id", owner)),
        display_name=_optional_str(frame, "display_name"),
        is_returning=bool(_required(frame, "is_returning", owner)),
        recognized_at_ms=int(_required(frame, "recognized_at_ms", owner)),
    )


def parse_transcript_payload(frame: Mapping[str, Any]) -> Optional[TranscriptEvent]:
    """Parse one 0x05 lane-attributed transcript payload.

    Strict on required fields, in every delta included. A payload without
    deltas parses to ``None``.
    """
    owner = f"{RoomEventType.TRANSCRIPT.value} event"
    raw_deltas = frame.get("deltas") or []
    if any(not isinstance(delta, Mapping) for delta in raw_deltas):
        raise ValueError(f"{owner} deltas must be objects")
    deltas = tuple(
        TranscriptDelta(
            text=str(_required(delta, "text", f"{owner} delta")),
            start_ms=int(_required(delta, "start_ms", f"{owner} delta")),
            end_ms=int(_required(delta, "end_ms", f"{owner} delta")),
        )
        for delta in raw_deltas
    )
    if not deltas:
        return None
    return TranscriptEvent(
        speaker_id=str(_required(frame, "speaker_id", owner)),
        deltas=deltas,
    )


# ---------------------------------------------------------------------------
# The realtime session event stream: one versioned envelope per event.

SESSION_EVENT_ENVELOPE_VERSION = 1
SESSION_ENVELOPE_KEYS = ("version", "session_id", "generation", "seq", "type")

SESSION_STARTED = "session.started"
SESSION_RESET = "session.reset"
SESSION_FINALIZED = "session.finalized"
SESSION_PROCESSOR_ERROR = "session.processor_error"
PROSODY_DIRECTIVE = "prosody.directive"
ANALYSIS_FINAL = "analysis.final"


@dataclass(frozen=True)
class SessionEvent:
    """One versioned event off a realtime session's stream.

    The envelope keys are fixed and reserved; ``fields`` carries the event
    body and is flattened beside them at the serialize site.
    """

    version: int
    session_id: str
    generation: int
    seq: int
    type: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "generation": self.generation,
            "seq": self.seq,
            "type": self.type,
            **dict(self.fields),
        }


# ---------------------------------------------------------------------------
# The vocabulary manifest: what the drift tests compare across trees.


def _field_names(event_class: type) -> tuple[str, ...]:
    return tuple(f.name for f in fields(event_class))


def vocabulary() -> dict[str, tuple[str, ...]]:
    """Every event type string mapped to its field names, in wire order."""
    entries: dict[str, tuple[str, ...]] = {
        event_class.TYPE.value: _field_names(event_class)
        for event_class in (
            TrackerSpeakerChangeEvent,
            TrackerNewSpeakerEvent,
            TrackerIdentityResolvedEvent,
            ConversationStateDeltaEvent,
            ConversationTurnBoundaryEvent,
            ConversationBargeInEvent,
            GatewaySpeakerChangeEvent,
            GatewayNewSpeakerEvent,
            GatewayIdentityResolvedEvent,
            GatewayAgentToolEvent,
            GatewayAgentThoughtEvent,
            GatewayAgentToolStatusEvent,
            IdentityEvent,
            TranscriptEvent,
        )
    }
    entries["session.envelope"] = SESSION_ENVELOPE_KEYS
    return entries
