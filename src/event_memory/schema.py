from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EventType = Literal[
    "preference",
    "plan",
    "personal_fact",
    "relationship",
    "constraint",
    "completed_event",
    "future_event",
    "negated_fact",
    "other",
]

UpdateRelationType = Literal[
    "supersedes",
    "corrects",
    "supplements",
    "conflicts_without_resolution",
    "unrelated",
]


@dataclass(frozen=True)
class DialogueTurn:
    user_id: str
    session_id: str
    turn_id: str
    speaker: str
    timestamp: str
    text: str


@dataclass
class Event:
    event_id: str
    user_id: str
    time: str
    speaker: str
    subject: str
    event_type: EventType
    content: str
    entities: list[str]
    evidence_turn_ids: list[str]
    source_context_ids: list[str] = field(default_factory=list)
    importance: float = 0.5
    superseded_by: str | None = None
    corrected_by: str | None = None


@dataclass(frozen=True)
class UpdateRelation:
    new_event_id: str
    old_event_id: str
    relation: UpdateRelationType
    reason: str
    evidence_turn_ids: list[str]


@dataclass(frozen=True)
class Persona:
    user_id: str
    persona_type: str
    name: str
    profile: str


@dataclass(frozen=True)
class QAItem:
    question_id: str
    user_id: str
    question: str
    question_type: str
    gold_answer: str
    gold_evidence_turn_ids: list[str]
    valid_time: str
    requires_abstention: bool = False
    gold_event_ids: list[str] = field(default_factory=list)
    gold_update_relations: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class QueryIntent:
    asks_current: bool
    asks_history: bool
    requires_abstention: bool
    time_hint: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    event: Event
    score: float
    evidence_turns: list[DialogueTurn]


@dataclass(frozen=True)
class Answer:
    question_id: str
    answer: str
    evidence_turn_ids: list[str]
    confidence: Literal["low", "medium", "high"]
    retrieved_event_ids: list[str]
