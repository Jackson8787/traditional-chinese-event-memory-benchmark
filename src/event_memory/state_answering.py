from __future__ import annotations

import re

from .event_structure import infer_event_structure
from .schema import Answer, QAItem, RetrievalResult


class StateAwareAnswerer:
    def __init__(self, fallback) -> None:
        self._fallback = fallback

    def answer(self, qa: QAItem, retrieved: list[RetrievalResult]) -> Answer:
        if qa.requires_abstention or not retrieved:
            return self._fallback.answer(qa, retrieved)

        topic = _topic_from_question(qa.question)
        if topic is None:
            return self._fallback.answer(qa, retrieved)

        state_events = [_state_event(result, topic) for result in retrieved]
        state_events = [event for event in state_events if event is not None]
        if not state_events:
            return self._fallback.answer(qa, retrieved)

        if _asks_change(qa):
            old = _oldest_state(state_events)
            new = _newest_state(state_events)
            if old is not None and new is not None and old.event_id != new.event_id:
                answer = f"使用者原本的{topic}是{old.value}；使用者目前的{topic}改成{new.value}"
                return _state_answer(qa, answer, [old, new], retrieved)
            return self._fallback.answer(qa, retrieved)

        current = _newest_state(state_events)
        if current is not None:
            answer = f"使用者目前的{topic}改成{current.value}"
            return _state_answer(qa, answer, [current], retrieved)

        return self._fallback.answer(qa, retrieved)


class _StateEvent:
    def __init__(self, event_id: str, value: str, time: str, evidence_turn_ids: list[str]) -> None:
        self.event_id = event_id
        self.value = value
        self.time = time
        self.evidence_turn_ids = evidence_turn_ids


def _state_answer(
    qa: QAItem,
    answer: str,
    state_events: list[_StateEvent],
    retrieved: list[RetrievalResult],
) -> Answer:
    evidence_turn_ids: list[str] = []
    for event in sorted(state_events, key=lambda item: item.time):
        for turn_id in event.evidence_turn_ids:
            if turn_id not in evidence_turn_ids:
                evidence_turn_ids.append(turn_id)
    return Answer(
        question_id=qa.question_id,
        answer=answer,
        evidence_turn_ids=evidence_turn_ids,
        confidence="high",
        retrieved_event_ids=[result.event.event_id for result in retrieved],
    )


def _topic_from_question(question: str) -> str | None:
    patterns = [
        r"目前的(.+?)是什麼",
        r"原本和後來的(.+?)分別",
        r"的(.+?)前後怎麼變化",
        r"目前狀態，.+?的(.+?)應該",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip("？? ")
    return None


def _asks_change(qa: QAItem) -> bool:
    return qa.question_type in {"temporal_reasoning", "multi_session_reasoning"} or any(
        term in qa.question for term in ["原本和後來", "前後", "變化"]
    )


def _state_event(result: RetrievalResult, topic: str) -> _StateEvent | None:
    event = result.event
    if not _mentions_topic(event.content, event.entities, topic):
        return None
    value = _value_from_content(event.content, topic)
    if value is None:
        value = _value_from_entities(event.entities, topic)
    if value is None:
        structure = infer_event_structure(event)
        value = structure.value
    if not value:
        return None
    return _StateEvent(event.event_id, value, event.time, event.evidence_turn_ids)


def _mentions_topic(content: str, entities: list[str], topic: str) -> bool:
    compact_topic = _compact_state_text(topic)
    compact_content = _compact_state_text(content)
    if compact_topic in compact_content:
        return True
    return any(
        compact_topic in _compact_state_text(entity) or _compact_state_text(entity) in compact_topic
        for entity in entities
    )


def _value_from_content(content: str, topic: str) -> str | None:
    escaped_topic = re.escape(topic)
    patterns = [
        rf"{escaped_topic}改成(?P<value>[^，。；;]+)",
        rf"{escaped_topic}改為(?P<value>[^，。；;]+)",
        rf"{escaped_topic}是(?P<value>[^，。；;]+)",
        rf"把{escaped_topic}[^，。；;]*?改成(?P<value>[^，。；;]+)",
        rf"將{escaped_topic}[^，。；;]*?改為(?P<value>[^，。；;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return _clean_state_value(match.group("value"))
    return None


def _value_from_entities(entities: list[str], topic: str) -> str | None:
    topic_compact = _compact_state_text(topic)
    for entity in reversed(entities):
        entity_compact = _compact_state_text(entity)
        if entity_compact and entity_compact != topic_compact and entity_compact not in topic_compact:
            return entity
    return None


def _oldest_state(events: list[_StateEvent]) -> _StateEvent | None:
    return min(events, key=lambda event: event.time, default=None)


def _newest_state(events: list[_StateEvent]) -> _StateEvent | None:
    return max(events, key=lambda event: event.time, default=None)


def _clean_state_value(value: str) -> str:
    return value.strip(" ，。；;")


def _compact_state_text(text: str) -> str:
    return re.sub(r"\s|[，。！？；：,.!?;:]", "", text)
