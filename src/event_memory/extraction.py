from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .llm_client import ChatMessage, LLMClient
from .schema import DialogueTurn, Event, EventType


VALID_EVENT_TYPES: set[str] = {
    "preference",
    "plan",
    "personal_fact",
    "relationship",
    "constraint",
    "completed_event",
    "future_event",
    "negated_fact",
    "other",
}


class RuleBasedEventExtractor:
    """Small deterministic extractor for the sample benchmark.

    This is a bootstrap implementation. It keeps the pipeline runnable before
    an LLM-backed extractor is connected.
    """

    def __init__(self) -> None:
        self._next_id = 1

    def extract_all(self, turns: Iterable[DialogueTurn]) -> list[Event]:
        events: list[Event] = []
        for turn in turns:
            events.extend(self.extract_turn(turn))
        return events

    def extract_turn(self, turn: DialogueTurn) -> list[Event]:
        if turn.speaker != "user":
            return []

        text = turn.text
        events: list[Event] = []

        if "準備研究所" in text or "研究所考試" in text:
            if "不考研究所" not in text:
                events.append(
                    self._event(
                        turn,
                        "plan",
                        "使用者準備研究所考試",
                        ["研究所"],
                        0.82,
                    )
                )

        if "不考研究所" in text:
            events.append(
                self._event(
                    turn,
                    "negated_fact",
                    "使用者不考研究所",
                    ["研究所"],
                    0.8,
                )
            )

        if "實習" in text:
            if "改" in text or "找" in text or "實習" in text:
                entities = ["實習"]
                if "研究所" in text:
                    entities.append("研究所")
                events.append(
                    self._event(
                        turn,
                        "plan",
                        "使用者改為尋找暑期實習" if "暑期" in text else "使用者尋找實習",
                        entities,
                        0.88,
                    )
                )

        if "台北" in text and "實習" in text and "不是台北" not in text:
            events.append(
                self._event(
                    turn,
                    "preference",
                    "使用者偏好台北的實習機會",
                    ["台北", "實習"],
                    0.68,
                )
            )

        if "不是台北" in text or "新竹" in text or "遠端" in text:
            entities = ["台北"]
            if "新竹" in text:
                entities.append("新竹")
            if "遠端" in text:
                entities.append("遠端")
            events.append(
                self._event(
                    turn,
                    "preference",
                    "使用者偏好新竹或遠端的機會",
                    entities,
                    0.74,
                )
            )

        if "每週三" in text or "晚上" in text:
            events.append(
                self._event(
                    turn,
                    "constraint",
                    "使用者每週三晚上需要照顧家人",
                    ["每週三", "晚上", "家人"],
                    0.72,
                )
            )

        return events

    def _event(
        self,
        turn: DialogueTurn,
        event_type: str,
        content: str,
        entities: list[str],
        importance: float,
    ) -> Event:
        event = Event(
            event_id=f"e_{self._next_id:04d}",
            user_id=turn.user_id,
            time=turn.timestamp,
            speaker=turn.speaker,
            subject="使用者",
            event_type=event_type,  # type: ignore[arg-type]
            content=content,
            entities=entities,
            evidence_turn_ids=[turn.turn_id],
            importance=importance,
        )
        self._next_id += 1
        return event


class LLMEventExtractor:
    """LLM-backed atomic event extractor.

    The extractor expects an OpenAI-compatible chat model and converts each
    user turn into zero or more schema-valid Event objects.
    """

    def __init__(self, llm: LLMClient, *, event_id_prefix: str = "llm_e") -> None:
        self.llm = llm
        self.event_id_prefix = event_id_prefix
        self._next_id = 1

    def extract_all(self, turns: Iterable[DialogueTurn], *, max_turns: int | None = None) -> list[Event]:
        events: list[Event] = []
        for index, turn in enumerate(turns):
            if max_turns is not None and index >= max_turns:
                break
            events.extend(self.extract_turn(turn))
        return events

    def extract_turn(self, turn: DialogueTurn) -> list[Event]:
        if turn.speaker != "user":
            return []

        result = self.llm.chat(
            [
                ChatMessage(role="system", content=_EXTRACT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=_turn_prompt(turn)),
            ],
            max_completion_tokens=900,
        )
        rows = _dedupe_event_rows(_parse_event_rows(result.content))
        return [self._event_from_row(turn, row) for row in rows]

    def _event_from_row(self, turn: DialogueTurn, row: dict) -> Event:
        event_type = str(row.get("event_type", "other")).strip()
        if event_type not in VALID_EVENT_TYPES:
            event_type = "other"

        content = str(row.get("content", "")).strip()
        if not content:
            content = turn.text.strip()

        entities_raw = row.get("entities", [])
        entities = [str(item).strip() for item in entities_raw if str(item).strip()] if isinstance(entities_raw, list) else []

        importance = _clamp_float(row.get("importance", 0.5), 0.0, 1.0)

        event = Event(
            event_id=f"{self.event_id_prefix}_{self._next_id:04d}",
            user_id=turn.user_id,
            time=turn.timestamp,
            speaker=turn.speaker,
            subject=str(row.get("subject", "使用者")).strip() or "使用者",
            event_type=event_type,  # type: ignore[arg-type]
            content=content,
            entities=entities,
            evidence_turn_ids=[turn.turn_id],
            importance=importance,
        )
        self._next_id += 1
        return event


_EXTRACT_SYSTEM_PROMPT = """你是繁體中文長期對話記憶的事件抽取器。
請從單一 user turn 抽取 atomic events。atomic event 是最小可更新記憶單位，一個事件只能描述一個核心事實、偏好、計畫、限制、關係、完成事件、未來事件、否定事實或其他資訊。

只輸出 JSON，格式必須是：
{
  "events": [
    {
      "event_type": "preference|plan|personal_fact|relationship|constraint|completed_event|future_event|negated_fact|other",
      "subject": "使用者",
      "content": "使用者...",
      "entities": ["..."],
      "importance": 0.0
    }
  ]
}

規則：
- 不要輸出 markdown。
- 不要猜測原文沒有說的資訊。
- 如果沒有可保存的長期記憶，輸出 {"events": []}。
- 不要建立 update relation；只抽取事件本身。
- evidence_turn_ids、user_id、timestamp 由系統補，不要輸出。
- 如果同一句話中兩個事件只是在重述同一個計畫或偏好，請只保留資訊量較完整的一個。
"""


def _turn_prompt(turn: DialogueTurn) -> str:
    return (
        f"user_id: {turn.user_id}\n"
        f"session_id: {turn.session_id}\n"
        f"turn_id: {turn.turn_id}\n"
        f"timestamp: {turn.timestamp}\n"
        f"text: {turn.text}\n"
    )


def _parse_event_rows(text: str) -> list[dict]:
    parsed = _loads_json_object(text)
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return []
    return [row for row in events if isinstance(row, dict)]


def _loads_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {"events": []}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"events": []}
    return parsed if isinstance(parsed, dict) else {"events": []}


def _clamp_float(value: object, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(low, min(high, number))


def _dedupe_event_rows(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for row in rows:
        event_type = str(row.get("event_type", "other"))
        entities_raw = row.get("entities", [])
        entities = tuple(sorted(str(item).strip() for item in entities_raw if str(item).strip())) if isinstance(entities_raw, list) else ()
        key = (event_type, entities)
        if entities and key in seen_keys:
            _merge_into_existing(deduped, key, row)
            continue
        seen_keys.add(key)
        deduped.append(row)
    return deduped


def _merge_into_existing(rows: list[dict], key: tuple[str, tuple[str, ...]], candidate: dict) -> None:
    candidate_content = str(candidate.get("content", ""))
    for index, row in enumerate(rows):
        event_type = str(row.get("event_type", "other"))
        entities_raw = row.get("entities", [])
        entities = tuple(sorted(str(item).strip() for item in entities_raw if str(item).strip())) if isinstance(entities_raw, list) else ()
        if (event_type, entities) != key:
            continue
        if len(candidate_content) > len(str(row.get("content", ""))):
            rows[index] = candidate
        return
