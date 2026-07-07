from __future__ import annotations

import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .experiment import apply_update_relations
from .io import load_qa, load_turns, read_jsonl, write_jsonl
from .llm_client import ChatMessage, LLMClient
from .schema import DialogueTurn, Event, UpdateRelation
from .update import LLMConflictAwareUpdater


MAX_EVENTS_PER_SESSION = 12
MAX_RELATION_CANDIDATES_PER_EVENT = 4
VALID_EVENT_TYPES = {
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


def build_llm_memory(
    dataset_dir: str | Path,
    output_dir: str | Path,
    llm: LLMClient,
    *,
    limit_qa: int = 60,
    force: bool = False,
) -> dict[str, int]:
    started = time.perf_counter()
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    sessions_dir = output / "sessions"
    output.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    if force:
        _clear_cache_dir(sessions_dir)
        for cache_file in [output / "relations.jsonl"]:
            if cache_file.exists():
                cache_file.unlink()

    qa_items = load_qa(dataset / "qa.jsonl")[:limit_qa]
    selected_users = {qa.user_id for qa in qa_items}
    turns = [turn for turn in load_turns(dataset / "dialogues.jsonl") if turn.user_id in selected_users]
    user_turns = [turn for turn in turns if turn.speaker == "user"]
    sessions = _group_sessions(user_turns)

    llm_calls_attempted = 0
    llm_calls_failed = 0
    session_cache_hits = 0
    session_cache_misses = 0
    errors: list[dict] = []
    events: list[Event] = []

    for session_key in sorted(sessions):
        session_turns = sessions[session_key]
        cache_path = sessions_dir / f"{session_key}.json"
        if cache_path.exists() and not force:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            session_cache_hits += 1
        else:
            cache = _build_session_cache(session_key, session_turns, llm)
            llm_calls_attempted += 1
            session_cache_misses += 1
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if cache["parse_status"] != "ok":
            llm_calls_failed += 1
            errors.append(
                {
                    "user_id": session_turns[0].user_id,
                    "session_id": session_turns[0].session_id,
                    "error_type": cache["parse_status"],
                    "message": cache.get("error_message", ""),
                }
            )
            continue

        events.extend(_events_from_cache(cache, len(events) + 1))

    relations, relation_stats = _load_or_build_relations(output / "relations.jsonl", events, llm, force=force)
    memory = apply_update_relations(events, relations)

    write_jsonl(output / "pred_events.jsonl", (_event_row(event) for event in memory))
    write_jsonl(output / "pred_update_relations.jsonl", (_relation_row(relation) for relation in relations))
    write_jsonl(output / "memory_build_errors.jsonl", errors)

    stats = {
        "selected_qa_count": len(qa_items),
        "selected_user_count": len(selected_users),
        "selected_session_count": len(sessions),
        "llm_calls_attempted": llm_calls_attempted,
        "llm_calls_failed": llm_calls_failed,
        "session_cache_hits": session_cache_hits,
        "session_cache_misses": session_cache_misses,
        "relation_judge_attempts": relation_stats["attempts"],
        "relation_judge_failures": relation_stats["failures"],
        "parsed_event_count": len(events),
        "predicted_relation_count": len(relations),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
    (output / "memory_build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "memory_build_manifest.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset),
                "limit_qa": limit_qa,
                "cache_granularity": "user_id+session_id",
                "selected_user_ids": sorted(selected_users),
                "selected_session_keys": sorted(sessions),
                "artifacts": [
                    "pred_events.jsonl",
                    "pred_update_relations.jsonl",
                    "memory_build_stats.json",
                    "memory_build_errors.jsonl",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return stats


def _group_sessions(turns: Iterable[DialogueTurn]) -> dict[str, list[DialogueTurn]]:
    grouped: dict[str, list[DialogueTurn]] = defaultdict(list)
    for turn in turns:
        grouped[f"{turn.user_id}__{turn.session_id}"].append(turn)
    return {key: sorted(value, key=lambda turn: turn.turn_id) for key, value in grouped.items()}


def _build_session_cache(session_key: str, turns: list[DialogueTurn], llm: LLMClient) -> dict:
    raw_response = ""
    try:
        result = llm.chat(
            [
                ChatMessage(role="system", content=_SESSION_EXTRACT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=_session_prompt(turns)),
            ],
            max_completion_tokens=1600,
        )
        raw_response = result.content
        events = _parse_session_events(raw_response, turns)[:MAX_EVENTS_PER_SESSION]
        return {
            "session_key": session_key,
            "user_id": turns[0].user_id,
            "session_id": turns[0].session_id,
            "turns": [_turn_row(turn) for turn in turns],
            "raw_response": raw_response,
            "parsed_events": events,
            "parse_status": "ok",
        }
    except Exception as exc:
        return {
            "session_key": session_key,
            "user_id": turns[0].user_id,
            "session_id": turns[0].session_id,
            "turns": [_turn_row(turn) for turn in turns],
            "raw_response": raw_response,
            "parsed_events": [],
            "parse_status": "parse_error",
            "error_message": str(exc),
        }


def _parse_session_events(text: str, turns: list[DialogueTurn]) -> list[dict]:
    parsed = _loads_json_object(text)
    rows = parsed.get("events", [])
    if not isinstance(rows, list):
        raise ValueError("events must be a list")

    turn_ids = {turn.turn_id for turn in turns}
    parsed_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_turn_id = str(row.get("source_turn_id", "")).strip()
        if source_turn_id not in turn_ids:
            continue
        event_type = str(row.get("event_type", "other")).strip()
        if event_type not in VALID_EVENT_TYPES:
            event_type = "other"
        entities = row.get("entities", [])
        parsed_rows.append(
            {
                "source_turn_id": source_turn_id,
                "event_type": event_type,
                "subject": str(row.get("subject", "使用者")).strip() or "使用者",
                "content": str(row.get("content", "")).strip(),
                "entities": [str(entity).strip() for entity in entities if str(entity).strip()]
                if isinstance(entities, list)
                else [],
                "importance": _clamp_float(row.get("importance", 0.5), 0.0, 1.0),
            }
        )
    return parsed_rows


def _events_from_cache(cache: dict, next_index: int) -> list[Event]:
    turns = {turn["turn_id"]: turn for turn in cache["turns"]}
    events: list[Event] = []
    for offset, row in enumerate(cache["parsed_events"]):
        turn = turns[row["source_turn_id"]]
        content = row["content"] or turn["text"]
        events.append(
            Event(
                event_id=f"pred_e_{next_index + offset:04d}",
                user_id=turn["user_id"],
                time=turn["timestamp"],
                speaker="user",
                subject=row["subject"],
                event_type=row["event_type"],
                content=content,
                entities=row["entities"],
                evidence_turn_ids=[row["source_turn_id"]],
                importance=row["importance"],
            )
        )
    return events


def _load_or_build_relations(
    cache_path: Path,
    events: list[Event],
    llm: LLMClient,
    *,
    force: bool,
) -> tuple[list[UpdateRelation], dict[str, int]]:
    if cache_path.exists() and not force:
        relations = [
            UpdateRelation(
                new_event_id=row["new_event_id"],
                old_event_id=row["old_event_id"],
                relation=row["relation"],
                reason=row["reason"],
                evidence_turn_ids=row["evidence_turn_ids"],
            )
            for row in read_jsonl(cache_path)
        ]
        return relations, {"attempts": 0, "failures": 0}
    relations, stats = _build_bounded_update_relations(events, llm)
    write_jsonl(cache_path, (_relation_row(relation) for relation in relations))
    return relations, stats


def _build_bounded_update_relations(events: list[Event], llm: LLMClient) -> tuple[list[UpdateRelation], dict[str, int]]:
    updater = LLMConflictAwareUpdater(llm)
    memory: list[Event] = []
    relations: list[UpdateRelation] = []
    attempts = 0
    failures = 0
    for event in events:
        if not _looks_like_update_event(event):
            memory.append(event)
            continue
        candidates = updater._candidates(event, memory)[-MAX_RELATION_CANDIDATES_PER_EVENT:]
        for old in candidates:
            attempts += 1
            try:
                relation = updater._judge(event, old)
            except Exception:
                failures += 1
                continue
            if relation is None:
                continue
            relations.append(relation)
            if relation.relation == "supersedes":
                old.superseded_by = event.event_id
            elif relation.relation == "corrects":
                old.corrected_by = event.event_id
        memory.append(event)
    return relations, {"attempts": attempts, "failures": failures}


def _looks_like_update_event(event: Event) -> bool:
    if event.event_type == "negated_fact":
        return True
    update_terms = [
        "改成",
        "改為",
        "換成",
        "後來",
        "不再",
        "不想繼續",
        "取消",
        "修正",
        "不是",
        "changed",
        "change",
    ]
    return any(term in event.content for term in update_terms)


def _loads_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("response did not contain a JSON object") from exc
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def _session_prompt(turns: list[DialogueTurn]) -> str:
    payload = {
        "user_id": turns[0].user_id,
        "session_id": turns[0].session_id,
        "turns": [_turn_row(turn) for turn in turns],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _turn_row(turn: DialogueTurn) -> dict:
    return {
        "user_id": turn.user_id,
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "speaker": turn.speaker,
        "timestamp": turn.timestamp,
        "text": turn.text,
    }


def _event_row(event: Event) -> dict:
    return {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "time": event.time,
        "speaker": event.speaker,
        "subject": event.subject,
        "event_type": event.event_type,
        "content": event.content,
        "entities": event.entities,
        "evidence_turn_ids": event.evidence_turn_ids,
        "source_context_ids": event.source_context_ids,
        "importance": event.importance,
        "superseded_by": event.superseded_by,
        "corrected_by": event.corrected_by,
    }


def _relation_row(relation: UpdateRelation) -> dict:
    return {
        "new_event_id": relation.new_event_id,
        "old_event_id": relation.old_event_id,
        "relation": relation.relation,
        "reason": relation.reason,
        "evidence_turn_ids": relation.evidence_turn_ids,
    }


def _clear_cache_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _clamp_float(value: object, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return min(max(number, low), high)


_SESSION_EXTRACT_SYSTEM_PROMPT = """你是繁體中文長期對話記憶的 session-level 事件抽取器。
你會收到同一個 session 的多個 user turns。請抽取 atomic events。

只輸出 JSON，格式必須是：
{
  "events": [
    {
      "source_turn_id": "原始 turn_id",
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
- 每個 event 必須有 source_turn_id，且必須來自輸入 turns。
- 每個 session 最多輸出 12 個 events。
- 不要建立 update relation。
- 不要猜測原文沒有說的資訊。
- 如果沒有可保存的長期記憶，輸出 {"events": []}。
"""
