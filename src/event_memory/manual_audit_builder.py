from __future__ import annotations

import json
from pathlib import Path

from .eval import is_answer_correct
from .event_structure import infer_event_structure
from .io import load_qa, read_jsonl, write_jsonl
from .schema import Answer, Event, QAItem


def build_manual_audit_package(
    *,
    dataset_dir: str | Path,
    memory_dir: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    relation_limit: int = 100,
    answer_limit: int = 50,
) -> dict:
    dataset = Path(dataset_dir)
    memory = Path(memory_dir)
    experiment = Path(experiment_dir)
    output = Path(output_dir)

    events = [_event_from_row(row) for row in read_jsonl(memory / "pred_events.jsonl")]
    relations = read_jsonl(memory / "pred_update_relations.jsonl")
    qa_items = load_qa(dataset / "qa.jsonl")
    answers = [_answer_from_row(row) for row in read_jsonl(experiment / "answers.jsonl")]
    error_rows = _read_optional_jsonl(experiment / "error_analysis.jsonl")
    turns = _load_turn_texts(dataset)

    relation_rows = build_relation_precision_audit_rows(
        events,
        relations,
        turns,
        limit=relation_limit,
    )
    answer_rows = build_answer_faithfulness_audit_rows(
        qa_items,
        answers,
        error_rows,
        turns,
        limit=answer_limit,
    )
    report = {
        "relation_audit_count": len(relation_rows),
        "answer_audit_count": len(answer_rows),
        "relation_limit": relation_limit,
        "answer_limit": answer_limit,
        "input_relation_count": len(relations),
        "input_answer_count": len(answers),
        "output_dir": str(output),
        "manual_label_schema": {
            "manual_relation_label": "correct|partially_correct|incorrect|unclear",
            "manual_answer_label": "correct|partially_correct|incorrect|abstention_correct|unclear",
            "manual_evidence_label": "complete|partial|wrong|not_needed|unclear",
        },
    }

    write_jsonl(output / "relation_precision_audit.jsonl", relation_rows)
    write_jsonl(output / "answer_faithfulness_audit.jsonl", answer_rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manual_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_relation_precision_audit_rows(
    events: list[Event],
    relations: list[dict],
    turn_texts: dict[str, str],
    *,
    limit: int = 100,
) -> list[dict]:
    events_by_id = {event.event_id: event for event in events}
    rows: list[dict] = []
    for index, relation in enumerate(_prioritize_relations(relations), start=1):
        if len(rows) >= limit:
            break
        old_event = events_by_id.get(str(relation.get("old_event_id", "")))
        new_event = events_by_id.get(str(relation.get("new_event_id", "")))
        if old_event is None or new_event is None:
            continue
        old_structure = infer_event_structure(old_event)
        new_structure = infer_event_structure(new_event)
        rows.append(
            {
                "audit_id": f"relation_audit_{index:04d}",
                "relation": relation.get("relation", ""),
                "old_event_id": old_event.event_id,
                "new_event_id": new_event.event_id,
                "old_event_content": old_event.content,
                "new_event_content": new_event.content,
                "old_slot": old_structure.slot,
                "new_slot": new_structure.slot,
                "old_value": old_structure.value,
                "new_value": new_structure.value,
                "slot_match": old_structure.slot == new_structure.slot,
                "relation_reason": relation.get("reason", ""),
                "old_evidence_turn_ids": old_event.evidence_turn_ids,
                "new_evidence_turn_ids": new_event.evidence_turn_ids,
                "old_evidence_text": _joined_turn_text(old_event.evidence_turn_ids, turn_texts),
                "new_evidence_text": _joined_turn_text(new_event.evidence_turn_ids, turn_texts),
                "manual_relation_label": "",
                "manual_relation_note": "",
            }
        )
    return rows


def build_answer_faithfulness_audit_rows(
    qa_items: list[QAItem],
    answers: list[Answer],
    error_rows: list[dict],
    turn_texts: dict[str, str],
    *,
    limit: int = 50,
) -> list[dict]:
    answers_by_id = {answer.question_id: answer for answer in answers}
    errors_by_id = {row.get("question_id"): row for row in error_rows}
    rows: list[dict] = []
    for index, qa in enumerate(_prioritize_qa_items(qa_items, errors_by_id), start=1):
        if len(rows) >= limit:
            break
        answer = answers_by_id.get(qa.question_id)
        if answer is None:
            continue
        gold_set = set(qa.gold_evidence_turn_ids)
        predicted_set = set(answer.evidence_turn_ids)
        rows.append(
            {
                "audit_id": f"answer_audit_{index:04d}",
                "question_id": qa.question_id,
                "question_type": qa.question_type,
                "valid_time": qa.valid_time,
                "question": qa.question,
                "gold_answer": qa.gold_answer,
                "predicted_answer": answer.answer,
                "automatic_normalized_correct": is_answer_correct(qa, answer),
                "automatic_gold_evidence_covered": gold_set.issubset(predicted_set),
                "gold_evidence_turn_ids": qa.gold_evidence_turn_ids,
                "predicted_evidence_turn_ids": answer.evidence_turn_ids,
                "missing_gold_evidence_turn_ids": sorted(gold_set - predicted_set),
                "extra_predicted_evidence_turn_ids": sorted(predicted_set - gold_set),
                "gold_evidence_text": _joined_turn_text(qa.gold_evidence_turn_ids, turn_texts),
                "predicted_evidence_text": _joined_turn_text(answer.evidence_turn_ids, turn_texts),
                "retrieved_event_ids": answer.retrieved_event_ids,
                "error_source": errors_by_id.get(qa.question_id, {}).get("error_source", ""),
                "manual_answer_label": "",
                "manual_evidence_label": "",
                "manual_answer_note": "",
            }
        )
    return rows


def _prioritize_relations(relations: list[dict]) -> list[dict]:
    priority = {
        "supersedes": 0,
        "corrects": 1,
        "conflicts_without_resolution": 2,
        "supplements": 3,
    }
    return sorted(
        relations,
        key=lambda row: (
            priority.get(str(row.get("relation", "")), 9),
            str(row.get("new_event_id", "")),
            str(row.get("old_event_id", "")),
        ),
    )


def _prioritize_qa_items(qa_items: list[QAItem], errors_by_id: dict[str, dict]) -> list[QAItem]:
    type_priority = {
        "knowledge_update": 0,
        "temporal_reasoning": 1,
        "conflict_resolution": 2,
        "multi_session_reasoning": 3,
        "single_session_fact": 4,
        "abstention": 5,
    }
    return sorted(
        qa_items,
        key=lambda qa: (
            0 if qa.question_id in errors_by_id else 1,
            type_priority.get(qa.question_type, 9),
            qa.question_id,
        ),
    )


def _load_turn_texts(dataset_dir: Path) -> dict[str, str]:
    for filename in ("dialogues.jsonl", "memory_turns.jsonl", "dialogue_turns.jsonl"):
        path = dataset_dir / filename
        if path.exists():
            return {str(row["turn_id"]): str(row.get("text", "")) for row in read_jsonl(path)}
    return {}


def _joined_turn_text(turn_ids: list[str], turn_texts: dict[str, str]) -> str:
    return "\n".join(f"{turn_id}: {turn_texts.get(turn_id, '')}" for turn_id in turn_ids)


def _read_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _event_from_row(row: dict) -> Event:
    return Event(
        event_id=str(row["event_id"]),
        user_id=str(row.get("user_id", "")),
        time=str(row.get("time", "")),
        speaker=str(row.get("speaker", "user")),
        subject=str(row.get("subject", "")),
        event_type=row.get("event_type", "other"),
        content=str(row.get("content", "")),
        entities=[str(entity) for entity in row.get("entities", [])],
        evidence_turn_ids=[str(turn_id) for turn_id in row.get("evidence_turn_ids", [])],
        source_context_ids=[str(context_id) for context_id in row.get("source_context_ids", [])],
        importance=float(row.get("importance", 0.5)),
        superseded_by=row.get("superseded_by"),
        corrected_by=row.get("corrected_by"),
    )


def _answer_from_row(row: dict) -> Answer:
    return Answer(
        question_id=str(row["question_id"]),
        answer=str(row.get("answer", "")),
        evidence_turn_ids=[str(turn_id) for turn_id in row.get("evidence_turn_ids", [])],
        confidence=row.get("confidence", "low"),
        retrieved_event_ids=[str(event_id) for event_id in row.get("retrieved_event_ids", [])],
    )
