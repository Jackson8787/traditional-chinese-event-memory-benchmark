from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .event_structure import infer_event_structure
from .io import read_jsonl, write_jsonl
from .schema import Event


UPDATE_CUES = {
    "改成",
    "改為",
    "改到",
    "換成",
    "換到",
    "後來",
    "現在",
    "不再",
    "取消",
    "修正",
    "更正",
    "不是",
    "應該是",
}


def structure_predicted_memory(memory_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(memory_dir)
    output = Path(output_dir)
    events_path = source / "pred_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing cached events: {events_path}")

    event_rows = read_jsonl(events_path)
    structured_rows = [structure_event_row(row) for row in event_rows]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "pred_events_structured.jsonl", structured_rows)
    _copy_memory_sidecars(source, output)

    slot_counts = Counter(row["memory_slot"] or "unknown" for row in structured_rows)
    structured_count = sum(1 for row in structured_rows if row["memory_slot"])
    report: dict[str, Any] = {
        "event_count": len(structured_rows),
        "structured_event_count": structured_count,
        "structured_event_ratio": round(structured_count / len(structured_rows), 4) if structured_rows else 0.0,
        "slot_counts": dict(sorted(slot_counts.items())),
        "artifact": "pred_events_structured.jsonl",
    }
    (output / "event_structure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def structure_event_row(row: dict) -> dict:
    event = _event_from_row(row)
    structure = infer_event_structure(event)
    is_state_update = _has_update_cue(event.content) or event.superseded_by is not None or event.corrected_by is not None
    return {
        **row,
        "memory_slot": structure.slot,
        "memory_value": structure.value,
        "structure_confidence": structure.confidence,
        "valid_from": event.time,
        "valid_until": None,
        "is_state_update": is_state_update,
        "structure_method": "heuristic_slot_value_v1",
    }


def _has_update_cue(text: str) -> bool:
    return any(cue in text for cue in UPDATE_CUES)


def _copy_memory_sidecars(source: Path, output: Path) -> None:
    skip = {"pred_events_structured.jsonl", "event_structure_report.json"}
    for path in source.iterdir():
        if path.name in skip or path.is_dir():
            continue
        target = output / path.name
        if path.is_file() and not target.exists():
            shutil.copy2(path, target)


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
