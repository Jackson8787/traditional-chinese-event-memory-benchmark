from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


EVENT_FILE = "pred_events.jsonl"
RELATION_FILES = [
    "pred_update_relations.jsonl",
    "relations.jsonl",
    "relation_filter_audit.jsonl",
]


def refresh_cached_memory_text(
    dataset_dir: str | Path,
    memory_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    memory_path = Path(memory_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if memory_path.resolve() != output_path.resolve():
        _copy_memory_tree(memory_path, output_path)

    turn_text_by_id = {
        str(row["turn_id"]): str(row.get("text") or "")
        for row in read_jsonl(dataset_path / "dialogues.jsonl")
        if row.get("turn_id")
    }

    event_count = _refresh_events(output_path / EVENT_FILE, turn_text_by_id)
    relation_count = 0
    for filename in RELATION_FILES:
        relation_count += _refresh_relations(output_path / filename)

    report = {
        "event_rows_refreshed": event_count,
        "relation_rows_refreshed": relation_count,
        "output_dir": str(output_path),
    }
    (output_path / "memory_text_refresh_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _copy_memory_tree(source: Path, target: Path) -> None:
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def _refresh_events(path: Path, turn_text_by_id: dict[str, str]) -> int:
    if not path.exists():
        return 0
    rows = read_jsonl(path)
    refreshed = 0
    for row in rows:
        evidence_texts = [
            turn_text_by_id[turn_id]
            for turn_id in row.get("evidence_turn_ids", [])
            if turn_id in turn_text_by_id and turn_text_by_id[turn_id]
        ]
        if not evidence_texts:
            continue
        row["content"] = _content_from_evidence(evidence_texts)
        row.pop("text_refresh_note", None)
        refreshed += 1
    write_jsonl(path, rows)
    return refreshed


def _refresh_relations(path: Path) -> int:
    if not path.exists():
        return 0
    rows = read_jsonl(path)
    for row in rows:
        relation = str(row.get("relation") or "unknown")
        row["reason"] = _relation_reason(relation)
        row.pop("text_refresh_note", None)
    write_jsonl(path, rows)
    return len(rows)


def _content_from_evidence(evidence_texts: list[str]) -> str:
    text = " / ".join(text.strip() for text in evidence_texts if text.strip())
    return f"使用者原文：{text}"


def _relation_reason(relation: str) -> str:
    return (
        f"本地刷新：保留既有 `{relation}` relation type；"
        "文字說明已依目前資料集刷新，未重新呼叫 LLM，仍需依 old/new evidence 人工確認。"
    )
