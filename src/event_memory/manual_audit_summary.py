from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl


RELATION_LABELS = ["correct", "partially_correct", "incorrect", "unclear"]
ANSWER_LABELS = ["correct", "partially_correct", "incorrect", "abstention_correct", "unclear"]
EVIDENCE_LABELS = ["complete", "partial", "wrong", "not_needed", "unclear"]


def summarize_manual_audit(audit_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    audit_path = Path(audit_dir)
    relation_rows = _read_optional_jsonl(audit_path / "relation_precision_audit.jsonl")
    answer_rows = _read_optional_jsonl(audit_path / "answer_faithfulness_audit.jsonl")

    summary = {
        "relations": _summarize_relations(relation_rows),
        "answers": _summarize_answers(answer_rows),
        "evidence": _summarize_evidence(answer_rows),
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _summarize_relations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    pending = 0
    labeled = 0
    for row in rows:
        relation = str(row.get("relation") or "unknown")
        label = str(row.get("manual_relation_label") or "").strip()
        if not label:
            pending += 1
            continue
        labeled += 1
        by_type[relation][label] += 1

    return {
        "total_rows": len(rows),
        "labeled_count": labeled,
        "pending_count": pending,
        "by_type": {relation: _label_stats(counter, RELATION_LABELS) for relation, counter in sorted(by_type.items())},
    }


def _summarize_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_question_type: dict[str, Counter[str]] = defaultdict(Counter)
    pending = 0
    labeled = 0
    for row in rows:
        question_type = str(row.get("question_type") or "unknown")
        label = str(row.get("manual_answer_label") or "").strip()
        if not label:
            pending += 1
            continue
        labeled += 1
        by_question_type[question_type][label] += 1

    return {
        "total_rows": len(rows),
        "labeled_count": labeled,
        "pending_count": pending,
        "by_question_type": {
            question_type: _label_stats(counter, ANSWER_LABELS)
            for question_type, counter in sorted(by_question_type.items())
        },
    }


def _summarize_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    pending = 0
    for row in rows:
        label = str(row.get("manual_evidence_label") or "").strip()
        if not label:
            pending += 1
            continue
        labels[label] += 1

    stats = _label_stats(labels, EVIDENCE_LABELS)
    stats.pop("precision", None)
    stats["pending_count"] = pending
    labeled = stats["labeled_count"]
    stats["complete_rate"] = labels.get("complete", 0) / labeled if labeled else None
    stats["complete_or_partial_rate"] = (
        (labels.get("complete", 0) + labels.get("partial", 0)) / labeled if labeled else None
    )
    return {"labels": stats}


def _label_stats(counter: Counter[str], labels: list[str]) -> dict[str, Any]:
    total = sum(counter.values())
    stats: dict[str, Any] = {label: counter.get(label, 0) for label in labels}
    stats["labeled_count"] = total
    stats["precision"] = counter.get("correct", 0) / total if total else None
    return stats


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)
