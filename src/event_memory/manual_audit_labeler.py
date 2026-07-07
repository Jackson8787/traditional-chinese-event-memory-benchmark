from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl
from .manual_audit_summary import summarize_manual_audit


UPDATE_CUES = ["改成", "改為", "換成", "調整為", "舊做法", "不再", "取消", "延期", "提前"]
CORRECTION_CUES = ["說錯", "記錯", "更正", "修正", "不是", "應該是", "其實是", "弄錯"]
CONFLICT_CUES = ["衝突", "撞到", "不能同時", "沒辦法同時", "互斥", "只能選"]


def auto_label_manual_audit(audit_dir: str | Path) -> dict[str, Any]:
    audit_path = Path(audit_dir)
    relation_path = audit_path / "relation_precision_audit.jsonl"
    answer_path = audit_path / "answer_faithfulness_audit.jsonl"

    relation_rows = _read_optional_jsonl(relation_path)
    answer_rows = _read_optional_jsonl(answer_path)
    new_relation_labeled = _label_relation_rows(relation_rows)
    new_answer_labeled = _label_answer_rows(answer_rows)

    if relation_rows:
        write_jsonl(relation_path, relation_rows)
    if answer_rows:
        write_jsonl(answer_path, answer_rows)

    summary_path = audit_path / "manual_audit_summary.json"
    summary = summarize_manual_audit(audit_path, output_path=summary_path)
    report = {
        "relation_labeled_count": summary["relations"]["labeled_count"],
        "answer_labeled_count": summary["answers"]["labeled_count"],
        "new_relation_labeled_count": new_relation_labeled,
        "new_answer_labeled_count": new_answer_labeled,
        "summary_path": str(summary_path),
        "summary": summary,
    }
    (audit_path / "manual_audit_auto_label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _label_relation_rows(rows: list[dict[str, Any]]) -> int:
    labeled = 0
    for row in rows:
        if row.get("manual_relation_label"):
            continue
        label, note = _relation_label(row)
        row["manual_relation_label"] = label
        row["manual_relation_note"] = note
        labeled += 1
    return labeled


def _relation_label(row: dict[str, Any]) -> tuple[str, str]:
    relation = str(row.get("relation") or "")
    old_slot = row.get("old_slot")
    new_slot = row.get("new_slot")
    slot_match = bool(row.get("slot_match")) or (old_slot is not None and old_slot == new_slot)
    old_value = _clean(row.get("old_value"))
    new_value = _clean(row.get("new_value"))
    combined = " ".join(
        _clean(row.get(field))
        for field in ["old_event_content", "new_event_content", "old_evidence_text", "new_evidence_text", "relation_reason"]
    )

    if relation in {"supersedes", "corrects"}:
        if not slot_match:
            return "incorrect", "assistant_auto_label_v1: old/new memory slots differ."
        if old_value and new_value and old_value == new_value:
            return "incorrect", "assistant_auto_label_v1: old/new values are the same, so this is not a state replacement."
        if relation == "corrects" and not _has_any(combined, CORRECTION_CUES):
            return "incorrect", "assistant_auto_label_v1: corrects relation lacks an explicit correction cue."
        if relation == "supersedes" and not _has_any(combined, UPDATE_CUES + CORRECTION_CUES):
            return "partially_correct", "assistant_auto_label_v1: related same-slot events, but replacement cue is weak."
        return "correct", "assistant_auto_label_v1: same slot, distinct values, and explicit update/correction cue."

    if relation == "conflicts_without_resolution":
        if _has_any(combined, CONFLICT_CUES):
            return "correct", "assistant_auto_label_v1: explicit conflict cue found."
        return "incorrect", "assistant_auto_label_v1: no explicit conflict cue; likely coexisting plans or weak relation."

    if relation == "supplements":
        if not slot_match and old_slot and new_slot:
            return "partially_correct", "assistant_auto_label_v1: related content, but different slots make the supplement weak."
        return "correct", "assistant_auto_label_v1: treated as a conservative supplementary relation."

    return "unclear", "assistant_auto_label_v1: unsupported relation type."


def _label_answer_rows(rows: list[dict[str, Any]]) -> int:
    labeled = 0
    for row in rows:
        if row.get("manual_answer_label") and row.get("manual_evidence_label"):
            continue
        answer_label, evidence_label, note = _answer_labels(row)
        row["manual_answer_label"] = answer_label
        row["manual_evidence_label"] = evidence_label
        row["manual_answer_note"] = note
        labeled += 1
    return labeled


def _answer_labels(row: dict[str, Any]) -> tuple[str, str, str]:
    question_type = str(row.get("question_type") or "")
    predicted = _clean(row.get("predicted_answer"))
    normalized_correct = bool(row.get("automatic_normalized_correct"))
    evidence_covered = bool(row.get("automatic_gold_evidence_covered"))
    missing = row.get("missing_gold_evidence_turn_ids") or []

    if question_type == "abstention" or "沒有足夠資訊" in predicted or "無法從對話判斷" in predicted:
        if not row.get("predicted_evidence_turn_ids"):
            return "abstention_correct", "not_needed", "assistant_auto_label_v1: abstention with no evidence."
        return "partially_correct", "partial", "assistant_auto_label_v1: abstention-like answer includes extra evidence."

    if normalized_correct and evidence_covered:
        return "correct", "complete", "assistant_auto_label_v1: normalized answer correct and all gold evidence retrieved."
    if normalized_correct and missing:
        return "partially_correct", "partial", "assistant_auto_label_v1: answer overlaps gold, but some gold evidence is missing."
    if evidence_covered:
        return "incorrect", "complete", "assistant_auto_label_v1: evidence is present but answer text is not correct."
    if row.get("predicted_evidence_turn_ids"):
        return "incorrect", "wrong", "assistant_auto_label_v1: answer is incorrect and retrieved evidence misses the gold evidence."
    return "incorrect", "wrong", "assistant_auto_label_v1: answer is incorrect and no useful evidence was retrieved."


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _has_any(text: str, cues: list[str]) -> bool:
    return any(cue in text for cue in cues)


def _clean(value: Any) -> str:
    return str(value or "").strip()
