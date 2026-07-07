from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .event_structure import compatible_memory_slots
from .io import read_jsonl, write_jsonl
from .schema import Event
from .text import overlap_score, tokenize


UPDATE_CUES = {
    "改成",
    "改為",
    "改到",
    "換成",
    "換到",
    "後來",
    "現在改",
    "不再",
    "取消",
    "延期",
    "提前",
}
CORRECTION_CUES = {
    "說錯",
    "記錯",
    "更正",
    "修正",
    "不是",
    "應該是",
    "其實是",
    "弄錯",
}
CONFLICT_CUES = {
    "衝突",
    "撞到",
    "不能同時",
    "沒辦法同時",
    "互斥",
    "只能選",
}
MAX_SHARED_ENTITY_FREQUENCY_FOR_SUPPLEMENT = 30
FILTER_VERSION = "slot_aware_conservative_v2"


def filter_predicted_memory(memory_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(memory_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events_path = source / "pred_events.jsonl"
    relations_path = source / "pred_update_relations.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing cached events: {events_path}")
    if not relations_path.exists():
        raise FileNotFoundError(f"Missing cached update relations: {relations_path}")

    event_rows = read_jsonl(events_path)
    relation_rows = read_jsonl(relations_path)
    events = {row["event_id"]: Event(**row) for row in event_rows}
    entity_frequencies = Counter(entity for event in events.values() for entity in event.entities)

    kept: list[dict] = []
    removed: list[dict] = []
    removed_by_relation: Counter[str] = Counter()
    kept_by_relation: Counter[str] = Counter()

    for row in relation_rows:
        keep, reason = _should_keep_relation(row, events, entity_frequencies)
        relation_type = str(row.get("relation", "unknown"))
        if keep:
            kept.append(row)
            kept_by_relation[relation_type] += 1
        else:
            removed_by_relation[relation_type] += 1
            removed.append({**row, "filter_reason": reason})

    write_jsonl(output / "pred_events.jsonl", event_rows)
    write_jsonl(output / "pred_update_relations.jsonl", kept)
    write_jsonl(output / "relation_filter_audit.jsonl", removed)
    _copy_sidecars(source, output)

    report: dict[str, Any] = {
        "input_event_count": len(event_rows),
        "input_relation_count": len(relation_rows),
        "kept_relation_count": len(kept),
        "removed_relation_count": len(removed),
        "kept_by_relation": dict(sorted(kept_by_relation.items())),
        "removed_by_relation": dict(sorted(removed_by_relation.items())),
        "filter_version": FILTER_VERSION,
        "max_shared_entity_frequency_for_supplement": MAX_SHARED_ENTITY_FREQUENCY_FOR_SUPPLEMENT,
    }
    (output / "relation_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _rewrite_stats_relation_count(output, len(kept))
    return report


def _should_keep_relation(
    row: dict,
    events: dict[str, Event],
    entity_frequencies: Counter[str],
) -> tuple[bool, str]:
    old = events.get(str(row.get("old_event_id", "")))
    new = events.get(str(row.get("new_event_id", "")))
    if old is None or new is None:
        return False, "missing_event"
    if old.user_id != new.user_id:
        return False, "different_user"

    relation = row.get("relation")
    related = _is_related(old, new)
    if relation == "supersedes":
        if not _compatible_event_type(old, new):
            return False, "incompatible_event_type"
        if not related:
            return False, "not_related_enough"
        if not compatible_memory_slots(old, new):
            return False, "different_memory_slot"
        if not _has_any_cue(new.content, cues=UPDATE_CUES | CORRECTION_CUES):
            return False, "no_explicit_update_cue_in_new_event"
        if _is_restatement(old, new):
            return False, "likely_restatement"
        return True, "kept"
    if relation == "corrects":
        if not related:
            return False, "not_related_enough"
        if not compatible_memory_slots(old, new):
            return False, "different_memory_slot"
        if not _has_any_cue(new.content, cues=CORRECTION_CUES):
            return False, "no_explicit_correction_cue_in_new_event"
        return True, "kept"
    if relation == "supplements":
        if not _is_strong_supplement(old, new, entity_frequencies):
            return False, "weak_supplement_relation"
        return True, "kept"
    if relation == "conflicts_without_resolution":
        if not related:
            return False, "not_related_enough"
        if not _has_any_cue(new.content, old.content, cues=CONFLICT_CUES):
            return False, "no_explicit_conflict_cue"
        return True, "kept"
    return False, "unsupported_or_unrelated_relation"


def _is_strong_supplement(old: Event, new: Event, entity_frequencies: Counter[str]) -> bool:
    score = overlap_score(old.content, new.content)
    shared_entities = set(old.entities) & set(new.entities)
    has_specific_shared_entity = any(
        entity_frequencies[entity] <= MAX_SHARED_ENTITY_FREQUENCY_FOR_SUPPLEMENT for entity in shared_entities
    )
    if has_specific_shared_entity and score >= 0.12:
        return True
    return score >= 0.55


def _compatible_event_type(old: Event, new: Event) -> bool:
    if old.event_type == new.event_type:
        return True
    compatible = {old.event_type, new.event_type}
    return compatible <= {"plan", "future_event", "constraint", "preference", "other"}


def _is_related(old: Event, new: Event) -> bool:
    old_entities = set(old.entities)
    new_entities = set(new.entities)
    if old_entities & new_entities:
        return True
    if _entity_token_overlap(old_entities, new_entities) >= 0.35:
        return True
    return overlap_score(old.content, new.content) >= 0.22


def _entity_token_overlap(old_entities: set[str], new_entities: set[str]) -> float:
    old_tokens = {token for entity in old_entities for token in tokenize(entity)}
    new_tokens = {token for entity in new_entities for token in tokenize(entity)}
    if not old_tokens or not new_tokens:
        return 0.0
    return len(old_tokens & new_tokens) / max(len(old_tokens), len(new_tokens))


def _is_restatement(old: Event, new: Event) -> bool:
    old_compact = _compact(old.content)
    new_compact = _compact(new.content)
    if old_compact == new_compact:
        return True
    if old_compact and new_compact and (old_compact in new_compact or new_compact in old_compact):
        return not _has_any_cue(new.content, "", cues=UPDATE_CUES | CORRECTION_CUES)
    return overlap_score(old.content, new.content) >= 0.9


def _has_any_cue(*texts: str, cues: set[str]) -> bool:
    combined = " ".join(text for text in texts if text)
    return any(cue in combined for cue in cues)


def _compact(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace() and ch not in "，。！？；：,.!?;:")


def _copy_sidecars(source: Path, output: Path) -> None:
    skip = {
        "pred_events.jsonl",
        "pred_update_relations.jsonl",
        "relation_filter_audit.jsonl",
        "relation_filter_report.json",
    }
    for path in source.iterdir():
        if path.name in skip or path.is_dir():
            continue
        target = output / path.name
        if path.is_file():
            shutil.copy2(path, target)


def _rewrite_stats_relation_count(output: Path, relation_count: int) -> None:
    stats_path = output / "memory_build_stats.json"
    if not stats_path.exists():
        return
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    stats["predicted_relation_count_raw"] = stats.get("predicted_relation_count")
    stats["predicted_relation_count"] = relation_count
    stats["relation_filter"] = FILTER_VERSION
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
