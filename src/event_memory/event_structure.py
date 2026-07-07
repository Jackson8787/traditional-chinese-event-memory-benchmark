from __future__ import annotations

from dataclasses import dataclass

from .schema import Event


@dataclass(frozen=True)
class EventStructure:
    slot: str | None
    value: str | None
    confidence: float


SLOT_CONFIDENCE_THRESHOLD = 0.65

_PLACE_TERMS = ["地點", "場地", "討論室", "總圖", "系館", "客廳", "教室", "會議室"]
_TOOL_TERMS = ["工具", "表格", "Notion", "Google 試算表", "Canva", "GitHub Pages", "清單"]
_SOURCE_TERMS = ["來源", "公告", "職涯中心", "公司網站", "借用系統", "資訊"]
_REMINDER_TERMS = ["提醒", "再看一次", "確認", "檢查"]
_TIME_TERMS = ["時間", "上午", "下午", "晚上", "下週", "六月", "週六", "週日"]


def infer_event_structure(event: Event) -> EventStructure:
    text = f"{event.content} {' '.join(event.entities)}"
    slot = _infer_slot(text)
    if slot is None:
        return EventStructure(slot=None, value=None, confidence=0.0)
    value = _infer_value(event, slot)
    confidence = 0.85 if value else 0.68
    return EventStructure(slot=slot, value=value, confidence=confidence)


def infer_query_slot(text: str) -> str | None:
    return _infer_slot(text)


def compatible_memory_slots(old: Event, new: Event) -> bool:
    old_structure = infer_event_structure(old)
    new_structure = infer_event_structure(new)
    if old_structure.confidence < SLOT_CONFIDENCE_THRESHOLD:
        return True
    if new_structure.confidence < SLOT_CONFIDENCE_THRESHOLD:
        return True
    return old_structure.slot == new_structure.slot


def _infer_slot(text: str) -> str | None:
    if _contains_any(text, _SOURCE_TERMS):
        return "source"
    if _contains_any(text, _PLACE_TERMS):
        return "place"
    if _contains_any(text, _TOOL_TERMS):
        return "tool"
    if _contains_any(text, _REMINDER_TERMS):
        return "reminder"
    if _contains_any(text, _TIME_TERMS):
        return "time"
    return None


def _infer_value(event: Event, slot: str) -> str | None:
    candidates = list(reversed(event.entities))
    if slot == "place":
        return _first_matching(candidates, _PLACE_TERMS) or _last_specific_entity(event)
    if slot == "tool":
        return _preferred_entity_near_positive_cue(event, _TOOL_TERMS) or _first_matching(candidates, _TOOL_TERMS) or _last_specific_entity(event)
    if slot == "source":
        return _first_matching(candidates, _SOURCE_TERMS) or _last_specific_entity(event)
    if slot == "reminder":
        return _first_matching(candidates, _TIME_TERMS + _REMINDER_TERMS) or _last_specific_entity(event)
    if slot == "time":
        return _first_matching(candidates, _TIME_TERMS) or _last_specific_entity(event)
    return _last_specific_entity(event)


def _first_matching(candidates: list[str], terms: list[str]) -> str | None:
    for candidate in candidates:
        if _contains_any(candidate, terms):
            return candidate
    return None


def _preferred_entity_near_positive_cue(event: Event, terms: list[str]) -> str | None:
    positive_cues = ["偏好用", "比較偏好用", "主要整理工具改成", "改成", "改為", "換成", "應該是"]
    best: tuple[int, str] | None = None
    for entity in event.entities:
        if not _contains_any(entity, terms):
            continue
        entity_index = event.content.find(entity)
        if entity_index < 0:
            continue
        cue_distance = _nearest_preceding_cue_distance(event.content, entity_index, positive_cues)
        if cue_distance is None:
            continue
        if best is None or cue_distance < best[0]:
            best = (cue_distance, entity)
    return best[1] if best else None


def _nearest_preceding_cue_distance(text: str, entity_index: int, cues: list[str]) -> int | None:
    distances: list[int] = []
    prefix = text[:entity_index]
    for cue in cues:
        cue_index = prefix.rfind(cue)
        if cue_index >= 0:
            distances.append(entity_index - cue_index)
    return min(distances) if distances else None


def _last_specific_entity(event: Event) -> str | None:
    generic_terms = {"週末整理", "專題", "校園專題", "資料", "時間", "表格", "清單"}
    for entity in reversed(event.entities):
        if entity and entity not in generic_terms:
            return entity
    return event.entities[-1] if event.entities else None


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)
