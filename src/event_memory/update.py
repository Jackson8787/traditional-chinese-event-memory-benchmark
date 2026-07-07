from __future__ import annotations

import json
import re

from .llm_client import ChatMessage, LLMClient
from .schema import Event, UpdateRelation, UpdateRelationType


VALID_RELATIONS: set[str] = {
    "supersedes",
    "corrects",
    "supplements",
    "conflicts_without_resolution",
    "unrelated",
}


class ConflictAwareUpdater:
    """Rule-constrained update relation detector for the MVP."""

    def apply(self, events: list[Event]) -> tuple[list[Event], list[UpdateRelation]]:
        memory: list[Event] = []
        relations: list[UpdateRelation] = []

        for event in events:
            candidates = self._candidates(event, memory)
            for old in candidates:
                relation = self._judge(event, old)
                if relation is None:
                    continue
                relations.append(relation)
                if relation.relation == "supersedes":
                    old.superseded_by = event.event_id
                elif relation.relation == "corrects":
                    old.corrected_by = event.event_id
            memory.append(event)

        return memory, relations

    def _candidates(self, event: Event, memory: list[Event]) -> list[Event]:
        event_entities = set(event.entities)
        candidates: list[Event] = []
        for old in memory:
            if old.user_id != event.user_id:
                continue
            if old.subject != event.subject:
                continue
            if old.event_type != event.event_type and event.event_type != "negated_fact":
                continue
            if event_entities & set(old.entities):
                candidates.append(old)
        return candidates

    def _judge(self, new: Event, old: Event) -> UpdateRelation | None:
        joined_evidence = ",".join(new.evidence_turn_ids)
        text = new.content
        old_text = old.content

        if new.event_type == "negated_fact" and set(new.entities) & set(old.entities):
            return UpdateRelation(
                new.event_id,
                old.event_id,
                "supersedes",
                f"新事件否定或取消舊事件：{old_text}",
                new.evidence_turn_ids,
            )

        if "改為" in text and set(new.entities) & set(old.entities):
            return UpdateRelation(
                new.event_id,
                old.event_id,
                "supersedes",
                f"新事件改變舊狀態：{old_text}",
                new.evidence_turn_ids,
            )

        if "新竹" in text and "台北" in old_text:
            return UpdateRelation(
                new.event_id,
                old.event_id,
                "corrects",
                f"新事件修正舊地點偏好，evidence={joined_evidence}",
                new.evidence_turn_ids,
            )

        if set(new.entities) & set(old.entities) and new.content != old.content:
            return UpdateRelation(
                new.event_id,
                old.event_id,
                "supplements",
                f"新事件補充舊事件：{old_text}",
                new.evidence_turn_ids,
            )

        return None


class LLMConflictAwareUpdater(ConflictAwareUpdater):
    """Rule-constrained candidate search plus LLM relation judging."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _judge(self, new: Event, old: Event) -> UpdateRelation | None:
        result = self.llm.chat(
            [
                ChatMessage(role="system", content=_UPDATE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=_pair_prompt(new, old)),
            ],
            max_completion_tokens=500,
        )
        row = _parse_relation_row(result.content)
        relation = str(row.get("relation", "unrelated")).strip()
        if relation not in VALID_RELATIONS:
            relation = "unrelated"
        if relation == "unrelated":
            return None
        reason = str(row.get("reason", "")).strip() or "LLM 判斷新舊事件存在更新關係"
        return UpdateRelation(
            new_event_id=new.event_id,
            old_event_id=old.event_id,
            relation=relation,  # type: ignore[arg-type]
            reason=reason,
            evidence_turn_ids=new.evidence_turn_ids,
        )


_UPDATE_SYSTEM_PROMPT = """你是繁體中文長期對話記憶的衝突/更新關係判斷器。
你會收到一個 old_event 和一個 new_event。請判斷 new_event 與 old_event 的關係。

只能輸出 JSON：
{
  "relation": "supersedes|corrects|supplements|conflicts_without_resolution|unrelated",
  "reason": "一句繁體中文理由"
}

定義：
- supersedes: 新事件取代舊事件，查目前狀態時應優先使用新事件。
- corrects: 新事件修正舊事件中的錯誤資訊。
- supplements: 新事件補充舊事件，不使舊事件失效。
- conflicts_without_resolution: 新舊事件矛盾，但無法判斷哪個有效。
- unrelated: 兩者沒有更新關係。

規則：
- 不要輸出 markdown。
- 不要猜測事件內容以外的資訊。
- 如果只是同一使用者不同主題，輸出 unrelated。
- 如果 new_event 明確包含「後來」、「不再」、「不想繼續」、「改成」、「改為」等更新語氣，且主題與 old_event 相同或重疊，請判斷為 supersedes，不要判斷為 conflicts_without_resolution。
- 只有在兩個事件矛盾但沒有任何時間順序或修正語氣能判斷新舊有效性時，才使用 conflicts_without_resolution。
"""


def _pair_prompt(new: Event, old: Event) -> str:
    return json.dumps(
        {
            "old_event": _event_payload(old),
            "new_event": _event_payload(new),
        },
        ensure_ascii=False,
        indent=2,
    )


def _event_payload(event: Event) -> dict:
    return {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "time": event.time,
        "subject": event.subject,
        "event_type": event.event_type,
        "content": event.content,
        "entities": event.entities,
        "evidence_turn_ids": event.evidence_turn_ids,
    }


def _parse_relation_row(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {"relation": "unrelated", "reason": ""}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"relation": "unrelated", "reason": ""}
    return parsed if isinstance(parsed, dict) else {"relation": "unrelated", "reason": ""}
