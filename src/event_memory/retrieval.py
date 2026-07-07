from __future__ import annotations

import re

from .event_structure import SLOT_CONFIDENCE_THRESHOLD, infer_event_structure, infer_query_slot
from .schema import DialogueTurn, Event, QueryIntent, RetrievalResult, UpdateRelation
from .text import overlap_score


class EventRetriever:
    def __init__(
        self,
        turns_by_id: dict[str, DialogueTurn],
        *,
        use_supersedes: bool = True,
        use_time_relevance: bool = True,
        use_entity_match: bool = True,
    ) -> None:
        self.turns_by_id = turns_by_id
        self.use_supersedes = use_supersedes
        self.use_time_relevance = use_time_relevance
        self.use_entity_match = use_entity_match

    def infer_intent(self, question: str) -> QueryIntent:
        asks_current = any(term in question for term in ["現在", "目前", "還在"])
        asks_history = any(term in question for term in ["一開始", "原本", "原先", "以前", "當時", "三月", "之前"])
        time_hint = "03" if "三月" in question else None
        return QueryIntent(
            asks_current=asks_current or not asks_history,
            asks_history=asks_history,
            requires_abstention=any(term in question for term in ["有沒有提過", "是否提過"]),
            time_hint=time_hint,
        )

    def retrieve(
        self,
        question: str,
        user_id: str,
        events: list[Event],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        intent = self.infer_intent(question)
        scored: list[RetrievalResult] = []

        for event in events:
            if event.user_id != user_id:
                continue
            evidence_turns = [self.turns_by_id[turn_id] for turn_id in event.evidence_turn_ids if turn_id in self.turns_by_id]
            score = self._score(question, intent, event)
            if score <= 0:
                continue
            scored.append(RetrievalResult(event, score, evidence_turns))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def _score(self, question: str, intent: QueryIntent, event: Event) -> float:
        embedding_similarity = overlap_score(question, event.content) * 0.65
        entity_match = 0.0
        if self.use_entity_match:
            entity_match = sum(0.18 for entity in event.entities if entity and entity in question)
        type_bonus = self._type_bonus(question, event)
        time_relevance = 0.0
        if self.use_time_relevance:
            if intent.time_hint and f"-{intent.time_hint}-" in event.time:
                time_relevance = 0.45
            elif intent.asks_current and not intent.asks_history:
                time_relevance = 0.08
        current_plan_bonus = _current_plan_bonus(question, event)
        slot_relevance = _slot_relevance_bonus(question, event)

        outdated_penalty = 0.0
        if self.use_supersedes and intent.asks_current and not intent.asks_history:
            if event.superseded_by or event.corrected_by:
                outdated_penalty = 0.75

        return (
            embedding_similarity
            + entity_match
            + type_bonus
            + time_relevance
            + current_plan_bonus
            + slot_relevance
            + event.importance * 0.12
            - outdated_penalty
        )

    def _type_bonus(self, question: str, event: Event) -> float:
        if any(term in question for term in ["接下來", "希望"]):
            if any(term in event.content for term in ["希望", "五月底", "六月前", "下一個時間點", "投"]):
                return 0.78
            if event.event_type == "future_event":
                return 0.46
            return -0.08
        if any(term in question for term in ["已經完成", "完成了什麼"]):
            return 0.58 if event.event_type == "completed_event" else -0.08
        if any(term in question for term in ["固定限制", "限制", "不能排", "時間限制"]):
            return 0.62 if event.event_type == "constraint" else -0.1
        if any(term in question for term in ["和誰", "合作", "照護", "關係"]):
            if event.event_type == "relationship":
                return 0.62
            if event.event_type == "personal_fact" and any(
                term in event.content for term in ["和", "室友", "一起", "合作", "照護", "照顧", "共用"]
            ):
                return 0.54
            return -0.1
        if any(term in question for term in ["地點", "哪裡"]):
            if event.event_type != "preference":
                return -0.1
            return 0.66 if _mentions_place(event.content) else -0.22
        if any(term in question for term in ["偏好", "穩定偏好", "方便"]):
            return 0.5 if event.event_type == "preference" else -0.08
        if any(term in question for term in ["主要計畫", "計畫", "準備", "做什麼", "找什麼"]):
            return 0.5 if event.event_type == "plan" else -0.08
        if any(term in question for term in ["時間", "會議", "排"]):
            return 0.35 if event.event_type == "constraint" else 0.0
        return 0.0


def _mentions_place(text: str) -> bool:
    return any(
        place in text
        for place in [
            "台北",
            "臺北",
            "新北",
            "桃園",
            "新竹",
            "台中",
            "臺中",
            "台南",
            "臺南",
            "高雄",
            "嘉義",
            "遠端",
            "線上",
        ]
    )


def _current_plan_bonus(question: str, event: Event) -> float:
    if "現在" not in question or "計畫" not in question or event.event_type not in {"plan", "future_event"}:
        return 0.0

    score = 0.0
    if any(term in event.content for term in ["改成", "改為", "轉向", "換成"]):
        score += 0.18
    if any(term in event.content for term in ["下一個時間點", "月底前", "五月底前", "接下來", "之後"]):
        score -= 0.18
    return score


def _slot_relevance_bonus(question: str, event: Event) -> float:
    query_slot = infer_query_slot(question)
    if query_slot is None:
        return 0.0
    event_structure = infer_event_structure(event)
    if event_structure.slot is None or event_structure.confidence < SLOT_CONFIDENCE_THRESHOLD:
        return 0.0
    if event_structure.slot == query_slot:
        return 0.55
    return -0.45


class EventRetrieverV2(EventRetriever):
    def __init__(
        self,
        turns_by_id: dict[str, DialogueTurn],
        relations: list[UpdateRelation],
        *,
        use_supersedes: bool = True,
        use_time_relevance: bool = True,
        use_entity_match: bool = True,
    ) -> None:
        super().__init__(
            turns_by_id,
            use_supersedes=use_supersedes,
            use_time_relevance=use_time_relevance,
            use_entity_match=use_entity_match,
        )
        self._related_event_ids: dict[str, list[str]] = {}
        for relation in relations:
            self._related_event_ids.setdefault(relation.new_event_id, []).append(relation.old_event_id)
            self._related_event_ids.setdefault(relation.old_event_id, []).append(relation.new_event_id)

    def infer_intent(self, question: str) -> QueryIntent:
        asks_numbered_memory = re.search(r"第(\d+)個記憶點", question) is not None
        asks_history = any(
            term in question
            for term in ["原本", "之前", "過去", "當時", "2026年3月", "3月", "從哪個狀態", "變到哪個狀態"]
        )
        asks_current = any(term in question for term in ["目前", "現在", "截至目前", "後來", "新舊資訊", "應採用"])
        if asks_numbered_memory:
            asks_history = False
            asks_current = False
        time_hint = "03" if any(term in question for term in ["2026年3月", "3月"]) else None
        return QueryIntent(
            asks_current=asks_current or (not asks_history and not asks_numbered_memory),
            asks_history=asks_history,
            requires_abstention=any(term in question for term in ["有沒有提過", "未公開", "無法判斷"]),
            time_hint=time_hint,
        )

    def retrieve(
        self,
        question: str,
        user_id: str,
        events: list[Event],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        intent = self.infer_intent(question)
        event_by_id = {event.event_id: event for event in events if event.user_id == user_id}
        scored: list[RetrievalResult] = []
        for event in event_by_id.values():
            evidence_turns = [self.turns_by_id[turn_id] for turn_id in event.evidence_turn_ids if turn_id in self.turns_by_id]
            score = self._score_v2(question, intent, event)
            if score <= 0:
                continue
            scored.append(RetrievalResult(event, score, evidence_turns))

        scored.sort(key=lambda result: result.score, reverse=True)
        return self._expand_and_diversify(scored, event_by_id, top_k)

    def _score_v2(self, question: str, intent: QueryIntent, event: Event) -> float:
        score = overlap_score(question, event.content) * 0.78
        numbered = re.search(r"第(\d+)個記憶點", question)
        if numbered:
            marker = f"第{numbered.group(1)}個記憶點"
            score += 1.15 if marker in event.content else -0.35
        if self.use_entity_match:
            score += sum(0.22 for entity in event.entities if entity and entity in question)
        score += self._type_bonus_v2(question, event)
        if self.use_time_relevance:
            score += self._time_relevance_v2(intent, event)
        score += _slot_relevance_bonus(question, event)
        if self.use_supersedes and intent.asks_current and not intent.asks_history:
            if event.superseded_by or event.corrected_by:
                score -= 0.85
        score += event.importance * 0.14
        return score

    def _type_bonus_v2(self, question: str, event: Event) -> float:
        if any(term in question for term in ["計畫", "安排", "活動"]):
            return 0.22 if event.event_type in {"plan", "future_event"} else 0.0
        if any(term in question for term in ["限制", "備份", "工具"]):
            return 0.22 if event.event_type == "constraint" else 0.0
        if any(term in question for term in ["地點", "偏好", "方式"]):
            return 0.22 if event.event_type == "preference" else 0.0
        if "合作" in question or "對象" in question:
            return 0.22 if event.event_type == "relationship" else 0.0
        return 0.0

    def _time_relevance_v2(self, intent: QueryIntent, event: Event) -> float:
        if intent.time_hint and f"-{intent.time_hint}-" in event.time:
            return 0.45
        if intent.asks_current and not intent.asks_history and event.time >= "2026-05-01":
            return 0.18
        if intent.asks_history and event.time < "2026-04-01":
            return 0.18
        return 0.0

    def _expand_and_diversify(
        self,
        scored: list[RetrievalResult],
        event_by_id: dict[str, Event],
        top_k: int,
    ) -> list[RetrievalResult]:
        selected: list[RetrievalResult] = []
        seen_events: set[str] = set()
        session_counts: dict[str, int] = {}
        target = max(top_k, 4)

        def add_result(result: RetrievalResult) -> None:
            if result.event.event_id in seen_events:
                return
            session_id = result.evidence_turns[0].session_id if result.evidence_turns else ""
            if session_id and session_counts.get(session_id, 0) >= 2 and len(selected) >= top_k:
                return
            selected.append(result)
            seen_events.add(result.event.event_id)
            if session_id:
                session_counts[session_id] = session_counts.get(session_id, 0) + 1

        for result in scored:
            add_result(result)
            if self.use_supersedes:
                for related_event_id in self._related_event_ids.get(result.event.event_id, []):
                    related = event_by_id.get(related_event_id)
                    if related is None:
                        continue
                    evidence_turns = [
                        self.turns_by_id[turn_id] for turn_id in related.evidence_turn_ids if turn_id in self.turns_by_id
                    ]
                    add_result(RetrievalResult(related, max(result.score - 0.03, 0.01), evidence_turns))
            if len(selected) >= target:
                break
        return selected[:target]
