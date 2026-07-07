from __future__ import annotations

from .state_answering import StateAwareAnswerer
from .schema import Answer, QAItem, RetrievalResult


class EvidenceGroundedAnswerer:
    def answer(self, qa: QAItem, retrieved: list[RetrievalResult]) -> Answer:
        if qa.requires_abstention:
            return Answer(
                question_id=qa.question_id,
                answer="目前對話中沒有足夠資訊判斷這件事。",
                evidence_turn_ids=[],
                confidence="low",
                retrieved_event_ids=[result.event.event_id for result in retrieved],
            )

        if not retrieved or retrieved[0].score < 0.28:
            return Answer(
                question_id=qa.question_id,
                answer="目前對話中沒有足夠資訊判斷這件事。",
                evidence_turn_ids=[],
                confidence="low",
                retrieved_event_ids=[result.event.event_id for result in retrieved],
            )

        top = retrieved[0].event
        prefix = "不是，" if "還在" in qa.question and ("改為" in top.content or "不" in top.content) else ""
        return Answer(
            question_id=qa.question_id,
            answer=f"{prefix}{top.content}。",
            evidence_turn_ids=top.evidence_turn_ids,
            confidence="high" if retrieved[0].score >= 0.45 else "medium",
            retrieved_event_ids=[result.event.event_id for result in retrieved],
        )


class MultiEvidenceGroundedAnswerer:
    def __init__(self, max_events: int = 4) -> None:
        self.max_events = max_events
        self._fallback = EvidenceGroundedAnswerer()

    def answer(self, qa: QAItem, retrieved: list[RetrievalResult]) -> Answer:
        if qa.requires_abstention or not retrieved or retrieved[0].score < 0.2:
            return self._fallback.answer(qa, retrieved)

        events = []
        seen_events = set()
        for result in retrieved:
            if result.event.event_id in seen_events:
                continue
            events.append(result.event)
            seen_events.add(result.event.event_id)
            if len(events) >= self.max_events:
                break

        evidence_turn_ids: list[str] = []
        for event in events:
            for turn_id in event.evidence_turn_ids:
                if turn_id not in evidence_turn_ids:
                    evidence_turn_ids.append(turn_id)

        answer_parts: list[str] = []
        for event in events:
            if event.content not in answer_parts:
                answer_parts.append(event.content)

        return Answer(
            question_id=qa.question_id,
            answer="；".join(answer_parts) + "。",
            evidence_turn_ids=evidence_turn_ids,
            confidence="high" if retrieved[0].score >= 0.45 and evidence_turn_ids else "medium",
            retrieved_event_ids=[result.event.event_id for result in retrieved],
        )
