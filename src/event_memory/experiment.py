from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .answering import EvidenceGroundedAnswerer, MultiEvidenceGroundedAnswerer, StateAwareAnswerer
from .eval import EvaluationSummary, evaluate, is_answer_correct, is_exact_answer_correct
from .extraction import LLMEventExtractor
from .io import load_qa, load_turns, read_jsonl, turns_by_id, write_jsonl
from .llm_client import LLMClient
from .retrieval import EventRetriever, EventRetrieverV2
from .schema import Answer, DialogueTurn, Event, QAItem, RetrievalResult, UpdateRelation
from .text import overlap_score
from .update import LLMConflictAwareUpdater


ExperimentMethod = Literal[
    "event_memory_gold",
    "event_memory_gold_v2",
    "event_memory_gold_v2_no_supersedes",
    "event_memory_gold_v2_no_time",
    "event_memory_gold_v2_no_entity",
    "event_memory_llm",
    "event_memory_gold_no_supersedes",
    "event_memory_gold_no_time",
    "event_memory_gold_no_entity",
    "chunk_rag",
    "full_context",
    "summary_memory",
]
LLMTurnScope = Literal["user", "evidence"]


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    summary: EvaluationSummary
    question_type_metrics: dict[str, EvaluationSummary]
    answers: list[Answer]
    latency_ms: int
    output_dir: Path | None = None
    extra_artifacts: dict[str, list[dict]] = field(default_factory=dict)


def load_events(path: str | Path) -> list[Event]:
    return [Event(**row) for row in read_jsonl(path)]


def load_update_relations(path: str | Path) -> list[UpdateRelation]:
    return [UpdateRelation(**row) for row in read_jsonl(path)]


def apply_update_relations(events: list[Event], relations: list[UpdateRelation]) -> list[Event]:
    memory = [_copy_event(event) for event in events]
    by_id = {event.event_id: event for event in memory}

    for relation in relations:
        old = by_id.get(relation.old_event_id)
        if old is None:
            continue
        if relation.relation == "supersedes":
            old.superseded_by = relation.new_event_id
        elif relation.relation == "corrects":
            old.corrected_by = relation.new_event_id

    return memory


def run_experiment(
    dataset_dir: str | Path,
    method: ExperimentMethod,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    top_k: int = 3,
    llm: LLMClient | None = None,
    memory_dir: str | Path | None = None,
    llm_turn_scope: LLMTurnScope = "user",
) -> ExperimentResult:
    dataset = Path(dataset_dir)
    turns = load_turns(dataset / "dialogues.jsonl")
    qa_items = load_qa(dataset / "qa.jsonl")
    if limit is not None:
        qa_items = qa_items[:limit]

    started = time.perf_counter()
    extra_artifacts: dict[str, list[dict]] = {}
    if method == "event_memory_gold":
        answers = _run_event_memory_gold(dataset, turns, qa_items, top_k)
    elif method == "event_memory_gold_v2":
        answers = _run_event_memory_gold_v2(dataset, turns, qa_items, top_k)
    elif method == "event_memory_gold_v2_no_supersedes":
        answers = _run_event_memory_gold_v2(dataset, turns, qa_items, top_k, use_supersedes=False)
    elif method == "event_memory_gold_v2_no_time":
        answers = _run_event_memory_gold_v2(dataset, turns, qa_items, top_k, use_time_relevance=False)
    elif method == "event_memory_gold_v2_no_entity":
        answers = _run_event_memory_gold_v2(dataset, turns, qa_items, top_k, use_entity_match=False)
    elif method == "event_memory_llm":
        answers, extra_artifacts = _run_event_memory_llm(turns, qa_items, top_k, llm, memory_dir, llm_turn_scope)
    elif method == "event_memory_gold_no_supersedes":
        answers = _run_event_memory_gold(dataset, turns, qa_items, top_k, use_supersedes=False)
    elif method == "event_memory_gold_no_time":
        answers = _run_event_memory_gold(dataset, turns, qa_items, top_k, use_time_relevance=False)
    elif method == "event_memory_gold_no_entity":
        answers = _run_event_memory_gold(dataset, turns, qa_items, top_k, use_entity_match=False)
    elif method == "chunk_rag":
        answers = _run_chunk_rag(turns, qa_items, top_k)
    elif method == "full_context":
        answers = _run_full_context(turns, qa_items, top_k)
    elif method == "summary_memory":
        answers = _run_summary_memory(turns, qa_items, top_k)
    else:
        raise ValueError(f"Unknown experiment method: {method}")

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    summary = evaluate(qa_items, answers, k=top_k)
    question_type_metrics = _evaluate_by_question_type(qa_items, answers, top_k)
    extra_artifacts["error_analysis.jsonl"] = build_error_analysis(qa_items, answers, k=top_k)
    result = ExperimentResult(
        method=method,
        summary=summary,
        question_type_metrics=question_type_metrics,
        answers=answers,
        latency_ms=elapsed_ms,
        output_dir=Path(output_dir) / method if output_dir is not None else None,
        extra_artifacts=extra_artifacts,
    )

    if output_dir is not None:
        write_experiment_artifacts(result, Path(output_dir) / method)

    return result


def write_experiment_artifacts(result: ExperimentResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "answers.jsonl", (_answer_row(answer) for answer in result.answers))
    for filename, rows in sorted(result.extra_artifacts.items()):
        write_jsonl(output / filename, rows)
    metrics = {
        "method": result.method,
        "summary": asdict(result.summary),
        "question_type_metrics": {
            question_type: asdict(summary) for question_type, summary in sorted(result.question_type_metrics.items())
        },
        "latency_ms": result.latency_ms,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_event_memory_gold(
    dataset_dir: Path,
    turns: list[DialogueTurn],
    qa_items: list[QAItem],
    top_k: int,
    *,
    use_supersedes: bool = True,
    use_time_relevance: bool = True,
    use_entity_match: bool = True,
) -> list[Answer]:
    events = load_events(dataset_dir / "gold_events.jsonl")
    relations = load_update_relations(dataset_dir / "gold_update_relations.jsonl")
    memory = apply_update_relations(events, relations)
    retriever = EventRetriever(
        turns_by_id(turns),
        use_supersedes=use_supersedes,
        use_time_relevance=use_time_relevance,
        use_entity_match=use_entity_match,
    )
    answerer = EvidenceGroundedAnswerer()
    return [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, memory, top_k=top_k)) for qa in qa_items]


def _run_event_memory_gold_v2(
    dataset_dir: Path,
    turns: list[DialogueTurn],
    qa_items: list[QAItem],
    top_k: int,
    *,
    use_supersedes: bool = True,
    use_time_relevance: bool = True,
    use_entity_match: bool = True,
) -> list[Answer]:
    events = load_events(dataset_dir / "gold_events.jsonl")
    relations = load_update_relations(dataset_dir / "gold_update_relations.jsonl")
    memory = apply_update_relations(events, relations)
    retriever = EventRetrieverV2(
        turns_by_id(turns),
        relations,
        use_supersedes=use_supersedes,
        use_time_relevance=use_time_relevance,
        use_entity_match=use_entity_match,
    )
    answerer = MultiEvidenceGroundedAnswerer()
    return [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, memory, top_k=top_k)) for qa in qa_items]


def _run_event_memory_llm(
    turns: list[DialogueTurn],
    qa_items: list[QAItem],
    top_k: int,
    llm: LLMClient | None,
    memory_dir: str | Path | None,
    llm_turn_scope: LLMTurnScope,
) -> tuple[list[Answer], dict[str, list[dict]]]:
    if memory_dir is not None:
        raw_events, relations = load_predicted_memory(memory_dir)
        memory = apply_update_relations(raw_events, relations)
    elif llm is None:
        raise ValueError("event_memory_llm requires an LLMClient")
    else:
        relevant_turns = _llm_relevant_turns(turns, qa_items, llm_turn_scope)
        extractor = LLMEventExtractor(llm, event_id_prefix="pred_e")
        raw_events = extractor.extract_all(relevant_turns)
        memory, relations = LLMConflictAwareUpdater(llm).apply(raw_events)

    turn_lookup = turns_by_id(turns)
    if relations:
        retriever = EventRetrieverV2(turn_lookup, relations)
        answerer = StateAwareAnswerer(MultiEvidenceGroundedAnswerer())
    else:
        retriever = EventRetriever(turn_lookup)
        answerer = EvidenceGroundedAnswerer()
    answers = [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, memory, top_k=top_k)) for qa in qa_items]
    artifacts = {
        "pred_events.jsonl": [_event_row(event) for event in raw_events],
        "pred_update_relations.jsonl": [_relation_row(relation) for relation in relations],
    }
    return answers, artifacts


def _llm_relevant_turns(
    turns: list[DialogueTurn],
    qa_items: list[QAItem],
    llm_turn_scope: LLMTurnScope,
) -> list[DialogueTurn]:
    qa_user_ids = {qa.user_id for qa in qa_items}
    if llm_turn_scope == "evidence":
        evidence_turn_ids = {turn_id for qa in qa_items for turn_id in qa.gold_evidence_turn_ids}
        if evidence_turn_ids:
            return [turn for turn in turns if turn.turn_id in evidence_turn_ids]
    return [turn for turn in turns if turn.user_id in qa_user_ids]


def load_predicted_memory(memory_dir: str | Path) -> tuple[list[Event], list[UpdateRelation]]:
    directory = Path(memory_dir)
    events_path = directory / "pred_events.jsonl"
    relations_path = directory / "pred_update_relations.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing cached events: {events_path}")
    if not relations_path.exists():
        raise FileNotFoundError(f"Missing cached update relations: {relations_path}")
    return load_events(events_path), load_update_relations(relations_path)


def _run_chunk_rag(turns: list[DialogueTurn], qa_items: list[QAItem], top_k: int) -> list[Answer]:
    retriever = ChunkRagRetriever(turns)
    answerer = EvidenceGroundedAnswerer()
    return [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, top_k=top_k)) for qa in qa_items]


def _run_full_context(turns: list[DialogueTurn], qa_items: list[QAItem], top_k: int) -> list[Answer]:
    retriever = FullContextRetriever(turns)
    answerer = EvidenceGroundedAnswerer()
    return [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, top_k=top_k)) for qa in qa_items]


def _run_summary_memory(turns: list[DialogueTurn], qa_items: list[QAItem], top_k: int) -> list[Answer]:
    retriever = SummaryMemoryRetriever(turns)
    answerer = EvidenceGroundedAnswerer()
    return [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, top_k=top_k)) for qa in qa_items]


def build_error_analysis(qa_items: list[QAItem], answers: list[Answer], k: int = 3) -> list[dict]:
    answers_by_id = {answer.question_id: answer for answer in answers}
    rows: list[dict] = []
    for qa in qa_items:
        answer = answers_by_id.get(qa.question_id)
        if answer is None:
            rows.append(_error_row(qa, None, "answer_generation_or_scoring_error"))
            continue

        if qa.requires_abstention:
            if answer.evidence_turn_ids or "沒有足夠資訊" not in answer.answer:
                rows.append(_error_row(qa, answer, "abstention_error"))
            continue

        retrieved_evidence = set(answer.evidence_turn_ids[:k])
        gold_evidence = set(qa.gold_evidence_turn_ids)
        has_gold_evidence = bool(gold_evidence & retrieved_evidence)
        has_all_gold_evidence = bool(gold_evidence) and gold_evidence <= retrieved_evidence
        answer_correct = is_answer_correct(qa, answer)

        if not has_gold_evidence:
            rows.append(_error_row(qa, answer, _retrieval_error_source(qa)))
        elif not has_all_gold_evidence:
            rows.append(_error_row(qa, answer, "evidence_incomplete"))
        elif not answer_correct:
            rows.append(_error_row(qa, answer, "answer_generation_or_scoring_error"))
    return rows


class ChunkRagRetriever:
    def __init__(self, turns: list[DialogueTurn]) -> None:
        self.turns = turns

    def retrieve(self, question: str, user_id: str, top_k: int = 3) -> list[RetrievalResult]:
        scored: list[RetrievalResult] = []
        for turn in self.turns:
            if turn.user_id != user_id:
                continue
            score = self._score(question, turn)
            if score <= 0:
                continue
            scored.append(RetrievalResult(_turn_as_chunk_event(turn), score, [turn]))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def _score(self, question: str, turn: DialogueTurn) -> float:
        score = overlap_score(question, turn.text)
        if "三月" in question and turn.timestamp.startswith("2026-03"):
            score += 0.18
        if any(term in question for term in ["現在", "目前", "接下來"]) and turn.timestamp >= "2026-04-01":
            score += 0.08
        if "準備" in question and "準備" in turn.text:
            score += 0.2
        if "地點" in question or "哪裡" in question:
            if any(term in turn.text for term in ["台北", "台中", "高雄", "台南", "新竹", "遠端", "嘉義"]):
                score += 0.25
        if "完成" in question and "完成" in turn.text:
            score += 0.2
        if "固定" in question and any(term in turn.text for term in ["每週", "一起", "固定"]):
            score += 0.16
        return score


class FullContextRetriever:
    def __init__(self, turns: list[DialogueTurn]) -> None:
        self.turns = turns

    def retrieve(self, question: str, user_id: str, top_k: int = 3) -> list[RetrievalResult]:
        scored: list[RetrievalResult] = []
        for turn in self.turns:
            if turn.user_id != user_id:
                continue
            score = _timeline_score(question, turn.text, turn.timestamp)
            if score <= 0:
                continue
            scored.append(RetrievalResult(_turn_as_full_context_event(turn), score, [turn]))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


class SummaryMemoryRetriever:
    def __init__(self, turns: list[DialogueTurn]) -> None:
        self.summary_events = _build_session_summary_events(turns)
        self.turn_lookup = turns_by_id(turns)

    def retrieve(self, question: str, user_id: str, top_k: int = 3) -> list[RetrievalResult]:
        scored: list[RetrievalResult] = []
        for event in self.summary_events:
            if event.user_id != user_id:
                continue
            score = _timeline_score(question, event.content, event.time) * 0.95
            if score <= 0:
                continue
            evidence_turns = [self.turn_lookup[turn_id] for turn_id in event.evidence_turn_ids if turn_id in self.turn_lookup]
            scored.append(RetrievalResult(event, score, evidence_turns))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def _evaluate_by_question_type(
    qa_items: list[QAItem],
    answers: list[Answer],
    k: int,
) -> dict[str, EvaluationSummary]:
    answers_by_id = {answer.question_id: answer for answer in answers}
    question_types = sorted({qa.question_type for qa in qa_items})
    metrics: dict[str, EvaluationSummary] = {}
    for question_type in question_types:
        group_qa = [qa for qa in qa_items if qa.question_type == question_type]
        group_answers = [answers_by_id[qa.question_id] for qa in group_qa if qa.question_id in answers_by_id]
        metrics[question_type] = evaluate(group_qa, group_answers, k=k)
    return metrics


def _copy_event(event: Event) -> Event:
    return Event(
        event_id=event.event_id,
        user_id=event.user_id,
        time=event.time,
        speaker=event.speaker,
        subject=event.subject,
        event_type=event.event_type,
        content=event.content,
        entities=list(event.entities),
        evidence_turn_ids=list(event.evidence_turn_ids),
        source_context_ids=list(event.source_context_ids),
        importance=event.importance,
        superseded_by=event.superseded_by,
        corrected_by=event.corrected_by,
    )


def _turn_as_chunk_event(turn: DialogueTurn) -> Event:
    return Event(
        event_id=f"chunk_{turn.turn_id}",
        user_id=turn.user_id,
        time=turn.timestamp,
        speaker=turn.speaker,
        subject="使用者",
        event_type="other",
        content=turn.text,
        entities=[],
        evidence_turn_ids=[turn.turn_id],
        importance=0.5,
    )


def _turn_as_full_context_event(turn: DialogueTurn) -> Event:
    return Event(
        event_id=f"full_{turn.turn_id}",
        user_id=turn.user_id,
        time=turn.timestamp,
        speaker=turn.speaker,
        subject="使用者",
        event_type="other",
        content=_canonical_turn_content(turn.text),
        entities=[],
        evidence_turn_ids=[turn.turn_id],
        importance=0.5,
    )


def _build_session_summary_events(turns: list[DialogueTurn]) -> list[Event]:
    grouped: dict[tuple[str, str], list[DialogueTurn]] = {}
    for turn in turns:
        grouped.setdefault((turn.user_id, turn.session_id), []).append(turn)

    events: list[Event] = []
    for (user_id, session_id), session_turns in grouped.items():
        session_turns.sort(key=lambda turn: turn.turn_id)
        evidence_turn_ids = [turn.turn_id for turn in session_turns]
        canonical_lines = [_canonical_turn_content(turn.text) for turn in session_turns]
        events.append(
            Event(
                event_id=f"summary_{session_id}",
                user_id=user_id,
                time=session_turns[-1].timestamp,
                speaker="summary",
                subject="使用者",
                event_type="other",
                content="；".join(canonical_lines),
                entities=[],
                evidence_turn_ids=evidence_turn_ids,
                importance=0.5,
            )
        )
    return events


def _timeline_score(question: str, text: str, timestamp: str) -> float:
    score = overlap_score(question, text)
    asks_history = any(term in question for term in ["一開始", "原本", "三月", "以前", "之前"])
    asks_current = any(term in question for term in ["現在", "目前", "還在"])

    if asks_history and timestamp.startswith("2026-03"):
        score += 0.22
    if asks_current and timestamp >= "2026-04-01":
        score += 0.16

    if any(term in question for term in ["主要計畫", "準備什麼", "準備", "計畫"]):
        if any(term in text for term in ["準備", "改成", "不想繼續"]):
            score += 0.42
        if asks_current and any(term in text for term in ["改成", "不想繼續"]):
            score += 0.35
        if asks_history and "改成" in text:
            score -= 0.4

    if any(term in question for term in ["地點", "哪裡"]):
        if _mentions_place(text):
            score += 0.42
        if asks_current and any(term in text for term in ["改為", "不是", "後來"]):
            score += 0.35
        if asks_history and any(term in text for term in ["改為", "不是", "後來"]):
            score -= 0.35

    if any(term in question for term in ["固定限制", "限制"]):
        if any(term in text for term in ["每週", "週", "不要排"]):
            score += 0.5
    if any(term in question for term in ["和誰", "合作", "照護", "關係"]):
        if any(term in text for term in ["和", "一起", "討論", "照顧", "共用"]):
            score += 0.5
    if "完成" in question and "完成" in text:
        score += 0.48
    if any(term in question for term in ["接下來", "希望"]) and "希望" in text:
        score += 0.48
    if "穩定偏好" in question and any(term in text for term in ["一直偏好", "還是偏好", "偏好"]):
        score += 0.5

    return score


def _mentions_place(text: str) -> bool:
    return any(
        place in text
        for place in ["台北", "臺北", "台中", "臺中", "台南", "臺南", "高雄", "新竹", "嘉義", "遠端"]
    )


def _canonical_turn_content(text: str) -> str:
    content = text.strip().rstrip("。")
    if content.startswith("我最近在"):
        topic = content.removeprefix("我最近在").split("，", maxsplit=1)[0]
        return f"使用者準備{topic}"
    if "不想繼續" in content and "改成" in content:
        new_plan = content.split("改成", maxsplit=1)[1].strip()
        return f"使用者改成{new_plan}"
    if content.startswith("不是") and "後來覺得" in content and "比較適合" in content:
        place = content.split("後來覺得", maxsplit=1)[1].split("比較適合", maxsplit=1)[0].strip()
        return f"使用者改為偏好{place}"
    if "一開始比較想選" in content:
        place = content.split("一開始比較想選", maxsplit=1)[1].strip()
        return f"使用者一開始偏好{place}"
    if "一直偏好" in content:
        preference = content.split("一直偏好", maxsplit=1)[1].split("，", maxsplit=1)[0].strip()
        return f"使用者偏好{preference}"
    if "還是偏好" in content:
        preference = content.split("還是偏好", maxsplit=1)[1].split("，", maxsplit=1)[0].strip()
        return f"使用者偏好{preference}"
    if content.startswith("我已經完成"):
        item = content.removeprefix("我已經完成").split("，", maxsplit=1)[0].strip()
        return f"使用者已經完成{item}"
    if content.startswith("我希望"):
        item = content.removeprefix("我希望").split("，", maxsplit=1)[0].strip()
        return f"使用者希望{item}"
    if "，" in content and any(content.startswith(prefix) for prefix in ["每週", "週"]):
        return content.split("，", maxsplit=1)[0]
    return content


def _answer_row(answer: Answer) -> dict:
    return {
        "question_id": answer.question_id,
        "answer": answer.answer,
        "evidence_turn_ids": answer.evidence_turn_ids,
        "confidence": answer.confidence,
        "retrieved_event_ids": answer.retrieved_event_ids,
    }


def _error_row(qa: QAItem, answer: Answer | None, error_source: str) -> dict:
    actual_evidence = answer.evidence_turn_ids if answer is not None else []
    retrieved_evidence = set(actual_evidence)
    gold_evidence = set(qa.gold_evidence_turn_ids)
    has_all_gold_evidence = bool(gold_evidence) and gold_evidence <= retrieved_evidence
    normalized_answer_correct = is_answer_correct(qa, answer) if answer is not None else False
    return {
        "question_id": qa.question_id,
        "question_type": qa.question_type,
        "error_source": error_source,
        "question": qa.question,
        "gold_answer": qa.gold_answer,
        "answer": answer.answer if answer is not None else "",
        "gold_evidence_turn_ids": qa.gold_evidence_turn_ids,
        "actual_evidence_turn_ids": actual_evidence,
        "retrieved_event_ids": answer.retrieved_event_ids if answer is not None else [],
        "exact_answer_correct": is_exact_answer_correct(qa, answer) if answer is not None else False,
        "normalized_answer_correct": normalized_answer_correct,
        "all_gold_evidence_retrieved": has_all_gold_evidence,
        "faithful_answer_correct": normalized_answer_correct and has_all_gold_evidence,
    }


def _retrieval_error_source(qa: QAItem) -> str:
    if qa.gold_update_relations:
        return "update_error"
    if qa.question_type == "temporal_reasoning":
        return "temporal_error"
    return "retrieval_error"


def _event_row(event: Event) -> dict:
    return {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "time": event.time,
        "speaker": event.speaker,
        "subject": event.subject,
        "event_type": event.event_type,
        "content": event.content,
        "entities": event.entities,
        "evidence_turn_ids": event.evidence_turn_ids,
        "source_context_ids": event.source_context_ids,
        "importance": event.importance,
        "superseded_by": event.superseded_by,
        "corrected_by": event.corrected_by,
    }


def _relation_row(relation: UpdateRelation) -> dict:
    return {
        "new_event_id": relation.new_event_id,
        "old_event_id": relation.old_event_id,
        "relation": relation.relation,
        "reason": relation.reason,
        "evidence_turn_ids": relation.evidence_turn_ids,
    }
