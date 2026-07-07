from __future__ import annotations

from dataclasses import dataclass
import re

from .schema import Answer, QAItem
from .text import tokenize


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    answer_accuracy: float
    evidence_recall_at_k: float
    abstention_accuracy: float
    update_accuracy: float = 0.0
    temporal_accuracy: float = 0.0
    average_token_cost: float = 0.0
    exact_answer_accuracy: float = 0.0
    all_evidence_recall_at_k: float = 0.0
    faithful_answer_accuracy: float = 0.0
    strict_state_accuracy: float = 0.0


def evaluate(qa_items: list[QAItem], answers: list[Answer], k: int = 3) -> EvaluationSummary:
    qa_by_id = {qa.question_id: qa for qa in qa_items}
    answer_correct = 0
    exact_answer_correct = 0
    evidence_correct = 0
    abstention_correct = 0
    update_total = 0
    update_correct = 0
    temporal_total = 0
    temporal_correct = 0
    all_evidence_correct = 0
    faithful_answer_correct = 0
    strict_state_total = 0
    strict_state_correct = 0
    token_cost = 0

    for answer in answers:
        qa = qa_by_id[answer.question_id]
        token_cost += _token_proxy(qa.question) + _token_proxy(answer.answer)
        if qa.requires_abstention:
            if not answer.evidence_turn_ids and "沒有足夠資訊" in answer.answer:
                abstention_correct += 1
                answer_correct += 1
                exact_answer_correct += 1
            continue

        if is_exact_answer_correct(qa, answer):
            exact_answer_correct += 1

        if is_answer_correct(qa, answer):
            answer_correct += 1

        retrieved_evidence = set(answer.evidence_turn_ids[:k])
        gold_evidence = set(qa.gold_evidence_turn_ids)
        has_gold_evidence = bool(gold_evidence & retrieved_evidence)
        has_all_gold_evidence = bool(gold_evidence) and gold_evidence <= retrieved_evidence
        if has_gold_evidence:
            evidence_correct += 1
        if has_all_gold_evidence:
            all_evidence_correct += 1

        normalized_correct = is_answer_correct(qa, answer)
        if normalized_correct and has_all_gold_evidence:
            faithful_answer_correct += 1

        if _is_state_sensitive_question(qa):
            strict_state_total += 1
            if is_exact_answer_correct(qa, answer) and has_all_gold_evidence:
                strict_state_correct += 1

        if qa.gold_update_relations:
            update_total += 1
            if has_gold_evidence:
                update_correct += 1

        if qa.question_type == "temporal_reasoning":
            temporal_total += 1
            if has_gold_evidence:
                temporal_correct += 1

    total = len(qa_items)
    abstention_total = sum(1 for qa in qa_items if qa.requires_abstention)
    non_abstention_total = max(total - abstention_total, 1)
    return EvaluationSummary(
        total=total,
        answer_accuracy=answer_correct / total if total else 0.0,
        evidence_recall_at_k=evidence_correct / non_abstention_total,
        abstention_accuracy=abstention_correct / abstention_total if abstention_total else 0.0,
        update_accuracy=update_correct / update_total if update_total else 0.0,
        temporal_accuracy=temporal_correct / temporal_total if temporal_total else 0.0,
        average_token_cost=token_cost / total if total else 0.0,
        exact_answer_accuracy=exact_answer_correct / total if total else 0.0,
        all_evidence_recall_at_k=all_evidence_correct / non_abstention_total,
        faithful_answer_accuracy=faithful_answer_correct / total if total else 0.0,
        strict_state_accuracy=strict_state_correct / strict_state_total if strict_state_total else 0.0,
    )


def is_exact_answer_correct(qa: QAItem, answer: Answer) -> bool:
    if qa.requires_abstention:
        return not answer.evidence_turn_ids and "沒有足夠資訊" in answer.answer
    return bool(qa.gold_answer and _compact(qa.gold_answer) in _compact(answer.answer))


def is_answer_correct(qa: QAItem, answer: Answer) -> bool:
    if is_exact_answer_correct(qa, answer):
        return True
    if qa.requires_abstention:
        return False
    if not qa.gold_answer or not answer.answer or "沒有足夠資訊" in answer.answer:
        return False

    gold = _normalize_for_semantics(qa.gold_answer)
    predicted = _normalize_for_semantics(answer.answer)
    required_terms = _required_terms(gold)
    predicted_terms = _answer_terms(predicted)
    if required_terms and not required_terms <= predicted_terms:
        return False

    gold_terms = _answer_terms(gold)
    if not gold_terms:
        return False
    overlap = len(gold_terms & predicted_terms)
    coverage = overlap / len(gold_terms)
    return coverage >= 0.5


def _token_proxy(text: str) -> int:
    return max(1, round(len(text) / 1.5))


def _is_state_sensitive_question(qa: QAItem) -> bool:
    return bool(qa.gold_update_relations) or qa.question_type in {
        "knowledge_update",
        "temporal_reasoning",
        "conflict_resolution",
        "multi_session_reasoning",
    }


def _compact(text: str) -> str:
    return re.sub(r"\s|[，。！？、；：,.!?;:]", "", text)


def _normalize_for_semantics(text: str) -> str:
    text = text.replace("臺", "台")
    text = re.sub(r"[，。！？、；：,.!?;:]", " ", text)
    for phrase in ["使用者", "目前", "現在", "原本", "一開始", "比較", "覺得", "作為", "安排", "主要"]:
        text = text.replace(phrase, " ")
    return re.sub(r"\s+", " ", text).strip()


def _answer_terms(text: str) -> set[str]:
    terms = tokenize(text)
    terms.update(_required_terms(text))
    return {term for term in terms if term not in _GENERIC_TERMS and len(term) >= 1}


def _required_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for term in _KNOWN_ENTITY_TERMS:
        if term in text:
            terms.add(term)
    terms.update(re.findall(r"[A-Za-z][A-Za-z0-9_+\-.]*", text))
    terms.update(re.findall(r"[一二三四五六七八九十\d]+(?:間|個|次|月|月底|週|點)", text))
    terms.update(re.findall(r"[阿小][\u4e00-\u9fff]{1,2}", text))
    return terms


_KNOWN_ENTITY_TERMS = {
    "台北",
    "新北",
    "桃園",
    "新竹",
    "台中",
    "台南",
    "高雄",
    "嘉義",
    "遠端",
    "線上",
    "研究所",
    "研究所考試",
    "暑期實習",
    "實習",
    "履歷初稿",
    "英文自傳",
    "五月底",
    "六月前",
    "社團開會",
    "室友",
    "租屋",
}

_GENERIC_TERMS = {
    "使用者",
    "偏好",
    "改為",
    "改成",
    "希望",
    "準備",
    "完成",
    "比較",
    "適合",
    "安排",
    "地點",
    "計畫",
    "想選",
}
