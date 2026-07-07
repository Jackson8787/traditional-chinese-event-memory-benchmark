from __future__ import annotations

import json
from pathlib import Path

from .eval import is_answer_correct
from .io import write_jsonl
from .llm_client import ChatMessage, LLMClient
from .schema import Answer, QAItem


def build_judge_audit(
    qa_items: list[QAItem],
    answers: list[Answer],
    error_rows: list[dict],
    llm: LLMClient,
    *,
    limit: int = 30,
) -> list[dict]:
    answers_by_id = {answer.question_id: answer for answer in answers}
    errors_by_id = {row["question_id"]: row for row in error_rows}
    selected = _select_audit_items(qa_items, errors_by_id, limit)
    rows: list[dict] = []

    for qa in selected:
        answer = answers_by_id[qa.question_id]
        error_row = errors_by_id.get(qa.question_id)
        judge = _judge_answer(llm, qa, answer)
        rows.append(
            {
                "question_id": qa.question_id,
                "question_type": qa.question_type,
                "question": qa.question,
                "gold_answer": qa.gold_answer,
                "predicted_answer": answer.answer,
                "gold_evidence_turn_ids": qa.gold_evidence_turn_ids,
                "predicted_evidence_turn_ids": answer.evidence_turn_ids,
                "retrieved_event_ids": answer.retrieved_event_ids,
                "automatic_normalized_correct": is_answer_correct(qa, answer),
                "error_source": error_row["error_source"] if error_row is not None else "",
                "judge_label": judge["label"],
                "judge_note": judge["note"],
            }
        )

    return rows


def write_judge_audit(path: str | Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)


def _select_audit_items(qa_items: list[QAItem], errors_by_id: dict[str, dict], limit: int) -> list[QAItem]:
    selected: list[QAItem] = []
    seen: set[str] = set()

    for qa in qa_items:
        if qa.question_id in errors_by_id:
            selected.append(qa)
            seen.add(qa.question_id)
        if len(selected) >= limit:
            return selected

    for qa in qa_items:
        if qa.question_id in seen:
            continue
        selected.append(qa)
        if len(selected) >= limit:
            return selected

    return selected


def _judge_answer(llm: LLMClient, qa: QAItem, answer: Answer) -> dict[str, str]:
    prompt = f"""
你是長期對話記憶 QA 的評審。請只根據 gold answer、gold evidence ids、predicted answer、predicted evidence ids 判斷。

輸出必須是 JSON，不要加 markdown：
{{"label": "correct|partially_correct|incorrect", "note": "一句繁體中文理由"}}

Question: {qa.question}
Question type: {qa.question_type}
Gold answer: {qa.gold_answer}
Gold evidence turn ids: {qa.gold_evidence_turn_ids}
Predicted answer: {answer.answer}
Predicted evidence turn ids: {answer.evidence_turn_ids}
Retrieved event ids: {answer.retrieved_event_ids}
Requires abstention: {qa.requires_abstention}
""".strip()
    result = llm.chat(
        [
            ChatMessage(role="system", content="你是嚴格但簡潔的繁體中文 NLP 評測員。"),
            ChatMessage(role="user", content=prompt),
        ],
        max_completion_tokens=160,
    )
    parsed = _parse_judge_json(result.content)
    label = parsed.get("label", "incorrect")
    if label not in {"correct", "partially_correct", "incorrect"}:
        label = "incorrect"
    note = parsed.get("note", "").strip() or "judge 未提供理由。"
    return {"label": label, "note": note}


def _parse_judge_json(text: str) -> dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {"label": "incorrect", "note": "judge 輸出不是合法 JSON。"}
    if not isinstance(raw, dict):
        return {"label": "incorrect", "note": "judge 輸出不是 JSON object。"}
    return {str(key): str(value) for key, value in raw.items()}
