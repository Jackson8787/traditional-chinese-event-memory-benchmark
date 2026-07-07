import unittest

from event_memory.eval import evaluate, is_answer_correct, is_exact_answer_correct
from event_memory.schema import Answer, QAItem


class EvalTest(unittest.TestCase):
    def test_normalized_answer_accepts_semantic_paraphrase(self) -> None:
        qa = _qa("使用者一開始偏好台北")
        answer = _answer("使用者一開始比較想選台北作為安排地點。")

        self.assertFalse(is_exact_answer_correct(qa, answer))
        self.assertTrue(is_answer_correct(qa, answer))

    def test_normalized_answer_accepts_update_paraphrase_with_entities(self) -> None:
        qa = _qa("使用者改為偏好新竹或遠端")
        answer = _answer("使用者覺得新竹或遠端比較適合，不是台北。")

        self.assertTrue(is_answer_correct(qa, answer))

    def test_normalized_answer_rejects_wrong_entity(self) -> None:
        qa = _qa("使用者改為偏好新竹或遠端")
        answer = _answer("使用者一開始比較想選台北作為安排地點。")

        self.assertFalse(is_answer_correct(qa, answer))

    def test_evaluate_reports_exact_and_normalized_accuracy_separately(self) -> None:
        qa = _qa("使用者準備研究所考試")
        answer = _answer("使用者最近在準備研究所考試，並打算先把時間安排起來。")

        summary = evaluate([qa], [answer])

        self.assertEqual(summary.answer_accuracy, 1.0)
        self.assertEqual(summary.exact_answer_accuracy, 0.0)

    def test_evaluate_reports_faithful_and_all_evidence_metrics(self) -> None:
        qa = QAItem(
            question_id="q1",
            user_id="u01",
            question="提醒前後怎麼變化？",
            question_type="knowledge_update",
            gold_answer="原本是下週五前再看一次，後來改成下週三晚上先確認。",
            gold_evidence_turn_ids=["old_turn", "new_turn"],
            valid_time="current",
            gold_update_relations=[{"old_event_id": "old", "new_event_id": "new"}],
        )
        partial_answer = Answer(
            question_id="q1",
            answer="原本是下週五前再看一次，後來改成下週三晚上先確認。",
            evidence_turn_ids=["old_turn"],
            confidence="high",
            retrieved_event_ids=["old_event"],
        )

        summary = evaluate([qa], [partial_answer])

        self.assertEqual(summary.answer_accuracy, 1.0)
        self.assertEqual(summary.evidence_recall_at_k, 1.0)
        self.assertEqual(summary.all_evidence_recall_at_k, 0.0)
        self.assertEqual(summary.faithful_answer_accuracy, 0.0)
        self.assertEqual(summary.strict_state_accuracy, 0.0)


def _qa(gold_answer: str) -> QAItem:
    return QAItem(
        question_id="q1",
        user_id="u01",
        question="測試問題？",
        question_type="single_session_fact",
        gold_answer=gold_answer,
        gold_evidence_turn_ids=["t1"],
        valid_time="current",
    )


def _answer(text: str) -> Answer:
    return Answer(
        question_id="q1",
        answer=text,
        evidence_turn_ids=["t1"],
        confidence="high",
        retrieved_event_ids=["e1"],
    )


if __name__ == "__main__":
    unittest.main()
