import unittest

from event_memory.answering import MultiEvidenceGroundedAnswerer, StateAwareAnswerer
from event_memory.eval import evaluate
from event_memory.schema import DialogueTurn, Event, QAItem, RetrievalResult


class StateAwareAnswererTest(unittest.TestCase):
    def test_outputs_exact_old_new_state_when_chain_evidence_is_available(self) -> None:
        qa = QAItem(
            question_id="q1",
            user_id="u01",
            question="宜庭原本和後來的週末整理地點分別是什麼？",
            question_type="temporal_reasoning",
            gold_answer="使用者原本的週末整理地點是總圖三樓；使用者目前的週末整理地點改成資工系館討論室",
            gold_evidence_turn_ids=["t_old", "t_new"],
            valid_time="current",
            gold_update_relations=[{"old_event_id": "old", "new_event_id": "new"}],
        )
        retrieved = [
            _result(
                "new",
                "使用者目前的週末整理地點改成資工系館討論室",
                ["週末整理地點", "資工系館討論室"],
                "2026-04-01",
                "t_new",
                0.9,
            ),
            _result(
                "old",
                "使用者原本的週末整理地點是總圖三樓",
                ["週末整理地點", "總圖三樓"],
                "2026-03-01",
                "t_old",
                0.87,
            ),
        ]

        answer = StateAwareAnswerer(MultiEvidenceGroundedAnswerer()).answer(qa, retrieved)
        summary = evaluate([qa], [answer])

        self.assertEqual(answer.answer, qa.gold_answer)
        self.assertEqual(answer.evidence_turn_ids, ["t_old", "t_new"])
        self.assertEqual(summary.strict_state_accuracy, 1.0)

    def test_change_question_falls_back_when_only_one_state_side_is_available(self) -> None:
        qa = QAItem(
            question_id="q1",
            user_id="u01",
            question="宜庭原本和後來的週末整理地點分別是什麼？",
            question_type="temporal_reasoning",
            gold_answer="使用者原本的週末整理地點是總圖三樓；使用者目前的週末整理地點改成資工系館討論室",
            gold_evidence_turn_ids=["t_old", "t_new"],
            valid_time="current",
            gold_update_relations=[{"old_event_id": "old", "new_event_id": "new"}],
        )
        retrieved = [
            _result(
                "old",
                "使用者原本的週末整理地點是總圖三樓",
                ["週末整理地點", "總圖三樓"],
                "2026-03-01",
                "t_old",
                0.87,
            )
        ]

        answer = StateAwareAnswerer(MultiEvidenceGroundedAnswerer()).answer(qa, retrieved)

        self.assertNotEqual(answer.answer, "使用者目前的週末整理地點改成總圖三樓")


def _result(event_id: str, content: str, entities: list[str], time: str, turn_id: str, score: float) -> RetrievalResult:
    event = Event(
        event_id=event_id,
        user_id="u01",
        time=time,
        speaker="user",
        subject="使用者",
        event_type="plan",
        content=content,
        entities=entities,
        evidence_turn_ids=[turn_id],
        importance=0.8,
    )
    turn = DialogueTurn(
        user_id="u01",
        session_id=turn_id.rsplit("_", 1)[0],
        turn_id=turn_id,
        speaker="user",
        timestamp=time,
        text=content,
    )
    return RetrievalResult(event=event, score=score, evidence_turns=[turn])


if __name__ == "__main__":
    unittest.main()
