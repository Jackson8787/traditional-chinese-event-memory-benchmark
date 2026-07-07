import json
import tempfile
import unittest
from pathlib import Path

from event_memory.config import EndpointConfig
from event_memory.judge_audit import build_judge_audit, write_judge_audit
from event_memory.llm_client import LLMClient
from event_memory.schema import Answer, QAItem


class JudgeAuditTest(unittest.TestCase):
    def test_build_judge_audit_writes_required_fields(self) -> None:
        qa_items = [
            QAItem(
                question_id="q001",
                user_id="u01",
                question="現在的主要計畫是什麼？",
                question_type="knowledge_update",
                gold_answer="使用者改成尋找暑期實習",
                gold_evidence_turn_ids=["t1"],
                valid_time="current",
                gold_event_ids=["e1"],
            )
        ]
        answers = [
            Answer(
                question_id="q001",
                answer="使用者改成尋找暑期實習。",
                evidence_turn_ids=["t1"],
                confidence="high",
                retrieved_event_ids=["e1"],
            )
        ]
        error_rows: list[dict] = []

        rows = build_judge_audit(qa_items, answers, error_rows, _fake_judge_llm(), limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            set(rows[0]),
            {
                "question_id",
                "question_type",
                "question",
                "gold_answer",
                "predicted_answer",
                "gold_evidence_turn_ids",
                "predicted_evidence_turn_ids",
                "retrieved_event_ids",
                "automatic_normalized_correct",
                "error_source",
                "judge_label",
                "judge_note",
            },
        )
        self.assertEqual(rows[0]["judge_label"], "correct")

    def test_write_judge_audit_creates_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "audit.jsonl"
            write_judge_audit(output_path, [{"question_id": "q001", "judge_label": "correct"}])

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["question_id"], "q001")


def _fake_judge_llm() -> LLMClient:
    def fake_transport(url, api_key, payload, timeout_seconds):
        return {
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"label": "correct", "note": "答案與 evidence 都符合 gold。"},
                            ensure_ascii=False,
                        )
                    }
                }
            ],
        }

    return LLMClient(
        EndpointConfig(provider="openai", base_url="https://example.test", api_key="secret", model="judge-test"),
        transport=fake_transport,
    )


if __name__ == "__main__":
    unittest.main()
