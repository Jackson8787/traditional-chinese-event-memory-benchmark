import json
import tempfile
import unittest
from pathlib import Path

from event_memory.manual_audit_builder import build_manual_audit_package


class ManualAuditBuilderTest(unittest.TestCase):
    def test_build_manual_audit_package_writes_relation_and_answer_forms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            memory_dir = root / "memory"
            experiment_dir = root / "experiment"
            output_dir = root / "audit"
            dataset_dir.mkdir()
            memory_dir.mkdir()
            experiment_dir.mkdir()

            _write_jsonl(
                dataset_dir / "dialogues.jsonl",
                [
                    {
                        "user_id": "u01",
                        "session_id": "s1",
                        "turn_id": "t1",
                        "speaker": "user",
                        "timestamp": "2026-03-01",
                        "text": "我原本週末要去總圖三樓整理資料。",
                    },
                    {
                        "user_id": "u01",
                        "session_id": "s2",
                        "turn_id": "t2",
                        "speaker": "user",
                        "timestamp": "2026-03-08",
                        "text": "我後來改去資工系館討論室整理資料。",
                    },
                ],
            )
            _write_jsonl(
                dataset_dir / "qa.jsonl",
                [
                    {
                        "question_id": "q001",
                        "user_id": "u01",
                        "question": "目前週末整理地點在哪裡？",
                        "question_type": "knowledge_update",
                        "gold_answer": "資工系館討論室",
                        "gold_evidence_turn_ids": ["t2"],
                        "valid_time": "current",
                        "gold_event_ids": ["e2"],
                    }
                ],
            )
            _write_jsonl(
                memory_dir / "pred_events.jsonl",
                [
                    {
                        "event_id": "e1",
                        "user_id": "u01",
                        "time": "2026-03-01",
                        "speaker": "user",
                        "subject": "u01",
                        "event_type": "plan",
                        "content": "週末整理地點是總圖三樓",
                        "entities": ["週末整理", "總圖三樓"],
                        "evidence_turn_ids": ["t1"],
                        "importance": 0.7,
                    },
                    {
                        "event_id": "e2",
                        "user_id": "u01",
                        "time": "2026-03-08",
                        "speaker": "user",
                        "subject": "u01",
                        "event_type": "plan",
                        "content": "週末整理地點改成資工系館討論室",
                        "entities": ["週末整理", "資工系館討論室"],
                        "evidence_turn_ids": ["t2"],
                        "importance": 0.7,
                    },
                ],
            )
            _write_jsonl(
                memory_dir / "pred_update_relations.jsonl",
                [
                    {
                        "new_event_id": "e2",
                        "old_event_id": "e1",
                        "relation": "supersedes",
                        "reason": "後來改去新地點。",
                        "evidence_turn_ids": ["t2"],
                    }
                ],
            )
            _write_jsonl(
                experiment_dir / "answers.jsonl",
                [
                    {
                        "question_id": "q001",
                        "answer": "資工系館討論室",
                        "evidence_turn_ids": ["t2"],
                        "confidence": "high",
                        "retrieved_event_ids": ["e2"],
                    }
                ],
            )
            _write_jsonl(experiment_dir / "error_analysis.jsonl", [])

            report = build_manual_audit_package(
                dataset_dir=dataset_dir,
                memory_dir=memory_dir,
                experiment_dir=experiment_dir,
                output_dir=output_dir,
                relation_limit=1,
                answer_limit=1,
            )

            self.assertEqual(report["relation_audit_count"], 1)
            self.assertEqual(report["answer_audit_count"], 1)
            relation_rows = _read_jsonl(output_dir / "relation_precision_audit.jsonl")
            answer_rows = _read_jsonl(output_dir / "answer_faithfulness_audit.jsonl")
            self.assertEqual(relation_rows[0]["manual_relation_label"], "")
            self.assertEqual(relation_rows[0]["old_event_content"], "週末整理地點是總圖三樓")
            self.assertEqual(relation_rows[0]["new_event_content"], "週末整理地點改成資工系館討論室")
            self.assertIn("slot_match", relation_rows[0])
            self.assertEqual(answer_rows[0]["manual_answer_label"], "")
            self.assertEqual(answer_rows[0]["automatic_gold_evidence_covered"], True)
            self.assertIn("manual_evidence_label", answer_rows[0])
            self.assertTrue((output_dir / "manual_audit_report.json").exists())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
