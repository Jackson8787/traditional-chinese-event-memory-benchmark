import json
import tempfile
import unittest
from pathlib import Path

from event_memory.manual_audit_labeler import auto_label_manual_audit


class ManualAuditLabelerTest(unittest.TestCase):
    def test_auto_labels_obvious_relation_and_answer_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir)
            _write_jsonl(
                audit_dir / "relation_precision_audit.jsonl",
                [
                    {
                        "relation": "supersedes",
                        "old_slot": "tool",
                        "new_slot": "tool",
                        "old_value": "Google 試算表",
                        "new_value": "Notion 表格",
                        "slot_match": True,
                        "old_event_content": "主要整理工具是 Google 試算表",
                        "new_event_content": "主要整理工具改成 Notion 表格",
                        "new_evidence_text": "我後來把主要整理工具改成 Notion 表格。",
                        "manual_relation_label": "",
                        "manual_relation_note": "",
                    },
                    {
                        "relation": "supersedes",
                        "old_slot": "place",
                        "new_slot": "place",
                        "old_value": "資工系館討論室",
                        "new_value": "資工系館討論室",
                        "slot_match": True,
                        "old_event_content": "把資工系館討論室加入表格",
                        "new_event_content": "通勤安排改成資工系館討論室",
                        "new_evidence_text": "我後來把通勤安排改成資工系館討論室。",
                        "manual_relation_label": "",
                        "manual_relation_note": "",
                    },
                ],
            )
            _write_jsonl(
                audit_dir / "answer_faithfulness_audit.jsonl",
                [
                    {
                        "question_type": "knowledge_update",
                        "automatic_normalized_correct": True,
                        "automatic_gold_evidence_covered": True,
                        "missing_gold_evidence_turn_ids": [],
                        "error_source": "",
                        "manual_answer_label": "",
                        "manual_evidence_label": "",
                        "manual_answer_note": "",
                    },
                    {
                        "question_type": "temporal_reasoning",
                        "automatic_normalized_correct": True,
                        "automatic_gold_evidence_covered": False,
                        "missing_gold_evidence_turn_ids": ["t2"],
                        "error_source": "evidence_incomplete",
                        "manual_answer_label": "",
                        "manual_evidence_label": "",
                        "manual_answer_note": "",
                    },
                ],
            )

            report = auto_label_manual_audit(audit_dir)
            second_report = auto_label_manual_audit(audit_dir)
            relation_rows = _read_jsonl(audit_dir / "relation_precision_audit.jsonl")
            answer_rows = _read_jsonl(audit_dir / "answer_faithfulness_audit.jsonl")

        self.assertEqual(report["relation_labeled_count"], 2)
        self.assertEqual(report["new_relation_labeled_count"], 2)
        self.assertEqual(second_report["relation_labeled_count"], 2)
        self.assertEqual(second_report["new_relation_labeled_count"], 0)
        self.assertEqual(relation_rows[0]["manual_relation_label"], "correct")
        self.assertEqual(relation_rows[1]["manual_relation_label"], "incorrect")
        self.assertEqual(report["answer_labeled_count"], 2)
        self.assertEqual(report["new_answer_labeled_count"], 2)
        self.assertEqual(second_report["answer_labeled_count"], 2)
        self.assertEqual(second_report["new_answer_labeled_count"], 0)
        self.assertEqual(answer_rows[0]["manual_answer_label"], "correct")
        self.assertEqual(answer_rows[0]["manual_evidence_label"], "complete")
        self.assertEqual(answer_rows[1]["manual_answer_label"], "partially_correct")
        self.assertEqual(answer_rows[1]["manual_evidence_label"], "partial")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
