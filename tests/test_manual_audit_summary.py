import json
import tempfile
import unittest
from pathlib import Path

from event_memory.manual_audit_summary import summarize_manual_audit


class ManualAuditSummaryTest(unittest.TestCase):
    def test_summarizes_relation_and_answer_manual_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir)
            _write_jsonl(
                audit_dir / "relation_precision_audit.jsonl",
                [
                    {
                        "relation": "supersedes",
                        "manual_relation_label": "correct",
                    },
                    {
                        "relation": "supersedes",
                        "manual_relation_label": "incorrect",
                    },
                    {
                        "relation": "supplements",
                        "manual_relation_label": "",
                    },
                ],
            )
            _write_jsonl(
                audit_dir / "answer_faithfulness_audit.jsonl",
                [
                    {
                        "question_type": "knowledge_update",
                        "manual_answer_label": "correct",
                        "manual_evidence_label": "complete",
                    },
                    {
                        "question_type": "temporal_reasoning",
                        "manual_answer_label": "partially_correct",
                        "manual_evidence_label": "partial",
                    },
                    {
                        "question_type": "temporal_reasoning",
                        "manual_answer_label": "",
                        "manual_evidence_label": "",
                    },
                ],
            )

            summary = summarize_manual_audit(audit_dir)

        self.assertEqual(summary["relations"]["labeled_count"], 2)
        self.assertEqual(summary["relations"]["by_type"]["supersedes"]["correct"], 1)
        self.assertEqual(summary["relations"]["by_type"]["supersedes"]["incorrect"], 1)
        self.assertEqual(summary["relations"]["by_type"]["supersedes"]["precision"], 0.5)
        self.assertEqual(summary["answers"]["labeled_count"], 2)
        self.assertEqual(summary["answers"]["by_question_type"]["temporal_reasoning"]["partially_correct"], 1)
        self.assertEqual(summary["evidence"]["labels"]["complete"], 1)
        self.assertEqual(summary["evidence"]["labels"]["partial"], 1)
        self.assertEqual(summary["evidence"]["labels"]["complete_rate"], 0.5)
        self.assertEqual(summary["evidence"]["labels"]["complete_or_partial_rate"], 1.0)
        self.assertNotIn("precision", summary["evidence"]["labels"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
