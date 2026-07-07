import json
import tempfile
import unittest
from pathlib import Path

from event_memory.io import write_jsonl
from event_memory.structured_events import structure_predicted_memory


class StructuredEventsTest(unittest.TestCase):
    def test_structure_predicted_memory_writes_slot_value_artifact_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir) / "memory"
            output_dir = Path(temp_dir) / "structured"
            memory_dir.mkdir()
            write_jsonl(
                memory_dir / "pred_events.jsonl",
                [
                    _event(
                        "e1",
                        "週末整理資料地點是總圖三樓",
                        ["週末整理", "總圖三樓"],
                        "2026-03-01",
                    ),
                    _event(
                        "e2",
                        "後來把週末整理資料地點改成資工系館討論室",
                        ["週末整理", "資工系館討論室"],
                        "2026-03-08",
                    ),
                ],
            )

            report = structure_predicted_memory(memory_dir, output_dir)

            rows = _read_jsonl(output_dir / "pred_events_structured.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["memory_slot"], "place")
            self.assertEqual(rows[0]["memory_value"], "總圖三樓")
            self.assertEqual(rows[0]["valid_from"], "2026-03-01")
            self.assertEqual(rows[0]["is_state_update"], False)
            self.assertEqual(rows[1]["memory_slot"], "place")
            self.assertEqual(rows[1]["memory_value"], "資工系館討論室")
            self.assertEqual(rows[1]["is_state_update"], True)
            self.assertEqual(report["event_count"], 2)
            self.assertEqual(report["structured_event_count"], 2)
            self.assertEqual(report["slot_counts"]["place"], 2)
            self.assertTrue((output_dir / "event_structure_report.json").exists())


def _event(event_id: str, content: str, entities: list[str], time: str) -> dict:
    return {
        "event_id": event_id,
        "user_id": "u01",
        "time": time,
        "speaker": "user",
        "subject": "使用者",
        "event_type": "plan",
        "content": content,
        "entities": entities,
        "evidence_turn_ids": [f"{event_id}_turn"],
        "source_context_ids": [],
        "importance": 0.8,
        "superseded_by": None,
        "corrected_by": None,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
