import json
import tempfile
import unittest
from pathlib import Path

from event_memory.io import read_jsonl
from event_memory.memory_text_refresher import refresh_cached_memory_text


class MemoryTextRefresherTest(unittest.TestCase):
    def test_refreshes_cached_event_content_from_current_dataset_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            memory_dir = root / "memory"
            output_dir = root / "refreshed"
            dataset_dir.mkdir()
            memory_dir.mkdir()
            _write_jsonl(
                dataset_dir / "dialogues.jsonl",
                [
                    {
                        "turn_id": "t1",
                        "speaker": "user",
                        "text": "晚上補紀錄時，我先把新的資料來源列成待確認項目。",
                    }
                ],
            )
            _write_jsonl(
                memory_dir / "pred_events.jsonl",
                [
                    {
                        "event_id": "e1",
                        "content": "其實我有點擔心，所以想先把資料來源記在旁邊。",
                        "evidence_turn_ids": ["t1"],
                    }
                ],
            )
            _write_jsonl(
                memory_dir / "pred_update_relations.jsonl",
                [
                    {
                        "new_event_id": "e1",
                        "old_event_id": "e0",
                        "relation": "supersedes",
                        "reason": "前面的安排先不用照舊。",
                        "evidence_turn_ids": ["t1"],
                    }
                ],
            )

            report = refresh_cached_memory_text(dataset_dir, memory_dir, output_dir)

            events = read_jsonl(output_dir / "pred_events.jsonl")
            relations = read_jsonl(output_dir / "pred_update_relations.jsonl")

        self.assertEqual(report["event_rows_refreshed"], 1)
        self.assertIn("晚上補紀錄時", events[0]["content"])
        self.assertNotIn("其實我有點擔心", events[0]["content"])
        self.assertNotIn("text_refresh_note", events[0])
        self.assertIn("本地刷新", relations[0]["reason"])
        self.assertNotIn("前面的安排先不用照舊", relations[0]["reason"])
        self.assertNotIn("text_refresh_note", relations[0])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
