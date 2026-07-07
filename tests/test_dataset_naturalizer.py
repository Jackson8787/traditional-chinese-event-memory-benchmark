import tempfile
import unittest
from pathlib import Path

from event_memory.dataset_naturalizer import naturalize_dataset
from event_memory.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "v0"


class DatasetNaturalizerTest(unittest.TestCase):
    def test_naturalizes_dialogues_while_preserving_gold_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            counts = naturalize_dataset(DATASET_DIR, output)

            source_turns = read_jsonl(DATASET_DIR / "dialogues.jsonl")
            rewritten_turns = read_jsonl(output / "dialogues.jsonl")

            self.assertEqual(counts["dialogue_turns"], 100)
            self.assertEqual([row["turn_id"] for row in source_turns], [row["turn_id"] for row in rewritten_turns])
            self.assertGreater(
                sum(1 for old, new in zip(source_turns, rewritten_turns) if old["text"] != new["text"]),
                80,
            )
            self.assertTrue(any("地點要更新一下" in row["text"] for row in rewritten_turns))
            self.assertIn("Naturalized", (output / "README.md").read_text(encoding="utf-8"))

            for filename in [
                "personas.jsonl",
                "gold_events.jsonl",
                "gold_update_relations.jsonl",
                "qa.jsonl",
            ]:
                self.assertEqual(
                    (DATASET_DIR / filename).read_text(encoding="utf-8"),
                    (output / filename).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
