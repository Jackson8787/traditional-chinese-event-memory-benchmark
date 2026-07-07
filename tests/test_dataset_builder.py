import tempfile
import unittest
from collections import Counter
from pathlib import Path

from event_memory.dataset_builder import build_dataset, validate_dataset
from event_memory.io import read_jsonl


class DatasetBuilderTest(unittest.TestCase):
    def test_builds_valid_v0_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            counts = build_dataset(output)
            errors = validate_dataset(output)

            self.assertEqual(errors, [])
            self.assertEqual(counts["personas"], 10)
            self.assertEqual(counts["qa"], 100)
            self.assertEqual(counts["gold_update_relations"], 40)

            qa_rows = read_jsonl(output / "qa.jsonl")
            relation_rows = read_jsonl(output / "gold_update_relations.jsonl")
            event_rows = read_jsonl(output / "gold_events.jsonl")
            self.assertEqual(sum(1 for row in qa_rows if row["requires_abstention"]), 10)
            self.assertTrue(any(row["gold_update_relations"] for row in qa_rows))
            self.assertEqual(
                Counter(row["question_type"] for row in qa_rows),
                Counter(
                    {
                        "single_session_fact": 50,
                        "knowledge_update": 10,
                        "temporal_reasoning": 10,
                        "conflict_resolution": 10,
                        "multi_session_reasoning": 10,
                        "abstention": 10,
                    }
                ),
            )
            self.assertEqual(
                Counter(row["relation"] for row in relation_rows),
                Counter({"supersedes": 20, "corrects": 10, "supplements": 10}),
            )
            self.assertEqual(
                Counter(row["event_type"] for row in event_rows),
                Counter(
                    {
                        "plan": 20,
                        "preference": 30,
                        "constraint": 10,
                        "relationship": 10,
                        "completed_event": 10,
                        "negated_fact": 10,
                        "future_event": 10,
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main()
