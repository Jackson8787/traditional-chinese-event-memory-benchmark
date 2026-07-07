import tempfile
import unittest
from collections import Counter
from pathlib import Path

from event_memory.grounded_dataset_builder import build_grounded_dataset, validate_grounded_dataset
from event_memory.io import read_jsonl


class GroundedDatasetBuilderTest(unittest.TestCase):
    def test_builds_valid_grounded_dataset_with_source_contexts_and_phase2_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            counts = build_grounded_dataset(output)
            errors = validate_grounded_dataset(output)

            self.assertEqual(errors, [])
            self.assertEqual(counts["personas"], 10)
            self.assertEqual(counts["qa"], 150)
            self.assertGreaterEqual(counts["source_contexts"], 20)
            self.assertGreaterEqual(counts["gold_update_relations"], 50)

            source_rows = read_jsonl(output / "source_contexts.jsonl")
            persona_rows = read_jsonl(output / "personas.jsonl")
            qa_rows = read_jsonl(output / "qa.jsonl")
            event_rows = read_jsonl(output / "gold_events.jsonl")
            relation_rows = read_jsonl(output / "gold_update_relations.jsonl")

            source_types = Counter(row["source_type"] for row in source_rows)
            self.assertGreaterEqual(source_types["public_context"], 10)
            self.assertGreaterEqual(source_types["simplified_chinese_adaptation"], 5)
            self.assertTrue(all(row["source_context_ids"] for row in persona_rows))
            self.assertTrue(all(row["source_context_ids"] for row in event_rows))
            self.assertTrue(any("NaturalConv" in row["source_label"] for row in source_rows))
            self.assertTrue(any("LCCC" in row["source_label"] for row in source_rows))

            qa_types = Counter(row["question_type"] for row in qa_rows)
            self.assertGreaterEqual(qa_types["knowledge_update"], 30)
            self.assertGreaterEqual(qa_types["temporal_reasoning"], 30)
            self.assertGreaterEqual(qa_types["conflict_resolution"], 30)
            self.assertGreaterEqual(qa_types["multi_session_reasoning"], 30)
            self.assertGreaterEqual(qa_types["abstention"], 20)

            relation_types = Counter(row["relation"] for row in relation_rows)
            self.assertGreaterEqual(relation_types["supersedes"], 20)
            self.assertGreaterEqual(relation_types["corrects"], 20)
            self.assertGreaterEqual(relation_types["supplements"], 10)

    def test_grounded_validation_rejects_missing_source_context_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_grounded_dataset(output)
            persona_rows = read_jsonl(output / "personas.jsonl")
            persona_rows[0]["source_context_ids"] = ["missing_context"]
            _write_jsonl_for_test(output / "personas.jsonl", persona_rows)

            errors = validate_grounded_dataset(output)

            self.assertIn("persona u01 references missing source_context missing_context", errors)


def _write_jsonl_for_test(path: Path, rows: list[dict]) -> None:
    import json

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
