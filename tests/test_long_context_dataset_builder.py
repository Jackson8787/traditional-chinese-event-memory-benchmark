import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from event_memory.io import read_jsonl
from event_memory.long_context_dataset_builder import (
    build_long_context_dataset,
    validate_long_context_dataset,
)


class LongContextDatasetBuilderTest(unittest.TestCase):
    def test_builds_v2_long_context_dataset_with_expected_shape_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)

            counts = build_long_context_dataset(output)
            errors = validate_long_context_dataset(output)

            self.assertEqual(errors, [])
            self.assertEqual(counts["personas"], 12)
            self.assertEqual(counts["dialogue_turns"], 2880)
            self.assertEqual(counts["qa"], 360)
            self.assertEqual(counts["source_adaptations"], 1728)
            self.assertEqual(counts["dataset_audit"], 576)
            self.assertEqual(counts["gold_update_relations"], 120)

            turns = read_jsonl(output / "dialogues.jsonl")
            qa_rows = read_jsonl(output / "qa.jsonl")
            relations = read_jsonl(output / "gold_update_relations.jsonl")
            adaptations = read_jsonl(output / "source_adaptations.jsonl")
            audit_rows = read_jsonl(output / "dataset_audit.jsonl")

            turns_by_user = Counter(row["user_id"] for row in turns)
            qa_by_user = Counter(row["user_id"] for row in qa_rows)
            self.assertEqual(set(turns_by_user.values()), {240})
            self.assertEqual(set(qa_by_user.values()), {30})

            sessions_by_user: dict[str, set[str]] = defaultdict(set)
            for turn in turns:
                sessions_by_user[turn["user_id"]].add(turn["session_id"])
            self.assertEqual({len(sessions) for sessions in sessions_by_user.values()}, {30})

            self.assertEqual(
                Counter(row["question_type"] for row in qa_rows),
                Counter(
                    {
                        "single_session_fact": 24,
                        "knowledge_update": 84,
                        "temporal_reasoning": 72,
                        "conflict_resolution": 72,
                        "multi_session_reasoning": 72,
                        "abstention": 36,
                    }
                ),
            )
            self.assertGreaterEqual(min(Counter(row["new_event_id"].split("_")[0] for row in relations).values()), 10)

            source_counts = Counter(row["source_dataset"] for row in adaptations)
            self.assertEqual(
                source_counts,
                Counter({"CrossWOZ": 432, "KdConv": 432, "NaturalConv": 432, "DuRecDial2": 432}),
            )
            self.assertTrue(all(row["usage_depth"] == "sentence_rewrite" for row in adaptations))
            self.assertTrue(all(row["linked_turn_ids"] for row in adaptations))
            self.assertTrue(all(row["source_text_hash"] for row in adaptations))

            linked_public_turns = {turn_id for row in adaptations for turn_id in row["linked_turn_ids"]}
            self.assertEqual(len(linked_public_turns), 1728)
            self.assertEqual(len(turns) - len(linked_public_turns), 1152)

            self.assertEqual(len(audit_rows), 576)
            self.assertEqual(
                {row["source_dataset"] for row in audit_rows},
                {"CrossWOZ", "KdConv", "NaturalConv", "DuRecDial2"},
            )
            self.assertEqual({row["user_id"] for row in audit_rows}, {f"u{i:02d}" for i in range(1, 13)})
            self.assertTrue((output / "README.md").exists())
            self.assertIn("non-commercial", (output / "README.md").read_text(encoding="utf-8").lower())

    def test_public_adapted_text_does_not_leak_generation_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_long_context_dataset(output)

            turns = read_jsonl(output / "dialogues.jsonl")
            adaptations = read_jsonl(output / "source_adaptations.jsonl")
            turns_by_id = {row["turn_id"]: row for row in turns}
            forbidden_fragments = [
                "以臺灣日常說法",
                "改寫一段",
                "風格對話",
                "知識導向對話",
                "實體延續",
                "自然話題轉移",
                "使用者 profile",
            ]

            checked_texts = [row["adapted_text"] for row in adaptations]
            checked_texts.extend(turns_by_id[turn_id]["text"] for row in adaptations for turn_id in row["linked_turn_ids"])

            for text in checked_texts:
                self.assertFalse(any(fragment in text for fragment in forbidden_fragments), text)

    def test_validation_rejects_missing_adaptation_turn_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_long_context_dataset(output)
            rows = read_jsonl(output / "source_adaptations.jsonl")
            rows[0]["linked_turn_ids"] = ["missing_turn"]
            _write_jsonl_for_test(output / "source_adaptations.jsonl", rows)

            errors = validate_long_context_dataset(output)

            self.assertIn("adaptation adapt_u01_001 references missing turn missing_turn", errors)

    def test_validation_rejects_adaptation_instruction_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_long_context_dataset(output)
            rows = read_jsonl(output / "source_adaptations.jsonl")
            rows[0]["adapted_text"] = "宜庭以臺灣日常說法改寫一段KdConv風格對話。"
            _write_jsonl_for_test(output / "source_adaptations.jsonl", rows)

            errors = validate_long_context_dataset(output)

            self.assertIn("adaptation adapt_u01_001 leaks generation instruction text", errors)


def _write_jsonl_for_test(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
