import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from event_memory.io import read_jsonl
from event_memory.public_grounded_dataset_builder import (
    build_public_grounded_hard_dataset,
    build_public_grounded_dataset,
    validate_public_grounded_hard_dataset,
    validate_public_grounded_dataset,
)


class PublicGroundedDatasetBuilderTest(unittest.TestCase):
    def test_builds_public_grounded_benchmark_dataset_with_expected_shape_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)

            counts = build_public_grounded_dataset(output)
            errors = validate_public_grounded_dataset(output)

            self.assertEqual(errors, [])
            self.assertEqual(counts["personas"], 12)
            self.assertEqual(counts["memory_turns"], 2880)
            self.assertEqual(counts["dialogue_turns"], 5760)
            self.assertEqual(counts["gold_events"], 2880)
            self.assertEqual(counts["gold_update_relations"], 120)
            self.assertEqual(counts["qa"], 360)
            self.assertEqual(counts["dataset_audit"], 576)
            for filename in [
                "source_manifest.jsonl",
                "source_facts.jsonl",
                "scenario_cards.jsonl",
                "memory_turns.jsonl",
                "source_audit.jsonl",
                "naturalness_audit.jsonl",
                "dataset_audit.jsonl",
                "README.md",
            ]:
                self.assertTrue((output / filename).exists(), filename)

            turns = read_jsonl(output / "dialogues.jsonl")
            memory_turns = read_jsonl(output / "memory_turns.jsonl")
            qa_rows = read_jsonl(output / "qa.jsonl")
            relations = read_jsonl(output / "gold_update_relations.jsonl")
            source_facts = read_jsonl(output / "source_facts.jsonl")
            scenario_cards = read_jsonl(output / "scenario_cards.jsonl")

            self.assertEqual(Counter(row["speaker"] for row in turns), Counter({"user": 2880, "assistant": 2880}))
            self.assertEqual(set(Counter(row["user_id"] for row in memory_turns).values()), {240})
            self.assertEqual(set(Counter(row["user_id"] for row in qa_rows).values()), {30})

            sessions_by_user: dict[str, set[str]] = defaultdict(set)
            for row in memory_turns:
                sessions_by_user[row["user_id"]].add(row["session_id"])
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
            self.assertEqual({len(row["source_fact_ids"]) for row in scenario_cards}, {8})
            self.assertGreaterEqual(min(Counter(row["user_id"] for row in source_facts).values()), 8)

    def test_source_facts_include_concrete_traceability_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)

            sources = read_jsonl(output / "source_manifest.jsonl")
            source_facts = read_jsonl(output / "source_facts.jsonl")

            for source in sources:
                self.assertTrue(source.get("source_page_title"), source)
                self.assertTrue(source.get("accessed_at"), source)
                self.assertTrue(source.get("license_evidence_note"), source)

            traceable_by_user: Counter[str] = Counter()
            for fact in source_facts:
                self.assertTrue(fact.get("fact_origin_type"), fact)
                self.assertTrue(fact.get("fact_span_hash"), fact)
                self.assertTrue(fact.get("extraction_note"), fact)
                if fact["fact_origin_type"] == "concrete_public_page_fact":
                    traceable_by_user[fact["user_id"]] += 1
                    self.assertTrue(fact.get("source_page_title"), fact)

            self.assertEqual(set(traceable_by_user), {f"u{index:02d}" for index in range(1, 13)})
            self.assertGreaterEqual(min(traceable_by_user.values()), 2)

    def test_sources_are_public_safe_and_assistant_turns_are_not_gold_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)

            sources = read_jsonl(output / "source_manifest.jsonl")
            turns = read_jsonl(output / "dialogues.jsonl")
            qa_rows = read_jsonl(output / "qa.jsonl")
            memory_rows = read_jsonl(output / "memory_turns.jsonl")

            self.assertTrue(all(row["license_status"] == "public_safe" for row in sources))
            self.assertTrue(all(row["derivative_allowed"] for row in sources))
            self.assertFalse(any("non-commercial" in row["license_note"].lower() for row in sources))
            self.assertNotIn("DuRecDial2", {row["source_label"] for row in sources})

            turns_by_id = {row["turn_id"]: row for row in turns}
            memory_turn_ids = {row["turn_id"] for row in memory_rows}
            for qa in qa_rows:
                for turn_id in qa["gold_evidence_turn_ids"]:
                    self.assertIn(turn_id, memory_turn_ids)
                    self.assertEqual(turns_by_id[turn_id]["speaker"], "user")

    def test_public_grounded_text_has_no_synthetic_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)

            checked_rows = []
            for filename in ["dialogues.jsonl", "gold_events.jsonl", "qa.jsonl"]:
                checked_rows.extend(read_jsonl(output / filename))
            forbidden = [
                "舊方案",
                "新方案",
                "第1次討論的第",
                "這和",
                "以臺灣日常說法",
                "改寫一段",
                "風格對話",
                "source",
                "dataset",
                "metadata",
                "prompt",
            ]

            for row in checked_rows:
                text = " ".join(str(value) for value in row.values())
                self.assertFalse(any(fragment in text for fragment in forbidden), text)

    def test_memory_user_text_does_not_repeat_within_a_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)

            turns = read_jsonl(output / "dialogues.jsonl")
            user_texts_by_session: dict[tuple[str, str], list[str]] = defaultdict(list)
            for row in turns:
                if row["speaker"] == "user":
                    user_texts_by_session[(row["user_id"], row["session_id"])].append(row["text"])

            for key, texts in user_texts_by_session.items():
                self.assertEqual(len(texts), len(set(texts)), f"{key}: {texts}")

    def test_public_grounded_text_avoids_high_frequency_fake_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)

            user_texts = [
                row["text"]
                for row in read_jsonl(output / "dialogues.jsonl")
                if row["speaker"] == "user"
            ]
            fake_fragments = [
                "其實我有點擔心",
                "所以想先把",
                "記在旁邊",
                "剛才想到一件事",
                "如果要讓",
                "也放進表格",
                "前面的安排先不用照舊",
                "現在以這個版本比較符合",
            ]

            for fragment in fake_fragments:
                self.assertLessEqual(sum(fragment in text for text in user_texts), 24, fragment)

    def test_public_grounded_user_text_has_enough_surface_variation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)

            user_texts = [
                row["text"]
                for row in read_jsonl(output / "dialogues.jsonl")
                if row["speaker"] == "user"
            ]
            text_counts = Counter(user_texts)

            self.assertGreaterEqual(len(text_counts), 1700)
            self.assertLessEqual(max(text_counts.values()), 3)

    def test_validation_rejects_non_public_source_and_synthetic_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)
            sources = read_jsonl(output / "source_manifest.jsonl")
            sources[0]["license_status"] = "non_commercial"
            _write_jsonl_for_test(output / "source_manifest.jsonl", sources)
            turns = read_jsonl(output / "dialogues.jsonl")
            turns[0]["text"] = "我原本的週末工作地點是校園專題舊方案1。"
            _write_jsonl_for_test(output / "dialogues.jsonl", turns)

            errors = validate_public_grounded_dataset(output)

            self.assertIn("source src_u01_01 is not public_safe", errors)
            self.assertIn("turn u01_s01_u01 leaks synthetic placeholder text", errors)

    def test_validation_rejects_missing_traceability_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            build_public_grounded_dataset(output)
            sources = read_jsonl(output / "source_manifest.jsonl")
            sources[0]["source_page_title"] = ""
            _write_jsonl_for_test(output / "source_manifest.jsonl", sources)
            facts = read_jsonl(output / "source_facts.jsonl")
            facts[0].pop("fact_span_hash", None)
            _write_jsonl_for_test(output / "source_facts.jsonl", facts)

            errors = validate_public_grounded_dataset(output)

            self.assertIn("source src_u01_01 missing source_page_title", errors)
            self.assertIn("fact fact_u01_01 missing fact_span_hash", errors)

    def test_builds_hard_challenge_split_without_changing_main_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "public_grounded"
            hard = Path(temp_dir) / "public_grounded_hard"
            build_public_grounded_dataset(base)

            counts = build_public_grounded_hard_dataset(base, hard)
            errors = validate_public_grounded_hard_dataset(hard)

            self.assertEqual(errors, [])
            self.assertEqual(counts["qa"], 72)
            self.assertEqual(len(read_jsonl(base / "qa.jsonl")), 360)
            hard_qa = read_jsonl(hard / "qa.jsonl")
            self.assertEqual(Counter(row["question_type"] for row in hard_qa), Counter({"hard_conflict": 48, "hard_temporal": 24}))
            self.assertTrue(all(row["gold_evidence_turn_ids"] for row in hard_qa))
            self.assertTrue(any(row["challenge_focus"] == "supersedes_expansion" for row in hard_qa))
            self.assertTrue(any(row["challenge_focus"] == "time_relevance" for row in hard_qa))


def _write_jsonl_for_test(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
