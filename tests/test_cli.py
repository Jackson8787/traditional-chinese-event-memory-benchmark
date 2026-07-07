import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from event_memory.config import EndpointConfig
from event_memory.eval import EvaluationSummary
from event_memory.experiment import ExperimentResult
from event_memory import cli


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "v0"


class CliTest(unittest.TestCase):
    def test_run_experiment_command_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "event-memory",
                "run-experiment",
                "--method",
                "chunk_rag",
                "--dataset-dir",
                str(DATASET_DIR),
                "--output-dir",
                temp_dir,
                "--limit",
                "3",
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

            metrics_path = Path(temp_dir) / "chunk_rag" / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["method"], "chunk_rag")
            self.assertEqual(metrics["summary"]["total"], 3)

    def test_run_experiment_command_accepts_summary_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "event-memory",
                "run-experiment",
                "--method",
                "summary_memory",
                "--dataset-dir",
                str(DATASET_DIR),
                "--output-dir",
                temp_dir,
                "--limit",
                "3",
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

            metrics_path = Path(temp_dir) / "summary_memory" / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["method"], "summary_memory")
            self.assertEqual(metrics["summary"]["total"], 3)

    def test_build_manual_audit_command_writes_audit_files(self) -> None:
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
            argv = [
                "event-memory",
                "build-manual-audit",
                "--dataset-dir",
                str(dataset_dir),
                "--memory-dir",
                str(memory_dir),
                "--experiment-dir",
                str(experiment_dir),
                "--output-dir",
                str(output_dir),
                "--relation-limit",
                "1",
                "--answer-limit",
                "1",
            ]

            with patch.object(sys, "argv", argv):
                cli.main()

            self.assertTrue((output_dir / "relation_precision_audit.jsonl").exists())
            self.assertTrue((output_dir / "answer_faithfulness_audit.jsonl").exists())
            self.assertTrue((output_dir / "manual_audit_report.json").exists())

    def test_structure_llm_events_command_writes_structured_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_dir = root / "memory"
            output_dir = root / "structured"
            memory_dir.mkdir()
            _write_jsonl(
                memory_dir / "pred_events.jsonl",
                [
                    {
                        "event_id": "e1",
                        "user_id": "u01",
                        "time": "2026-03-01",
                        "speaker": "user",
                        "subject": "使用者",
                        "event_type": "plan",
                        "content": "後來把週末整理資料地點改成資工系館討論室",
                        "entities": ["週末整理", "資工系館討論室"],
                        "evidence_turn_ids": ["t1"],
                    }
                ],
            )
            argv = [
                "event-memory",
                "structure-llm-events",
                "--memory-dir",
                str(memory_dir),
                "--output-dir",
                str(output_dir),
            ]

            with patch.object(sys, "argv", argv):
                cli.main()

            rows = [
                json.loads(line)
                for line in (output_dir / "pred_events_structured.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["memory_slot"], "place")
            self.assertTrue((output_dir / "event_structure_report.json").exists())

    def test_refresh_cached_memory_text_command_writes_refreshed_memory(self) -> None:
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
                        "text": "午休前整理筆記時，我先把資料來源列成待確認項目。",
                    }
                ],
            )
            _write_jsonl(
                memory_dir / "pred_events.jsonl",
                [
                    {
                        "event_id": "e1",
                        "content": "所以想先把資料來源記在旁邊。",
                        "evidence_turn_ids": ["t1"],
                    }
                ],
            )
            argv = [
                "event-memory",
                "refresh-cached-memory-text",
                "--dataset-dir",
                str(dataset_dir),
                "--memory-dir",
                str(memory_dir),
                "--output-dir",
                str(output_dir),
            ]

            with patch.object(sys, "argv", argv):
                cli.main()

            rows = [
                json.loads(line)
                for line in (output_dir / "pred_events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("午休前整理筆記時", rows[0]["content"])
            self.assertTrue((output_dir / "memory_text_refresh_report.json").exists())

    def test_run_experiment_command_passes_llm_client_for_event_memory_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "event-memory",
                "run-experiment",
                "--method",
                "event_memory_llm",
                "--dataset-dir",
                str(DATASET_DIR),
                "--output-dir",
                temp_dir,
                "--limit",
                "3",
            ]
            fake_result = ExperimentResult(
                method="event_memory_llm",
                summary=EvaluationSummary(total=3, answer_accuracy=0.0, evidence_recall_at_k=0.0, abstention_accuracy=0.0),
                question_type_metrics={},
                answers=[],
                latency_ms=1,
            )
            fake_config = type(
                "FakeConfig",
                (),
                {
                    "llm": EndpointConfig(
                        provider="openai",
                        base_url="https://example.test",
                        api_key="secret",
                        model="gpt-test",
                    )
                },
            )()
            with (
                patch.object(sys, "argv", argv),
                patch.object(cli, "load_config", return_value=fake_config),
                patch.object(cli, "run_experiment_pipeline", return_value=fake_result) as runner,
            ):
                cli.main()

            self.assertIsNotNone(runner.call_args.kwargs["llm"])
            self.assertEqual(runner.call_args.kwargs["method"], "event_memory_llm")

    def test_run_experiment_command_uses_memory_dir_without_llm_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir) / "memory"
            memory_dir.mkdir()
            argv = [
                "event-memory",
                "run-experiment",
                "--method",
                "event_memory_llm",
                "--dataset-dir",
                str(DATASET_DIR),
                "--output-dir",
                temp_dir,
                "--memory-dir",
                str(memory_dir),
            ]
            fake_result = ExperimentResult(
                method="event_memory_llm",
                summary=EvaluationSummary(total=1, answer_accuracy=0.0, evidence_recall_at_k=0.0, abstention_accuracy=0.0),
                question_type_metrics={},
                answers=[],
                latency_ms=1,
            )
            with (
                patch.object(sys, "argv", argv),
                patch.object(cli, "load_config") as load_config_mock,
                patch.object(cli, "run_experiment_pipeline", return_value=fake_result) as runner,
            ):
                cli.main()

            load_config_mock.assert_not_called()
            self.assertIsNone(runner.call_args.kwargs["llm"])
            self.assertEqual(runner.call_args.kwargs["memory_dir"], memory_dir)

    def test_build_llm_memory_command_writes_realistic_llm_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "memory"
            fake_config = type(
                "FakeConfig",
                (),
                {
                    "llm": EndpointConfig(
                        provider="openai",
                        base_url="https://example.test",
                        api_key="secret",
                        model="gpt-test",
                    )
                },
            )()

            def fake_build(dataset_dir, output_dir_arg, llm, *, limit_qa, force):
                output_dir_arg.mkdir(parents=True)
                (output_dir_arg / "pred_events.jsonl").write_text("{}\n", encoding="utf-8")
                (output_dir_arg / "pred_update_relations.jsonl").write_text("", encoding="utf-8")
                (output_dir_arg / "memory_build_stats.json").write_text(
                    json.dumps(
                        {
                            "selected_qa_count": limit_qa,
                            "selected_user_count": 1,
                            "selected_session_count": 2,
                            "llm_calls_attempted": 2,
                            "llm_calls_failed": 0,
                            "session_cache_hits": 0,
                            "session_cache_misses": 2,
                            "relation_judge_attempts": 0,
                            "relation_judge_failures": 0,
                            "parsed_event_count": 1,
                            "predicted_relation_count": 0,
                            "elapsed_ms": 1,
                        }
                    ),
                    encoding="utf-8",
                )
                (output_dir_arg / "memory_build_manifest.json").write_text("{}\n", encoding="utf-8")
                (output_dir_arg / "memory_build_errors.jsonl").write_text("", encoding="utf-8")
                return {
                    "selected_qa_count": limit_qa,
                    "selected_user_count": 1,
                    "selected_session_count": 2,
                    "llm_calls_attempted": 2,
                    "llm_calls_failed": 0,
                    "session_cache_hits": 0,
                    "session_cache_misses": 2,
                    "relation_judge_attempts": 0,
                    "relation_judge_failures": 0,
                    "parsed_event_count": 1,
                    "predicted_relation_count": 0,
                    "elapsed_ms": 1,
                }

            argv = [
                "event-memory",
                "build-llm-memory",
                "--dataset-dir",
                str(DATASET_DIR),
                "--output-dir",
                str(output_dir),
                "--limit-qa",
                "60",
                "--config",
                "api_config.env",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(cli, "load_config", return_value=fake_config),
                patch.object(cli, "build_llm_memory", side_effect=fake_build) as builder,
            ):
                cli.main()

            self.assertEqual(builder.call_args.kwargs["limit_qa"], 60)
            self.assertFalse(builder.call_args.kwargs["force"])
            self.assertTrue((output_dir / "pred_events.jsonl").exists())
            self.assertTrue((output_dir / "pred_update_relations.jsonl").exists())
            self.assertTrue((output_dir / "memory_build_stats.json").exists())

    def test_filter_llm_relations_command_writes_filtered_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir) / "memory"
            output_dir = Path(temp_dir) / "filtered"
            memory_dir.mkdir()
            (memory_dir / "pred_events.jsonl").write_text("{}\n", encoding="utf-8")
            (memory_dir / "pred_update_relations.jsonl").write_text("", encoding="utf-8")

            def fake_filter(memory_dir_arg, output_dir_arg):
                output_dir_arg.mkdir(parents=True)
                (output_dir_arg / "pred_events.jsonl").write_text("{}\n", encoding="utf-8")
                (output_dir_arg / "pred_update_relations.jsonl").write_text("", encoding="utf-8")
                (output_dir_arg / "relation_filter_report.json").write_text(
                    json.dumps(
                        {
                            "input_relation_count": 0,
                            "kept_relation_count": 0,
                            "removed_relation_count": 0,
                            "removed_by_relation": {},
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "input_relation_count": 0,
                    "kept_relation_count": 0,
                    "removed_relation_count": 0,
                    "removed_by_relation": {},
                }

            argv = [
                "event-memory",
                "filter-llm-relations",
                "--memory-dir",
                str(memory_dir),
                "--output-dir",
                str(output_dir),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(cli, "filter_predicted_memory", side_effect=fake_filter) as filterer,
            ):
                cli.main()

            self.assertEqual(filterer.call_args.args[0], memory_dir)
            self.assertEqual(filterer.call_args.args[1], output_dir)
            self.assertTrue((output_dir / "pred_events.jsonl").exists())
            self.assertTrue((output_dir / "pred_update_relations.jsonl").exists())
            self.assertTrue((output_dir / "relation_filter_report.json").exists())

    def test_naturalize_dataset_command_writes_output_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "naturalized"
            argv = [
                "event-memory",
                "naturalize-dataset",
                "--dataset-dir",
                str(DATASET_DIR),
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

            self.assertTrue((output_dir / "dialogues.jsonl").exists())
            self.assertTrue((output_dir / "qa.jsonl").exists())

    def test_generate_grounded_dataset_command_writes_source_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "v1_grounded"
            argv = [
                "event-memory",
                "generate-grounded-dataset",
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

            self.assertTrue((output_dir / "source_contexts.jsonl").exists())
            metrics = [json.loads(line) for line in (output_dir / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(metrics), 150)

    def test_generate_long_context_dataset_command_writes_v2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "v2_long_context"
            argv = [
                "event-memory",
                "generate-long-context-dataset",
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

            self.assertTrue((output_dir / "source_adaptations.jsonl").exists())
            self.assertTrue((output_dir / "dataset_audit.jsonl").exists())
            qa_rows = [json.loads(line) for line in (output_dir / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(qa_rows), 360)

    def test_audit_long_context_dataset_command_accepts_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "v2_long_context"
            cli.generate_long_context_dataset(output_dir)
            argv = [
                "event-memory",
                "audit-long-context-dataset",
                "--dataset-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

    def test_generate_public_grounded_dataset_command_writes_public_grounded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "public_grounded_benchmark"
            argv = [
                "event-memory",
                "generate-public-grounded-dataset",
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

            self.assertTrue((output_dir / "source_manifest.jsonl").exists())
            self.assertTrue((output_dir / "source_facts.jsonl").exists())
            self.assertTrue((output_dir / "memory_turns.jsonl").exists())
            qa_rows = [json.loads(line) for line in (output_dir / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(qa_rows), 360)

    def test_audit_public_grounded_dataset_command_accepts_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "public_grounded_benchmark"
            cli.generate_public_grounded_dataset(output_dir)
            argv = [
                "event-memory",
                "audit-public-grounded-dataset",
                "--dataset-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                cli.main()

    def test_generate_and_audit_public_grounded_hard_dataset_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "public_grounded_benchmark"
            hard_dir = Path(temp_dir) / "public_grounded_hard_challenge"
            cli.generate_public_grounded_dataset(base_dir)
            generate_argv = [
                "event-memory",
                "generate-public-grounded-hard-dataset",
                "--dataset-dir",
                str(base_dir),
                "--output-dir",
                str(hard_dir),
            ]
            with patch.object(sys, "argv", generate_argv):
                cli.main()

            qa_rows = [json.loads(line) for line in (hard_dir / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(qa_rows), 72)
            self.assertTrue((base_dir / "challenge_qa.jsonl").exists())

            audit_argv = [
                "event-memory",
                "audit-public-grounded-hard-dataset",
                "--dataset-dir",
                str(hard_dir),
            ]
            with patch.object(sys, "argv", audit_argv):
                cli.main()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
