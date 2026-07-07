import json
import tempfile
import unittest
from pathlib import Path

from event_memory.config import EndpointConfig
from event_memory.experiment import (
    apply_update_relations,
    build_error_analysis,
    load_events,
    load_update_relations,
    run_experiment,
)
from event_memory.grounded_dataset_builder import build_grounded_dataset
from event_memory.io import write_jsonl
from event_memory.long_context_dataset_builder import build_long_context_dataset
from event_memory.llm_client import LLMClient
from event_memory.public_grounded_dataset_builder import build_public_grounded_dataset, build_public_grounded_hard_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "v0"


class ExperimentTest(unittest.TestCase):
    def test_loads_gold_events_and_marks_superseded_events_without_mutating_input(self) -> None:
        events = load_events(DATASET_DIR / "gold_events.jsonl")
        relations = load_update_relations(DATASET_DIR / "gold_update_relations.jsonl")

        memory = apply_update_relations(events, relations)

        self.assertEqual(events[0].event_id, "u01_e01")
        self.assertIsNone(events[0].superseded_by)

        by_id = {event.event_id: event for event in memory}
        self.assertEqual(by_id["u01_e01"].superseded_by, "u01_e08")
        self.assertEqual(by_id["u01_e02"].corrected_by, "u01_e09")
        self.assertIsNone(by_id["u01_e03"].superseded_by)

    def test_event_memory_gold_experiment_writes_metrics_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_gold",
                output_dir=Path(temp_dir),
                limit=12,
            )

            self.assertEqual(result.method, "event_memory_gold")
            self.assertEqual(result.summary.total, 12)
            self.assertEqual(len(result.answers), 12)
            self.assertIn("single_session_fact", result.question_type_metrics)

            metrics_path = Path(temp_dir) / "event_memory_gold" / "metrics.json"
            answers_path = Path(temp_dir) / "event_memory_gold" / "answers.jsonl"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(answers_path.exists())

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["method"], "event_memory_gold")
            self.assertEqual(metrics["summary"]["total"], 12)
            self.assertIn("update_accuracy", metrics["summary"])
            self.assertIn("temporal_accuracy", metrics["summary"])
            self.assertIn("average_token_cost", metrics["summary"])
            self.assertIn("latency_ms", metrics)

    def test_chunk_rag_experiment_runs_without_gold_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment(
                dataset_dir=DATASET_DIR,
                method="chunk_rag",
                output_dir=Path(temp_dir),
                limit=10,
            )

            self.assertEqual(result.method, "chunk_rag")
            self.assertEqual(result.summary.total, 10)
            self.assertEqual(len(result.answers), 10)
            self.assertTrue(
                any(event_id.startswith("chunk_") for answer in result.answers for event_id in answer.retrieved_event_ids)
            )

    def test_experiment_writes_error_analysis_rows_for_failed_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_experiment(
                dataset_dir=DATASET_DIR,
                method="chunk_rag",
                output_dir=Path(temp_dir),
                limit=10,
            )

            error_path = Path(temp_dir) / "chunk_rag" / "error_analysis.jsonl"
            self.assertTrue(error_path.exists())
            rows = [json.loads(line) for line in error_path.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(rows)
            self.assertTrue(any(row["error_source"] == "retrieval_error" for row in rows))
            self.assertTrue(
                all(
                    row["error_source"]
                    in {
                        "retrieval_error",
                        "update_error",
                        "temporal_error",
                        "abstention_error",
                        "evidence_incomplete",
                        "answer_generation_or_scoring_error",
                    }
                    for row in rows
                )
            )
            self.assertEqual(
                {
                    "question_id",
                    "question_type",
                    "error_source",
                    "question",
                    "gold_answer",
                    "answer",
                    "gold_evidence_turn_ids",
                    "actual_evidence_turn_ids",
                    "retrieved_event_ids",
                    "exact_answer_correct",
                    "normalized_answer_correct",
                    "all_gold_evidence_retrieved",
                    "faithful_answer_correct",
                },
                set(rows[0]),
            )

    def test_error_analysis_flags_incomplete_multi_evidence_answers(self) -> None:
        qa = _qa_item(
            gold_answer="原本是下週五前再看一次，後來改成下週三晚上先確認。",
            gold_evidence_turn_ids=["old_turn", "new_turn"],
        )
        answer = _answer_item(
            answer="原本是下週五前再看一次，後來改成下週三晚上先確認。",
            evidence_turn_ids=["old_turn"],
        )

        rows = build_error_analysis([qa], [answer])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error_source"], "evidence_incomplete")
        self.assertTrue(rows[0]["normalized_answer_correct"])
        self.assertFalse(rows[0]["all_gold_evidence_retrieved"])

    def test_full_context_experiment_runs_over_raw_dialogue_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment(
                dataset_dir=DATASET_DIR,
                method="full_context",
                output_dir=Path(temp_dir),
                limit=10,
            )

            self.assertEqual(result.method, "full_context")
            self.assertEqual(result.summary.total, 10)
            self.assertEqual(len(result.answers), 10)
            self.assertTrue(
                any(event_id.startswith("full_") for answer in result.answers for event_id in answer.retrieved_event_ids)
            )

    def test_summary_memory_experiment_runs_over_session_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment(
                dataset_dir=DATASET_DIR,
                method="summary_memory",
                output_dir=Path(temp_dir),
                limit=10,
            )

            self.assertEqual(result.method, "summary_memory")
            self.assertEqual(result.summary.total, 10)
            self.assertEqual(len(result.answers), 10)
            self.assertTrue(
                any(
                    event_id.startswith("summary_")
                    for answer in result.answers
                    for event_id in answer.retrieved_event_ids
                )
            )

    def test_event_memory_llm_experiment_writes_predicted_memory_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_llm",
                output_dir=Path(temp_dir),
                limit=8,
                llm=_fake_memory_llm(),
            )

            self.assertEqual(result.method, "event_memory_llm")
            self.assertEqual(result.summary.total, 8)
            self.assertEqual(len(result.answers), 8)

            output_dir = Path(temp_dir) / "event_memory_llm"
            events_path = output_dir / "pred_events.jsonl"
            relations_path = output_dir / "pred_update_relations.jsonl"
            self.assertTrue(events_path.exists())
            self.assertTrue(relations_path.exists())

            event_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            relation_rows = [json.loads(line) for line in relations_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["event_id"].startswith("pred_e_") for row in event_rows))
            self.assertTrue(any(row["relation"] == "supersedes" for row in relation_rows))
            self.assertTrue(any(answer.retrieved_event_ids for answer in result.answers))

    def test_event_memory_llm_can_rerun_from_cached_predicted_memory_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cached_memory"
            cache_dir.mkdir()
            (cache_dir / "pred_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_id": "pred_e_0001",
                                "user_id": "u01",
                                "time": "2026-03-01",
                                "speaker": "user",
                                "subject": "使用者",
                                "event_type": "plan",
                                "content": "使用者準備研究所考試",
                                "entities": ["研究所考試"],
                                "evidence_turn_ids": ["u01_s01_t01"],
                                "importance": 0.82,
                                "superseded_by": None,
                                "corrected_by": None,
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "event_id": "pred_e_0002",
                                "user_id": "u01",
                                "time": "2026-04-12",
                                "speaker": "user",
                                "subject": "使用者",
                                "event_type": "plan",
                                "content": "使用者改成尋找暑期實習",
                                "entities": ["研究所考試", "暑期實習"],
                                "evidence_turn_ids": ["u01_s07_t01"],
                                "importance": 0.9,
                                "superseded_by": None,
                                "corrected_by": None,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (cache_dir / "pred_update_relations.jsonl").write_text(
                json.dumps(
                    {
                        "new_event_id": "pred_e_0002",
                        "old_event_id": "pred_e_0001",
                        "relation": "supersedes",
                        "reason": "cached relation",
                        "evidence_turn_ids": ["u01_s07_t01"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = Path(temp_dir) / "outputs"
            result = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_llm",
                output_dir=output_dir,
                limit=2,
                memory_dir=cache_dir,
            )

            self.assertEqual(result.summary.total, 2)
            self.assertGreaterEqual(result.summary.evidence_recall_at_k, 0.5)
            written_events = (output_dir / "event_memory_llm" / "pred_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("pred_e_0002", written_events)

    def test_event_memory_llm_cached_relations_expand_old_and_new_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            cache_dir = Path(temp_dir) / "cached_memory"
            dataset_dir.mkdir()
            cache_dir.mkdir()
            _write_cached_relation_fixture(dataset_dir, cache_dir)

            result = run_experiment(
                dataset_dir=dataset_dir,
                method="event_memory_llm",
                limit=1,
                memory_dir=cache_dir,
            )

            answer = result.answers[0]
            self.assertEqual({"u01_s01_t01", "u01_s02_t01"}, set(answer.evidence_turn_ids))
            self.assertEqual(result.summary.all_evidence_recall_at_k, 1.0)
            self.assertEqual(result.summary.faithful_answer_accuracy, 1.0)

    def test_event_memory_llm_evidence_turn_scope_only_extracts_gold_evidence_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "fixture"
            dataset_dir.mkdir()
            _write_llm_scope_fixture(dataset_dir)
            extracted_prompts: list[str] = []

            def fake_transport(url, api_key, payload, timeout_seconds):
                system_prompt = payload["messages"][0]["content"]
                user_prompt = payload["messages"][1]["content"]
                if "events" in system_prompt and "event_type" in system_prompt:
                    extracted_prompts.append(user_prompt)
                    content = json.dumps(
                        {
                            "events": [
                                {
                                    "event_type": "preference",
                                    "subject": "使用者",
                                    "content": "使用者目前週末改去北車共享空間整理研究筆記",
                                    "entities": ["北車共享空間", "研究筆記"],
                                    "importance": 0.9,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                else:
                    content = json.dumps({"relation": "unrelated", "reason": "scope test"}, ensure_ascii=False)
                return {"model": payload["model"], "choices": [{"message": {"content": content}}]}

            llm = LLMClient(
                EndpointConfig(provider="openai", base_url="https://example.test", api_key="secret", model="gpt-test"),
                transport=fake_transport,
            )

            run_experiment(
                dataset_dir=dataset_dir,
                method="event_memory_llm",
                limit=1,
                llm=llm,
                llm_turn_scope="evidence",
            )

            self.assertEqual(len(extracted_prompts), 2)
            self.assertTrue(all("u01_s03_t01" not in prompt for prompt in extracted_prompts))

    def test_event_memory_ablation_methods_run_and_write_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            baseline = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_gold",
                output_dir=output_dir,
            )
            no_supersedes = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_gold_no_supersedes",
                output_dir=output_dir,
            )
            no_time = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_gold_no_time",
                output_dir=output_dir,
            )
            no_entity = run_experiment(
                dataset_dir=DATASET_DIR,
                method="event_memory_gold_no_entity",
                output_dir=output_dir,
            )

            self.assertLess(no_supersedes.summary.update_accuracy, baseline.summary.update_accuracy)
            self.assertLess(no_time.summary.temporal_accuracy, baseline.summary.temporal_accuracy)
            self.assertEqual(no_entity.summary.total, baseline.summary.total)
            for method in [
                "event_memory_gold_no_supersedes",
                "event_memory_gold_no_time",
                "event_memory_gold_no_entity",
            ]:
                self.assertTrue((output_dir / method / "metrics.json").exists())

    def test_event_memory_gold_runs_on_grounded_dataset_with_source_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "v1_grounded"
            output_dir = Path(temp_dir) / "outputs"
            build_grounded_dataset(dataset_dir)

            result = run_experiment(
                dataset_dir=dataset_dir,
                method="event_memory_gold",
                output_dir=output_dir,
                limit=15,
            )

            self.assertEqual(result.summary.total, 15)
            self.assertTrue((output_dir / "event_memory_gold" / "metrics.json").exists())

    def test_event_memory_gold_v2_runs_on_long_context_dataset_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "v2"
            output_dir = Path(temp_dir) / "outputs"
            build_long_context_dataset(dataset_dir)

            result = run_experiment(
                dataset_dir=dataset_dir,
                method="event_memory_gold_v2",
                output_dir=output_dir,
            )

            self.assertEqual(result.method, "event_memory_gold_v2")
            self.assertEqual(result.summary.total, 360)
            self.assertTrue((output_dir / "event_memory_gold_v2" / "metrics.json").exists())
            self.assertTrue((output_dir / "event_memory_gold_v2" / "answers.jsonl").exists())
            self.assertIn("multi_session_reasoning", result.question_type_metrics)

    def test_event_memory_gold_v2_collects_more_chain_evidence_than_v1_answerer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "fixture"
            dataset_dir.mkdir()
            _write_chain_fixture(dataset_dir)

            v1 = run_experiment(dataset_dir=dataset_dir, method="event_memory_gold", limit=1)
            v2 = run_experiment(dataset_dir=dataset_dir, method="event_memory_gold_v2", limit=1)

            self.assertEqual(len(v1.answers[0].evidence_turn_ids), 1)
            self.assertEqual(set(v2.answers[0].evidence_turn_ids), {"u01_s01_t01", "u01_s02_t01"})

    def test_event_memory_gold_v2_ablation_methods_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "v2"
            build_long_context_dataset(dataset_dir)

            for method in [
                "event_memory_gold_v2_no_supersedes",
                "event_memory_gold_v2_no_time",
                "event_memory_gold_v2_no_entity",
            ]:
                result = run_experiment(dataset_dir=dataset_dir, method=method, limit=24)
                self.assertEqual(result.method, method)
                self.assertEqual(result.summary.total, 24)

    def test_public_grounded_hard_split_exposes_supersedes_ablation_drop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "public_grounded"
            hard_dir = Path(temp_dir) / "public_grounded_hard"
            build_public_grounded_dataset(base_dir)
            build_public_grounded_hard_dataset(base_dir, hard_dir)

            baseline = run_experiment(dataset_dir=hard_dir, method="event_memory_gold_v2")
            no_supersedes = run_experiment(dataset_dir=hard_dir, method="event_memory_gold_v2_no_supersedes")

            self.assertEqual(baseline.summary.total, 72)
            self.assertLess(no_supersedes.summary.evidence_recall_at_k, baseline.summary.evidence_recall_at_k)

    def test_grounded_dataset_entity_ablation_degrades_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "v1_grounded"
            build_grounded_dataset(dataset_dir)

            baseline = run_experiment(dataset_dir=dataset_dir, method="event_memory_gold")
            no_entity = run_experiment(dataset_dir=dataset_dir, method="event_memory_gold_no_entity")

            self.assertLess(no_entity.summary.evidence_recall_at_k, baseline.summary.evidence_recall_at_k)


if __name__ == "__main__":
    unittest.main()


def _qa_item(gold_answer: str, gold_evidence_turn_ids: list[str]):
    from event_memory.schema import QAItem

    return QAItem(
        question_id="q_partial",
        user_id="u01",
        question="提醒前後怎麼變化？",
        question_type="knowledge_update",
        gold_answer=gold_answer,
        gold_evidence_turn_ids=gold_evidence_turn_ids,
        valid_time="current",
        gold_update_relations=[{"old_event_id": "old", "new_event_id": "new"}],
    )


def _answer_item(answer: str, evidence_turn_ids: list[str]):
    from event_memory.schema import Answer

    return Answer(
        question_id="q_partial",
        answer=answer,
        evidence_turn_ids=evidence_turn_ids,
        confidence="high",
        retrieved_event_ids=["old_event"],
    )


def _write_cached_relation_fixture(dataset_dir: Path, cache_dir: Path) -> None:
    write_jsonl(
        dataset_dir / "dialogues.jsonl",
        [
            {
                "user_id": "u01",
                "session_id": "u01_s01",
                "turn_id": "u01_s01_t01",
                "speaker": "user",
                "timestamp": "2026-03-01",
                "text": "我一開始週末整理資料的地點是總圖三樓。",
            },
            {
                "user_id": "u01",
                "session_id": "u01_s02",
                "turn_id": "u01_s02_t01",
                "speaker": "user",
                "timestamp": "2026-04-01",
                "text": "我後來把週末整理資料的地點改成資工系館討論室。",
            },
        ],
    )
    write_jsonl(
        dataset_dir / "qa.jsonl",
        [
            {
                "question_id": "q_change",
                "user_id": "u01",
                "question": "週末整理資料的地點前後怎麼變？",
                "question_type": "knowledge_update",
                "gold_answer": "一開始是總圖三樓，後來改成資工系館討論室。",
                "gold_evidence_turn_ids": ["u01_s01_t01", "u01_s02_t01"],
                "valid_time": "current",
                "gold_update_relations": [{"old_event_id": "pred_old", "new_event_id": "pred_new"}],
            }
        ],
    )
    write_jsonl(
        cache_dir / "pred_events.jsonl",
        [
            {
                "event_id": "pred_old",
                "user_id": "u01",
                "time": "2026-03-01",
                "speaker": "user",
                "subject": "使用者",
                "event_type": "plan",
                "content": "使用者一開始週末整理資料的地點是總圖三樓",
                "entities": ["週末整理資料", "總圖三樓"],
                "evidence_turn_ids": ["u01_s01_t01"],
                "source_context_ids": [],
                "importance": 0.8,
                "superseded_by": None,
                "corrected_by": None,
            },
            {
                "event_id": "pred_new",
                "user_id": "u01",
                "time": "2026-04-01",
                "speaker": "user",
                "subject": "使用者",
                "event_type": "plan",
                "content": "使用者後來把週末整理資料的地點改成資工系館討論室",
                "entities": ["週末整理資料", "資工系館討論室"],
                "evidence_turn_ids": ["u01_s02_t01"],
                "source_context_ids": [],
                "importance": 0.9,
                "superseded_by": None,
                "corrected_by": None,
            },
        ],
    )
    write_jsonl(
        cache_dir / "pred_update_relations.jsonl",
        [
            {
                "new_event_id": "pred_new",
                "old_event_id": "pred_old",
                "relation": "supersedes",
                "reason": "使用者後來把週末整理資料的地點改成資工系館討論室。",
                "evidence_turn_ids": ["u01_s02_t01"],
            }
        ],
    )


def _fake_memory_llm() -> LLMClient:
    def fake_transport(url, api_key, payload, timeout_seconds):
        messages = payload["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        if "事件抽取器" in system_prompt:
            content = _fake_extraction_content(user_prompt)
        elif "衝突/更新關係判斷器" in system_prompt:
            content = json.dumps({"relation": "supersedes", "reason": "新計畫取代舊計畫"}, ensure_ascii=False)
        else:
            content = "{}"
        return {"model": payload["model"], "choices": [{"message": {"content": content}}]}

    return LLMClient(
        EndpointConfig(
            provider="openai",
            base_url="https://example.test",
            api_key="secret",
            model="gpt-test",
            max_retries=0,
        ),
        transport=fake_transport,
    )


def _fake_extraction_content(prompt: str) -> str:
    if "u01_s01_t01" in prompt:
        return json.dumps(
            {
                "events": [
                    {
                        "event_type": "plan",
                        "subject": "使用者",
                        "content": "使用者準備研究所考試",
                        "entities": ["研究所考試"],
                        "importance": 0.82,
                    }
                ]
            },
            ensure_ascii=False,
        )
    if "u01_s07_t01" in prompt:
        return json.dumps(
            {
                "events": [
                    {
                        "event_type": "plan",
                        "subject": "使用者",
                        "content": "使用者改成尋找暑期實習",
                        "entities": ["研究所考試", "暑期實習"],
                        "importance": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
        )
    return json.dumps({"events": []}, ensure_ascii=False)


def _write_chain_fixture(dataset_dir: Path) -> None:
    _write_llm_scope_fixture(dataset_dir)
    write_jsonl(
        dataset_dir / "dialogues.jsonl",
        [
            {
                "user_id": "u01",
                "session_id": "u01_s01",
                "turn_id": "u01_s01_t01",
                "speaker": "user",
                "timestamp": "2026-03-01",
                "text": "我原本週末都在公館咖啡店整理研究筆記。",
            },
            {
                "user_id": "u01",
                "session_id": "u01_s02",
                "turn_id": "u01_s02_t01",
                "speaker": "user",
                "timestamp": "2026-05-01",
                "text": "後來我改成週末去北車共享空間整理研究筆記。",
            },
        ],
    )


def _write_llm_scope_fixture(dataset_dir: Path) -> None:
    write_jsonl(
        dataset_dir / "dialogues.jsonl",
        [
            {
                "user_id": "u01",
                "session_id": "u01_s01",
                "turn_id": "u01_s01_t01",
                "speaker": "user",
                "timestamp": "2026-03-01",
                "text": "我原本週末都在公館咖啡店整理研究筆記。",
            },
            {
                "user_id": "u01",
                "session_id": "u01_s02",
                "turn_id": "u01_s02_t01",
                "speaker": "user",
                "timestamp": "2026-05-01",
                "text": "後來我改成週末去北車共享空間整理研究筆記。",
            },
            {
                "user_id": "u01",
                "session_id": "u01_s03",
                "turn_id": "u01_s03_t01",
                "speaker": "user",
                "timestamp": "2026-05-08",
                "text": "這一輪是無關的長對話干擾內容。",
            },
        ],
    )
    write_jsonl(
        dataset_dir / "personas.jsonl",
        [{"user_id": "u01", "persona_type": "student", "name": "測試使用者", "profile": "研究生"}],
    )
    write_jsonl(
        dataset_dir / "gold_events.jsonl",
        [
            {
                "event_id": "u01_e001",
                "user_id": "u01",
                "time": "2026-03-01",
                "speaker": "user",
                "subject": "使用者",
                "event_type": "preference",
                "content": "使用者原本週末在公館咖啡店整理研究筆記",
                "entities": ["公館咖啡店", "研究筆記"],
                "evidence_turn_ids": ["u01_s01_t01"],
                "source_context_ids": [],
                "importance": 0.7,
            },
            {
                "event_id": "u01_e002",
                "user_id": "u01",
                "time": "2026-05-01",
                "speaker": "user",
                "subject": "使用者",
                "event_type": "preference",
                "content": "使用者目前週末改去北車共享空間整理研究筆記",
                "entities": ["北車共享空間", "研究筆記"],
                "evidence_turn_ids": ["u01_s02_t01"],
                "source_context_ids": [],
                "importance": 0.9,
            },
        ],
    )
    write_jsonl(
        dataset_dir / "gold_update_relations.jsonl",
        [
            {
                "new_event_id": "u01_e002",
                "old_event_id": "u01_e001",
                "relation": "supersedes",
                "reason": "新地點取代舊地點",
                "evidence_turn_ids": ["u01_s02_t01"],
            }
        ],
    )
    write_jsonl(
        dataset_dir / "qa.jsonl",
        [
            {
                "question_id": "u01_q001",
                "user_id": "u01",
                "question": "使用者週末整理研究筆記的地點從哪裡改到哪裡？",
                "question_type": "multi_session_reasoning",
                "gold_answer": "使用者原本週末在公館咖啡店整理研究筆記；使用者目前週末改去北車共享空間整理研究筆記",
                "gold_evidence_turn_ids": ["u01_s01_t01", "u01_s02_t01"],
                "valid_time": "2026-05-01",
                "requires_abstention": False,
                "gold_event_ids": ["u01_e001", "u01_e002"],
                "gold_update_relations": [{"new_event_id": "u01_e002", "old_event_id": "u01_e001"}],
            }
        ],
    )
    write_jsonl(
        dataset_dir / "personas.jsonl",
        [{"user_id": "u01", "persona_type": "student", "name": "測試使用者", "profile": "研究生"}],
    )
    write_jsonl(
        dataset_dir / "gold_events.jsonl",
        [
            {
                "event_id": "u01_e001",
                "user_id": "u01",
                "time": "2026-03-01",
                "speaker": "user",
                "subject": "使用者",
                "event_type": "preference",
                "content": "使用者原本週末在公館咖啡店整理研究筆記",
                "entities": ["公館咖啡店", "研究筆記"],
                "evidence_turn_ids": ["u01_s01_t01"],
                "source_context_ids": [],
                "importance": 0.7,
            },
            {
                "event_id": "u01_e002",
                "user_id": "u01",
                "time": "2026-05-01",
                "speaker": "user",
                "subject": "使用者",
                "event_type": "preference",
                "content": "使用者目前週末改去北車共享空間整理研究筆記",
                "entities": ["北車共享空間", "研究筆記"],
                "evidence_turn_ids": ["u01_s02_t01"],
                "source_context_ids": [],
                "importance": 0.9,
            },
        ],
    )
    write_jsonl(
        dataset_dir / "gold_update_relations.jsonl",
        [
            {
                "new_event_id": "u01_e002",
                "old_event_id": "u01_e001",
                "relation": "supersedes",
                "reason": "新地點取代舊地點",
                "evidence_turn_ids": ["u01_s02_t01"],
            }
        ],
    )
    write_jsonl(
        dataset_dir / "qa.jsonl",
        [
            {
                "question_id": "u01_q001",
                "user_id": "u01",
                "question": "使用者週末整理研究筆記的地點從哪裡改到哪裡？",
                "question_type": "multi_session_reasoning",
                "gold_answer": "使用者原本週末在公館咖啡店整理研究筆記；使用者目前週末改去北車共享空間整理研究筆記",
                "gold_evidence_turn_ids": ["u01_s01_t01", "u01_s02_t01"],
                "valid_time": "2026-05-01",
                "requires_abstention": False,
                "gold_event_ids": ["u01_e001", "u01_e002"],
                "gold_update_relations": [{"new_event_id": "u01_e002", "old_event_id": "u01_e001"}],
            }
        ],
    )
