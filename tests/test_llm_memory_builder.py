import json
import tempfile
import unittest
from pathlib import Path

from event_memory.config import EndpointConfig
from event_memory.llm_client import LLMClient
from event_memory.llm_memory_builder import build_llm_memory


class LlmMemoryBuilderTest(unittest.TestCase):
    def test_builds_session_level_memory_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output_dir = Path(temp_dir) / "memory"
            _write_dataset(dataset_dir)
            transport = _SessionEventTransport()
            llm = _client(transport)

            stats = build_llm_memory(dataset_dir, output_dir, llm, limit_qa=2)

            self.assertEqual(stats["selected_qa_count"], 2)
            self.assertEqual(stats["selected_user_count"], 1)
            self.assertEqual(stats["selected_session_count"], 2)
            self.assertEqual(stats["llm_calls_attempted"], 2)
            self.assertEqual(stats["llm_calls_failed"], 0)
            self.assertEqual(stats["parsed_event_count"], 2)
            self.assertTrue((output_dir / "pred_events.jsonl").exists())
            self.assertTrue((output_dir / "pred_update_relations.jsonl").exists())
            self.assertTrue((output_dir / "memory_build_manifest.json").exists())
            self.assertTrue((output_dir / "memory_build_stats.json").exists())
            self.assertTrue((output_dir / "sessions" / "u01__u01_s01.json").exists())

            events = _read_jsonl(output_dir / "pred_events.jsonl")
            self.assertEqual([event["evidence_turn_ids"] for event in events], [["u01_s01_u01"], ["u01_s02_u01"]])

            failing_transport = _FailingTransport()
            cached_stats = build_llm_memory(dataset_dir, output_dir, _client(failing_transport), limit_qa=2)

            self.assertEqual(cached_stats["llm_calls_attempted"], 0)
            self.assertEqual(cached_stats["parsed_event_count"], 2)
            self.assertEqual(failing_transport.calls, 0)

    def test_parse_failure_is_recorded_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output_dir = Path(temp_dir) / "memory"
            _write_dataset(dataset_dir)
            llm = _client(_BadSecondSessionTransport())

            stats = build_llm_memory(dataset_dir, output_dir, llm, limit_qa=2, force=True)

            self.assertEqual(stats["selected_session_count"], 2)
            self.assertEqual(stats["llm_calls_attempted"], 2)
            self.assertEqual(stats["llm_calls_failed"], 1)
            self.assertEqual(stats["parsed_event_count"], 1)
            errors = _read_jsonl(output_dir / "memory_build_errors.jsonl")
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["session_id"], "u01_s02")
            self.assertEqual(errors[0]["error_type"], "parse_error")
            bad_cache = json.loads((output_dir / "sessions" / "u01__u01_s02.json").read_text(encoding="utf-8"))
            self.assertEqual(bad_cache["raw_response"], "not json")

    def test_relation_judging_skips_non_update_restatements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output_dir = Path(temp_dir) / "memory"
            _write_dataset(dataset_dir)
            transport = _SameEntityNoUpdateTransport()

            stats = build_llm_memory(dataset_dir, output_dir, _client(transport), limit_qa=2)

            self.assertEqual(stats["parsed_event_count"], 2)
            self.assertEqual(stats["predicted_relation_count"], 0)
            self.assertEqual(transport.calls, 2)


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    _write_jsonl(
        dataset_dir / "dialogues.jsonl",
        [
            {
                "user_id": "u01",
                "session_id": "u01_s01",
                "turn_id": "u01_s01_u01",
                "speaker": "user",
                "timestamp": "2026-03-01",
                "text": "I first planned to use the library desk.",
            },
            {
                "user_id": "u01",
                "session_id": "u01_s01",
                "turn_id": "u01_s01_a01",
                "speaker": "assistant",
                "timestamp": "2026-03-01",
                "text": "Noted.",
            },
            {
                "user_id": "u01",
                "session_id": "u01_s02",
                "turn_id": "u01_s02_u01",
                "speaker": "user",
                "timestamp": "2026-03-08",
                "text": "I changed the plan to the lab room.",
            },
        ],
    )
    _write_jsonl(
        dataset_dir / "qa.jsonl",
        [
            {
                "question_id": "q01",
                "user_id": "u01",
                "question": "Where was the first plan?",
                "question_type": "single_session_fact",
                "gold_answer": "library desk",
                "gold_evidence_turn_ids": ["u01_s01_u01"],
                "valid_time": "2026-06-01",
            },
            {
                "question_id": "q02",
                "user_id": "u01",
                "question": "Where is the current plan?",
                "question_type": "knowledge_update",
                "gold_answer": "lab room",
                "gold_evidence_turn_ids": ["u01_s02_u01"],
                "valid_time": "2026-06-01",
            },
        ],
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _client(transport) -> LLMClient:
    return LLMClient(
        EndpointConfig(
            provider="openai",
            base_url="https://example.test",
            api_key="secret",
            model="gpt-test",
            max_retries=0,
        ),
        transport=transport,
    )


class _SessionEventTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url, api_key, payload, timeout_seconds):
        self.calls += 1
        prompt = payload["messages"][-1]["content"]
        turn_id = "u01_s02_u01" if "u01_s02_u01" in prompt else "u01_s01_u01"
        content = "User changed the plan to the lab room." if turn_id == "u01_s02_u01" else "User planned to use the library desk."
        return {
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "events": [
                                    {
                                        "source_turn_id": turn_id,
                                        "event_type": "plan",
                                        "subject": "user",
                                        "content": content,
                                        "entities": [content.rsplit(" ", 1)[-1].strip(".")],
                                        "importance": 0.8,
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
        }


class _BadSecondSessionTransport(_SessionEventTransport):
    def __call__(self, url, api_key, payload, timeout_seconds):
        self.calls += 1
        prompt = payload["messages"][-1]["content"]
        if "u01_s02_u01" in prompt:
            return {"model": payload["model"], "choices": [{"message": {"content": "not json"}}]}
        return {
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "events": [
                                    {
                                        "source_turn_id": "u01_s01_u01",
                                        "event_type": "plan",
                                        "subject": "user",
                                        "content": "User planned to use the library desk.",
                                        "entities": ["desk"],
                                        "importance": 0.8,
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
        }


class _SameEntityNoUpdateTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url, api_key, payload, timeout_seconds):
        self.calls += 1
        prompt = payload["messages"][-1]["content"]
        turn_id = "u01_s02_u01" if "u01_s02_u01" in prompt else "u01_s01_u01"
        content = "User mentioned the location detail again." if turn_id == "u01_s02_u01" else "User mentioned the location detail."
        return {
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "events": [
                                    {
                                        "source_turn_id": turn_id,
                                        "event_type": "personal_fact",
                                        "subject": "user",
                                        "content": content,
                                        "entities": ["location"],
                                        "importance": 0.5,
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
        }


class _FailingTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url, api_key, payload, timeout_seconds):
        self.calls += 1
        raise AssertionError("cache should avoid live LLM calls")


if __name__ == "__main__":
    unittest.main()
