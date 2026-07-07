import json
import unittest

from event_memory.config import EndpointConfig
from event_memory.llm_client import LLMClient
from event_memory.schema import Event
from event_memory.update import LLMConflictAwareUpdater


class LLMUpdateTest(unittest.TestCase):
    def test_llm_supersedes_marks_old_event(self) -> None:
        updater = LLMConflictAwareUpdater(_client_with_relation("supersedes"))
        old = _event("e1", "plan", "使用者準備研究所考試", ["研究所"])
        new = _event("e2", "negated_fact", "使用者不再準備研究所考試", ["研究所"])

        memory, relations = updater.apply([old, new])

        self.assertEqual(len(memory), 2)
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].relation, "supersedes")
        self.assertEqual(old.superseded_by, "e2")

    def test_llm_unrelated_creates_no_relation(self) -> None:
        updater = LLMConflictAwareUpdater(_client_with_relation("unrelated"))
        old = _event("e1", "preference", "使用者偏好台北", ["台北"])
        new = _event("e2", "preference", "使用者偏好新竹", ["台北", "新竹"])

        _memory, relations = updater.apply([old, new])

        self.assertEqual(relations, [])
        self.assertIsNone(old.superseded_by)

    def test_invalid_relation_falls_back_to_unrelated(self) -> None:
        updater = LLMConflictAwareUpdater(_client_with_raw_content('{"relation":"bad","reason":"x"}'))
        old = _event("e1", "preference", "使用者偏好台北", ["台北"])
        new = _event("e2", "preference", "使用者偏好新竹", ["台北", "新竹"])

        _memory, relations = updater.apply([old, new])

        self.assertEqual(relations, [])


def _event(event_id: str, event_type: str, content: str, entities: list[str]) -> Event:
    return Event(
        event_id=event_id,
        user_id="u01",
        time="2026-04-10",
        speaker="user",
        subject="使用者",
        event_type=event_type,  # type: ignore[arg-type]
        content=content,
        entities=entities,
        evidence_turn_ids=["t01"],
        importance=0.8,
    )


def _client_with_relation(relation: str) -> LLMClient:
    return _client_with_raw_content(json.dumps({"relation": relation, "reason": "測試理由"}, ensure_ascii=False))


def _client_with_raw_content(content: str) -> LLMClient:
    def fake_transport(url, api_key, payload, timeout_seconds):
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


if __name__ == "__main__":
    unittest.main()
