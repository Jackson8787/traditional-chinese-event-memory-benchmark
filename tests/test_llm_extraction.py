import unittest

from event_memory.config import EndpointConfig
from event_memory.extraction import LLMEventExtractor
from event_memory.llm_client import LLMClient
from event_memory.schema import DialogueTurn


class LLMExtractionTest(unittest.TestCase):
    def test_extracts_schema_valid_events_from_json(self) -> None:
        extractor = LLMEventExtractor(_client_with_response({
            "events": [
                {
                    "event_type": "plan",
                    "subject": "使用者",
                    "content": "使用者改成尋找暑期實習",
                    "entities": ["暑期實習"],
                    "importance": 0.91,
                }
            ]
        }))

        events = extractor.extract_turn(_turn())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "plan")
        self.assertEqual(events[0].content, "使用者改成尋找暑期實習")
        self.assertEqual(events[0].entities, ["暑期實習"])
        self.assertEqual(events[0].evidence_turn_ids, ["t01"])
        self.assertEqual(events[0].importance, 0.91)

    def test_invalid_event_type_and_importance_fall_back(self) -> None:
        extractor = LLMEventExtractor(_client_with_raw_content('{"events":[{"event_type":"bad","content":"","importance":2}]}'))

        events = extractor.extract_turn(_turn())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "other")
        self.assertEqual(events[0].content, "我後來不考研究所了，改成找暑期實習。")
        self.assertEqual(events[0].importance, 1.0)

    def test_non_json_response_returns_no_events(self) -> None:
        extractor = LLMEventExtractor(_client_with_raw_content("我不知道"))

        self.assertEqual(extractor.extract_turn(_turn()), [])

    def test_deduplicates_same_type_and_entities_within_turn(self) -> None:
        extractor = LLMEventExtractor(_client_with_response({
            "events": [
                {
                    "event_type": "plan",
                    "content": "使用者準備研究所考試",
                    "entities": ["研究所考試"],
                    "importance": 0.8,
                },
                {
                    "event_type": "plan",
                    "content": "使用者最近在準備研究所考試並安排時間",
                    "entities": ["研究所考試"],
                    "importance": 0.9,
                },
            ]
        }))

        events = extractor.extract_turn(_turn())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content, "使用者最近在準備研究所考試並安排時間")


def _turn() -> DialogueTurn:
    return DialogueTurn(
        user_id="u01",
        session_id="s01",
        turn_id="t01",
        speaker="user",
        timestamp="2026-04-10",
        text="我後來不考研究所了，改成找暑期實習。",
    )


def _client_with_response(response: dict) -> LLMClient:
    import json

    return _client_with_raw_content(json.dumps(response, ensure_ascii=False))


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
