import unittest

from event_memory.event_structure import compatible_memory_slots, infer_event_structure
from event_memory.schema import Event


class EventStructureTest(unittest.TestCase):
    def test_infers_place_slot_and_value(self) -> None:
        event = _event(
            "週末整理資料地點改成資工系館討論室。",
            ["週末整理", "資工系館討論室"],
        )

        structure = infer_event_structure(event)

        self.assertEqual(structure.slot, "place")
        self.assertEqual(structure.value, "資工系館討論室")
        self.assertGreaterEqual(structure.confidence, 0.7)

    def test_infers_tool_slot_and_value(self) -> None:
        event = _event(
            "我剛剛說錯，不是 Google 試算表，應該是 Notion 表格。",
            ["照護排程", "Google 試算表", "Notion 表格"],
        )

        structure = infer_event_structure(event)

        self.assertEqual(structure.slot, "tool")
        self.assertEqual(structure.value, "Notion 表格")

    def test_prefers_positive_tool_value_over_negative_comparison(self) -> None:
        event = _event(
            "目前比較偏好用 Notion 表格整理，覺得 Google 試算表容易越整理越散。",
            ["Notion 表格", "Google 試算表"],
        )

        structure = infer_event_structure(event)

        self.assertEqual(structure.slot, "tool")
        self.assertEqual(structure.value, "Notion 表格")

    def test_marks_different_slots_as_incompatible(self) -> None:
        old = _event("週末整理資料地點是總圖三樓。", ["週末整理", "總圖三樓"])
        new = _event("週末整理資料來源改成學校職涯中心公告。", ["週末整理", "學校職涯中心"])

        self.assertFalse(compatible_memory_slots(old, new))


def _event(content: str, entities: list[str]) -> Event:
    return Event(
        event_id="e1",
        user_id="u01",
        time="2026-04-01",
        speaker="user",
        subject="使用者",
        event_type="plan",
        content=content,
        entities=entities,
        evidence_turn_ids=["t1"],
        importance=0.8,
    )


if __name__ == "__main__":
    unittest.main()
