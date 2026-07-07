import unittest
from pathlib import Path

from event_memory.experiment import apply_update_relations, load_events, load_update_relations
from event_memory.io import load_qa, load_turns, turns_by_id
from event_memory.retrieval import EventRetriever
from event_memory.schema import Event


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "v0"


class EventRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        turns = load_turns(DATASET_DIR / "dialogues.jsonl")
        events = load_events(DATASET_DIR / "gold_events.jsonl")
        relations = load_update_relations(DATASET_DIR / "gold_update_relations.jsonl")
        self.memory = apply_update_relations(events, relations)
        self.qa_by_id = {qa.question_id: qa for qa in load_qa(DATASET_DIR / "qa.jsonl")}
        self.retriever = EventRetriever(turns_by_id(turns))

    def test_history_query_prefers_original_plan(self) -> None:
        result = self.retriever.retrieve(self.qa_by_id["q001"].question, "u01", self.memory)[0]
        self.assertEqual(result.event.event_id, "u01_e01")

    def test_current_plan_query_prefers_updated_plan(self) -> None:
        result = self.retriever.retrieve(self.qa_by_id["q002"].question, "u01", self.memory)[0]
        self.assertEqual(result.event.event_id, "u01_e08")
        self.assertGreaterEqual(result.score, 0.28)

    def test_current_location_query_prefers_corrected_location(self) -> None:
        result = self.retriever.retrieve(self.qa_by_id["q004"].question, "u01", self.memory)[0]
        self.assertEqual(result.event.event_id, "u01_e09")

    def test_fixed_constraint_query_prefers_constraint_event(self) -> None:
        result = self.retriever.retrieve(self.qa_by_id["q005"].question, "u01", self.memory)[0]
        self.assertEqual(result.event.event_id, "u01_e04")

    def test_relationship_query_prefers_relationship_event(self) -> None:
        result = self.retriever.retrieve(self.qa_by_id["q006"].question, "u01", self.memory)[0]
        self.assertEqual(result.event.event_id, "u01_e05")

    def test_relationship_query_accepts_personal_fact_with_relationship_cues(self) -> None:
        events = [
            Event(
                event_id="pred_e_0005",
                user_id="u01",
                time="2026-03-29",
                speaker="user",
                subject="使用者",
                event_type="personal_fact",
                content="使用者和室友阿哲一起租屋。",
                entities=["室友阿哲", "租屋"],
                evidence_turn_ids=["u01_s05_t01"],
                importance=0.82,
            )
        ]

        result = self.retriever.retrieve(self.qa_by_id["q006"].question, "u01", events)[0]

        self.assertEqual(result.event.event_id, "pred_e_0005")
        self.assertGreaterEqual(result.score, 0.28)

    def test_future_goal_query_prefers_future_event(self) -> None:
        result = self.retriever.retrieve(self.qa_by_id["q008"].question, "u01", self.memory)[0]
        self.assertEqual(result.event.event_id, "u01_e10")

    def test_future_goal_query_prefers_explicit_hope_content_even_if_type_is_plan(self) -> None:
        events = [
            Event(
                event_id="pred_e_0008",
                user_id="u01",
                time="2026-04-05",
                speaker="user",
                subject="使用者",
                event_type="future_event",
                content="使用者接下來可以進下一步。",
                entities=["下一步"],
                evidence_turn_ids=["u01_s06_t01"],
                importance=0.52,
            ),
            Event(
                event_id="pred_e_0013",
                user_id="u01",
                time="2026-05-03",
                speaker="user",
                subject="使用者",
                event_type="plan",
                content="使用者希望在五月底前投三間實習，並將五月底視為下一個時間點。",
                entities=["五月底", "三間實習"],
                evidence_turn_ids=["u01_s10_t01"],
                importance=0.82,
            ),
        ]

        result = self.retriever.retrieve(self.qa_by_id["q008"].question, "u01", events)[0]

        self.assertEqual(result.event.event_id, "pred_e_0013")

    def test_current_plan_query_prefers_update_event_over_future_goal(self) -> None:
        events = [
            Event(
                event_id="pred_e_0010",
                user_id="u01",
                time="2026-04-12",
                speaker="user",
                subject="使用者",
                event_type="plan",
                content="使用者改成尋找暑期實習。",
                entities=["暑期實習"],
                evidence_turn_ids=["u01_s07_t01"],
                importance=0.88,
            ),
            Event(
                event_id="pred_e_0014",
                user_id="u01",
                time="2026-05-03",
                speaker="user",
                subject="使用者",
                event_type="plan",
                content="使用者希望在五月底前投三間實習，作為下一個時間點的計畫。",
                entities=["五月底", "三間實習"],
                evidence_turn_ids=["u01_s10_t01"],
                importance=0.74,
            ),
        ]

        result = self.retriever.retrieve(self.qa_by_id["q002"].question, "u01", events)[0]

        self.assertEqual(result.event.event_id, "pred_e_0010")

    def test_tool_query_prefers_tool_slot_over_source_overlap(self) -> None:
        events = [
            Event(
                event_id="pred_e_source",
                user_id="u01",
                time="2026-05-01",
                speaker="user",
                subject="使用者",
                event_type="preference",
                content="使用者目前把校園專題主要整理工具的資料來源改成臺北市資料大平臺。",
                entities=["主要整理工具", "資料來源", "臺北市資料大平臺"],
                evidence_turn_ids=["u01_s21_t01"],
                importance=0.9,
            ),
            Event(
                event_id="pred_e_tool",
                user_id="u01",
                time="2026-05-02",
                speaker="user",
                subject="使用者",
                event_type="preference",
                content="使用者改成 Notion 表格。",
                entities=["Notion 表格"],
                evidence_turn_ids=["u01_s22_t01"],
                importance=0.7,
            ),
        ]

        result = self.retriever.retrieve("宜庭目前的主要整理工具是什麼？", "u01", events)[0]

        self.assertEqual(result.event.event_id, "pred_e_tool")


if __name__ == "__main__":
    unittest.main()
