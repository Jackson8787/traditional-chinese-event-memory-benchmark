import json
import tempfile
import unittest
from pathlib import Path

from event_memory.io import write_jsonl
from event_memory.relation_filter import filter_predicted_memory


class RelationFilterTest(unittest.TestCase):
    def test_filters_overbroad_relations_and_keeps_explicit_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir) / "memory"
            output_dir = Path(temp_dir) / "filtered"
            memory_dir.mkdir()
            write_jsonl(memory_dir / "pred_events.jsonl", _event_rows())
            write_jsonl(memory_dir / "pred_update_relations.jsonl", _relation_rows())

            report = filter_predicted_memory(memory_dir, output_dir)

            relations = [
                json.loads(line)
                for line in (output_dir / "pred_update_relations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            relation_pairs = {(row["relation"], row["old_event_id"], row["new_event_id"]) for row in relations}

            self.assertIn(("supersedes", "e_old_place", "e_new_place"), relation_pairs)
            self.assertIn(("corrects", "e_old_tool", "e_new_tool"), relation_pairs)
            self.assertIn(("supplements", "e_topic", "e_topic_detail"), relation_pairs)
            self.assertNotIn(("supersedes", "e_topic", "e_topic_restate"), relation_pairs)
            self.assertNotIn(("supersedes", "e_old_place", "e_source_update"), relation_pairs)
            self.assertNotIn(("corrects", "e_school_source", "e_internship_source"), relation_pairs)
            self.assertNotIn(("conflicts_without_resolution", "e_plan_a", "e_plan_b"), relation_pairs)

            copied_events = (output_dir / "pred_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("e_new_place", copied_events)
            self.assertEqual(report["input_relation_count"], 7)
            self.assertEqual(report["kept_relation_count"], 3)
            self.assertEqual(report["removed_relation_count"], 4)
            self.assertEqual(report["removed_by_relation"]["supersedes"], 2)
            self.assertEqual(report["removed_by_relation"]["corrects"], 1)
            self.assertEqual(report["removed_by_relation"]["conflicts_without_resolution"], 1)
            self.assertTrue((output_dir / "relation_filter_report.json").exists())
            self.assertTrue((output_dir / "relation_filter_audit.jsonl").exists())


def _event_rows() -> list[dict]:
    return [
        _event("e_old_place", "plan", "週末整理資料地點是總圖三樓", ["週末整理", "總圖三樓"], "u01_s01_t01"),
        _event("e_new_place", "plan", "後來把週末整理資料地點改成資工系館討論室", ["週末整理", "資工系館討論室"], "u01_s02_t01"),
        _event("e_topic", "plan", "我這週要整理專題訪談紀錄", ["專題", "訪談紀錄"], "u01_s03_t01"),
        _event("e_topic_restate", "plan", "這週專題訪談紀錄要整理", ["專題", "訪談紀錄"], "u01_s04_t01"),
        _event("e_old_tool", "plan", "我把照護排程放在 Google 試算表", ["照護排程", "Google 試算表"], "u01_s05_t01"),
        _event("e_new_tool", "plan", "我剛剛說錯，不是 Google 試算表，應該是 Notion 表格", ["照護排程", "Notion 表格"], "u01_s06_t01"),
        _event("e_school_source", "other", "我先看學校職涯中心的實習公告", ["學校職涯中心", "實習"], "u01_s07_t01"),
        _event("e_internship_source", "other", "暑期實習公司網站也有職缺資訊", ["暑期實習", "公司網站"], "u01_s08_t01"),
        _event("e_topic_detail", "constraint", "訪談紀錄要另外標出受訪者同意書狀態", ["專題", "訪談紀錄", "同意書"], "u01_s09_t01"),
        _event("e_source_update", "plan", "後來把週末整理資料來源改成學校職涯中心公告", ["週末整理", "學校職涯中心"], "u01_s09_t02"),
        _event("e_plan_a", "plan", "週六上午去新竹面試", ["新竹面試"], "u01_s10_t01"),
        _event("e_plan_b", "plan", "週日下午陪家人復健", ["家人復健"], "u01_s11_t01"),
    ]


def _relation_rows() -> list[dict]:
    return [
        _relation("e_old_place", "e_new_place", "supersedes", "later update"),
        _relation("e_topic", "e_topic_restate", "supersedes", "same topic later"),
        _relation("e_old_place", "e_source_update", "supersedes", "same topic but different slot"),
        _relation("e_old_tool", "e_new_tool", "corrects", "explicit correction"),
        _relation("e_school_source", "e_internship_source", "corrects", "different source"),
        _relation("e_topic", "e_topic_detail", "supplements", "adds detail"),
        _relation("e_plan_a", "e_plan_b", "conflicts_without_resolution", "both plans"),
    ]


def _event(event_id: str, event_type: str, content: str, entities: list[str], turn_id: str) -> dict:
    return {
        "event_id": event_id,
        "user_id": "u01",
        "time": "2026-04-01",
        "speaker": "user",
        "subject": "使用者",
        "event_type": event_type,
        "content": content,
        "entities": entities,
        "evidence_turn_ids": [turn_id],
        "source_context_ids": [],
        "importance": 0.8,
        "superseded_by": None,
        "corrected_by": None,
    }


def _relation(old_event_id: str, new_event_id: str, relation: str, reason: str) -> dict:
    return {
        "new_event_id": new_event_id,
        "old_event_id": old_event_id,
        "relation": relation,
        "reason": reason,
        "evidence_turn_ids": [f"{new_event_id}_turn"],
    }


if __name__ == "__main__":
    unittest.main()
