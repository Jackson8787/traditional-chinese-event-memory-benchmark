import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReportConsistencyTest(unittest.TestCase):
    def test_public_grounded_reports_label_gold_v2_as_oracle_and_llm_as_diagnostic_sample(self) -> None:
        for relative_path in [
            "README.md",
            "reports/experiment_summary_zh.md",
        ]:
            text = (ROOT / relative_path).read_text(encoding="utf-8")

            self.assertIn("oracle", text.lower(), relative_path)
            self.assertIn("upper-bound", text.lower(), relative_path)
            self.assertIn("diagnostic sample", text.lower(), relative_path)
            self.assertIn("--llm-turn-scope evidence", text, relative_path)
            self.assertTrue(
                "not the full 360-question main result" in text.lower() or "不是完整主結果" in text,
                relative_path,
            )
            self.assertIn("baseline fairness", text.lower(), relative_path)

    def test_realistic_llm_reports_label_realistic_sample_scope(self) -> None:
        for relative_path in [
            "README.md",
            "REPRODUCIBILITY.md",
            "reports/experiment_summary_zh.md",
        ]:
            text = (ROOT / relative_path).read_text(encoding="utf-8")

            self.assertIn("realistic sample", text.lower(), relative_path)
            self.assertIn("user-scope", text.lower(), relative_path)
            self.assertTrue(
                "not full 360-question main result" in text.lower()
                or "not the full 360-question main result" in text.lower()
                or "不是完整 360 題主結果" in text,
                relative_path,
            )
            self.assertIn("oracle", text.lower(), relative_path)
            self.assertIn("upper-bound", text.lower(), relative_path)

    def test_public_docs_explain_known_failure_modes(self) -> None:
        for relative_path in [
            "README.md",
            "README.zh-TW.md",
            "reports/failure_analysis.md",
            "reports/failure_analysis_zh.md",
        ]:
            text = (ROOT / relative_path).read_text(encoding="utf-8")

            self.assertIn("hard_temporal", text, relative_path)
            self.assertIn("0/24", text, relative_path)
            self.assertIn("60", text, relative_path)
            self.assertIn("360", text, relative_path)
            self.assertIn("auto-labeled", text, relative_path)


if __name__ == "__main__":
    unittest.main()
