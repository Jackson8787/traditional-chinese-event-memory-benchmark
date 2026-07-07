import json
import unittest
from pathlib import Path


CATALOG_PATH = Path("data/source_catalog/chinese_corpus_candidates.jsonl")
REQUIRED_FIELDS = {
    "corpus_id",
    "name",
    "script",
    "language_region",
    "source_type",
    "source_url",
    "evidence_urls",
    "scale",
    "access_status",
    "license_or_terms",
    "long_dialogue_value",
    "recommended_use",
    "risks",
    "priority",
}


class CorpusCatalogTest(unittest.TestCase):
    def test_catalog_has_required_shape_and_coverage(self) -> None:
        self.assertTrue(CATALOG_PATH.exists(), "corpus catalog file is missing")
        rows = _read_jsonl(CATALOG_PATH)

        self.assertGreaterEqual(len(rows), 12)
        self.assertTrue(all(REQUIRED_FIELDS <= set(row) for row in rows))
        self.assertEqual(len({row["corpus_id"] for row in rows}), len(rows))
        self.assertTrue(all(row["source_url"].startswith("https://") for row in rows))
        self.assertTrue(all(row["evidence_urls"] for row in rows))
        self.assertTrue(all(row["recommended_use"] for row in rows))
        self.assertTrue(all(row["risks"] for row in rows))

        corpus_ids = {row["corpus_id"] for row in rows}
        for required in {"naturalconv", "kdconv", "crosswoz", "lccc", "taiwanchat"}:
            self.assertIn(required, corpus_ids)

        taiwan_or_traditional = [
            row
            for row in rows
            if row["script"] == "traditional" or row["language_region"] == "Taiwan"
        ]
        self.assertGreaterEqual(len(taiwan_or_traditional), 4)

        long_dialogue_candidates = [
            row
            for row in rows
            if row["long_dialogue_value"] == "high"
            or (isinstance(row.get("avg_turns"), (int, float)) and row["avg_turns"] >= 15)
        ]
        self.assertGreaterEqual(len(long_dialogue_candidates), 5)

        simplified_adaptation_candidates = [
            row for row in rows if row["script"].startswith("simplified")
        ]
        self.assertGreaterEqual(len(simplified_adaptation_candidates), 6)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
