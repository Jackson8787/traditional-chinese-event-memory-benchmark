import unittest
from pathlib import Path

from event_memory.answering import EvidenceGroundedAnswerer
from event_memory.eval import evaluate
from event_memory.extraction import RuleBasedEventExtractor
from event_memory.io import load_qa, load_turns, turns_by_id
from event_memory.retrieval import EventRetriever
from event_memory.update import ConflictAwareUpdater


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_sample_pipeline_runs(self) -> None:
        turns = load_turns(ROOT / "data" / "sample" / "dialogues.jsonl")
        qa_items = load_qa(ROOT / "data" / "sample" / "qa.jsonl")

        raw_events = RuleBasedEventExtractor().extract_all(turns)
        memory, relations = ConflictAwareUpdater().apply(raw_events)

        self.assertTrue(raw_events)
        self.assertTrue(any(relation.relation == "supersedes" for relation in relations))
        self.assertTrue(any(relation.relation == "corrects" for relation in relations))

        retriever = EventRetriever(turns_by_id(turns))
        answerer = EvidenceGroundedAnswerer()
        answers = [answerer.answer(qa, retriever.retrieve(qa.question, qa.user_id, memory)) for qa in qa_items]
        summary = evaluate(qa_items, answers)

        self.assertEqual(summary.total, 4)
        self.assertGreaterEqual(summary.evidence_recall_at_k, 0.5)


if __name__ == "__main__":
    unittest.main()
