from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .answering import EvidenceGroundedAnswerer
from .config import load_config
from .dataset_builder import build_dataset, validate_dataset
from .grounded_dataset_builder import build_grounded_dataset, validate_grounded_dataset
from .dataset_naturalizer import naturalize_dataset
from .long_context_dataset_builder import build_long_context_dataset, validate_long_context_dataset
from .public_grounded_dataset_builder import (
    build_public_grounded_hard_dataset,
    build_public_grounded_dataset,
    validate_public_grounded_hard_dataset,
    validate_public_grounded_dataset,
)
from .embedding_client import EmbeddingClient
from .eval import evaluate
from .experiment import run_experiment as run_experiment_pipeline
from .extraction import LLMEventExtractor, RuleBasedEventExtractor
from .io import load_qa, load_turns, turns_by_id
from .llm_client import ChatMessage, LLMClient
from .llm_memory_builder import build_llm_memory
from .memory_text_refresher import refresh_cached_memory_text
from .manual_audit_builder import build_manual_audit_package
from .manual_audit_labeler import auto_label_manual_audit
from .manual_audit_summary import summarize_manual_audit
from .relation_filter import filter_predicted_memory
from .retrieval import EventRetriever
from .structured_events import structure_predicted_memory
from .update import ConflictAwareUpdater, LLMConflictAwareUpdater


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "sample"
DATA_V0_DIR = ROOT / "data" / "v0"
DATA_V0_NATURALIZED_DIR = ROOT / "data" / "v0_naturalized"
DATA_V1_GROUNDED_DIR = ROOT / "data" / "v1_grounded"
DATA_V2_LONG_CONTEXT_DIR = ROOT / "data" / "v2_long_context"
DATA_PUBLIC_GROUNDED_DIR = ROOT / "data" / "public_grounded_benchmark"
DATA_PUBLIC_GROUNDED_HARD_DIR = ROOT / "data" / "public_grounded_hard_challenge"
EXPERIMENTS_DIR = ROOT / "outputs" / "experiments"


def doctor() -> None:
    modules = [
        "torch",
        "transformers",
        "sentence_transformers",
        "datasets",
        "numpy",
        "pandas",
        "sklearn",
        "openai",
        "chromadb",
        "faiss",
    ]
    for module in modules:
        found = importlib.util.find_spec(module) is not None
        print(f"{module}: {'ok' if found else 'missing'}")


def run_demo() -> None:
    turns = load_turns(SAMPLE_DIR / "dialogues.jsonl")
    qa_items = load_qa(SAMPLE_DIR / "qa.jsonl")

    extractor = RuleBasedEventExtractor()
    raw_events = extractor.extract_all(turns)
    memory, relations = ConflictAwareUpdater().apply(raw_events)

    retriever = EventRetriever(turns_by_id(turns))
    answerer = EvidenceGroundedAnswerer()

    answers = []
    for qa in qa_items:
        retrieved = retriever.retrieve(qa.question, qa.user_id, memory)
        answer = answerer.answer(qa, retrieved)
        answers.append(answer)
        print(f"\n[{qa.question_id}] {qa.question}")
        print(f"answer: {answer.answer}")
        print(f"evidence: {answer.evidence_turn_ids}")
        print(f"retrieved: {answer.retrieved_event_ids}")

    summary = evaluate(qa_items, answers)
    print("\n--- summary ---")
    print(f"events: {len(memory)}")
    print(f"relations: {len(relations)}")
    print(f"answer_accuracy: {summary.answer_accuracy:.2f}")
    print(f"evidence_recall_at_3: {summary.evidence_recall_at_k:.2f}")
    print(f"abstention_accuracy: {summary.abstention_accuracy:.2f}")


def generate_dataset(output_dir: Path = DATA_V0_DIR) -> None:
    counts = build_dataset(output_dir)
    errors = validate_dataset(output_dir)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if errors:
        print("\nvalidation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def generate_grounded_dataset(output_dir: Path = DATA_V1_GROUNDED_DIR) -> None:
    counts = build_grounded_dataset(output_dir)
    errors = validate_grounded_dataset(output_dir)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if errors:
        print("\nvalidation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def generate_long_context_dataset(output_dir: Path = DATA_V2_LONG_CONTEXT_DIR) -> None:
    counts = build_long_context_dataset(output_dir)
    errors = validate_long_context_dataset(output_dir)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if errors:
        print("\nvalidation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def audit_long_context_dataset(dataset_dir: Path = DATA_V2_LONG_CONTEXT_DIR) -> None:
    errors = validate_long_context_dataset(dataset_dir)
    if errors:
        print("validation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def generate_public_grounded_dataset(output_dir: Path = DATA_PUBLIC_GROUNDED_DIR) -> None:
    counts = build_public_grounded_dataset(output_dir)
    errors = validate_public_grounded_dataset(output_dir)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if errors:
        print("\nvalidation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def audit_public_grounded_dataset(dataset_dir: Path = DATA_PUBLIC_GROUNDED_DIR) -> None:
    errors = validate_public_grounded_dataset(dataset_dir)
    if errors:
        print("validation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def generate_public_grounded_hard_dataset(
    dataset_dir: Path = DATA_PUBLIC_GROUNDED_DIR,
    output_dir: Path = DATA_PUBLIC_GROUNDED_HARD_DIR,
) -> None:
    counts = build_public_grounded_hard_dataset(dataset_dir, output_dir)
    errors = validate_public_grounded_hard_dataset(output_dir)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if errors:
        print("\nvalidation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def audit_public_grounded_hard_dataset(dataset_dir: Path = DATA_PUBLIC_GROUNDED_HARD_DIR) -> None:
    errors = validate_public_grounded_hard_dataset(dataset_dir)
    if errors:
        print("validation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def naturalize_dataset_command(dataset_dir: Path, output_dir: Path) -> None:
    counts = naturalize_dataset(dataset_dir, output_dir)
    errors = validate_dataset(output_dir)
    for name, count in counts.items():
        print(f"{name}: {count}")
    if errors:
        print("\nvalidation_errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: ok")


def check_api(config_path: Path) -> None:
    config = load_config(config_path)
    print("config: ok")
    print(f"llm_provider: {config.llm.provider}")
    print(f"llm_base_url: {config.llm.base_url}")
    print(f"llm_model: {config.llm.model}")
    print(f"embedding_provider: {config.embedding.provider}")
    print(f"embedding_base_url: {config.embedding.base_url}")
    print(f"embedding_model: {config.embedding.model}")

    llm = LLMClient(config.llm)
    chat_result = llm.chat(
        [ChatMessage(role="user", content="Reply with exactly: OK")],
        max_completion_tokens=8,
    )
    print(f"chat: ok model={chat_result.model} reply={chat_result.content.strip()[:40]}")

    embedding = EmbeddingClient(config.embedding)
    embedding_result = embedding.embed("繁體中文長期對話記憶測試")
    print(f"embedding: ok model={embedding_result.model} dimensions={len(embedding_result.embedding)}")


def extract_events(config_path: Path, dataset_dir: Path, limit: int) -> None:
    config = load_config(config_path)
    turns = load_turns(dataset_dir / "dialogues.jsonl")
    extractor = LLMEventExtractor(LLMClient(config.llm))
    events = extractor.extract_all(turns, max_turns=limit)
    print(f"turns_scanned: {min(limit, len(turns))}")
    print(f"events_extracted: {len(events)}")
    for event in events:
        print(
            {
                "event_id": event.event_id,
                "user_id": event.user_id,
                "time": event.time,
                "event_type": event.event_type,
                "content": event.content,
                "entities": event.entities,
                "evidence_turn_ids": event.evidence_turn_ids,
                "importance": event.importance,
            }
        )


def judge_updates(config_path: Path, dataset_dir: Path, limit: int) -> None:
    config = load_config(config_path)
    turns = load_turns(dataset_dir / "dialogues.jsonl")
    extractor = LLMEventExtractor(LLMClient(config.llm))
    events = extractor.extract_all(turns, max_turns=limit)
    updater = LLMConflictAwareUpdater(LLMClient(config.llm))
    memory, relations = updater.apply(events)
    print(f"turns_scanned: {min(limit, len(turns))}")
    print(f"events_extracted: {len(events)}")
    print(f"memory_events: {len(memory)}")
    print(f"relations_judged: {len(relations)}")
    for relation in relations:
        print(
            {
                "new_event_id": relation.new_event_id,
                "old_event_id": relation.old_event_id,
                "relation": relation.relation,
                "reason": relation.reason,
                "evidence_turn_ids": relation.evidence_turn_ids,
            }
        )


def run_experiment_command(
    dataset_dir: Path,
    output_dir: Path,
    method: str,
    limit: int | None,
    top_k: int,
    config_path: Path,
    memory_dir: Path | None,
    llm_turn_scope: str,
) -> None:
    llm = LLMClient(load_config(config_path).llm) if method == "event_memory_llm" and memory_dir is None else None
    result = run_experiment_pipeline(
        dataset_dir=dataset_dir,
        method=method,  # type: ignore[arg-type]
        output_dir=output_dir,
        limit=limit,
        top_k=top_k,
        llm=llm,
        memory_dir=memory_dir,
        llm_turn_scope=llm_turn_scope,  # type: ignore[arg-type]
    )
    print(f"method: {result.method}")
    print(f"total: {result.summary.total}")
    print(f"answer_accuracy: {result.summary.answer_accuracy:.2f}")
    print(f"evidence_recall_at_{top_k}: {result.summary.evidence_recall_at_k:.2f}")
    print(f"abstention_accuracy: {result.summary.abstention_accuracy:.2f}")
    print(f"latency_ms: {result.latency_ms}")
    print(f"output_dir: {output_dir / method}")


def build_llm_memory_command(
    dataset_dir: Path,
    output_dir: Path,
    config_path: Path,
    limit_qa: int,
    force: bool,
) -> None:
    llm = LLMClient(load_config(config_path).llm)
    stats = build_llm_memory(dataset_dir, output_dir, llm, limit_qa=limit_qa, force=force)
    print(f"selected_qa_count: {stats['selected_qa_count']}")
    print(f"selected_user_count: {stats['selected_user_count']}")
    print(f"selected_session_count: {stats['selected_session_count']}")
    print(f"llm_calls_attempted: {stats['llm_calls_attempted']}")
    print(f"llm_calls_failed: {stats['llm_calls_failed']}")
    print(f"session_cache_hits: {stats['session_cache_hits']}")
    print(f"session_cache_misses: {stats['session_cache_misses']}")
    print(f"relation_judge_attempts: {stats['relation_judge_attempts']}")
    print(f"relation_judge_failures: {stats['relation_judge_failures']}")
    print(f"parsed_event_count: {stats['parsed_event_count']}")
    print(f"predicted_relation_count: {stats['predicted_relation_count']}")
    print(f"elapsed_ms: {stats['elapsed_ms']}")
    print(f"output_dir: {output_dir}")


def filter_llm_relations_command(memory_dir: Path, output_dir: Path) -> None:
    report = filter_predicted_memory(memory_dir, output_dir)
    print(f"input_relation_count: {report['input_relation_count']}")
    print(f"kept_relation_count: {report['kept_relation_count']}")
    print(f"removed_relation_count: {report['removed_relation_count']}")
    print(f"output_dir: {output_dir}")


def refresh_cached_memory_text_command(dataset_dir: Path, memory_dir: Path, output_dir: Path) -> None:
    report = refresh_cached_memory_text(dataset_dir, memory_dir, output_dir)
    print(f"event_rows_refreshed: {report['event_rows_refreshed']}")
    print(f"relation_rows_refreshed: {report['relation_rows_refreshed']}")
    print(f"output_dir: {report['output_dir']}")


def build_manual_audit_command(
    dataset_dir: Path,
    memory_dir: Path,
    experiment_dir: Path,
    output_dir: Path,
    relation_limit: int,
    answer_limit: int,
) -> None:
    report = build_manual_audit_package(
        dataset_dir=dataset_dir,
        memory_dir=memory_dir,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        relation_limit=relation_limit,
        answer_limit=answer_limit,
    )
    print(f"relation_audit_count: {report['relation_audit_count']}")
    print(f"answer_audit_count: {report['answer_audit_count']}")
    print(f"output_dir: {output_dir}")


def structure_llm_events_command(memory_dir: Path, output_dir: Path) -> None:
    report = structure_predicted_memory(memory_dir, output_dir)
    print(f"event_count: {report['event_count']}")
    print(f"structured_event_count: {report['structured_event_count']}")
    print(f"structured_event_ratio: {report['structured_event_ratio']}")
    print(f"output_dir: {output_dir}")


def summarize_manual_audit_command(audit_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manual_audit_summary.json"
    summary = summarize_manual_audit(audit_dir, output_path=output_path)
    print(f"relation_labeled_count: {summary['relations']['labeled_count']}")
    print(f"answer_labeled_count: {summary['answers']['labeled_count']}")
    print(f"output_path: {output_path}")


def auto_label_manual_audit_command(audit_dir: Path) -> None:
    report = auto_label_manual_audit(audit_dir)
    print(f"relation_labeled_count: {report['relation_labeled_count']}")
    print(f"answer_labeled_count: {report['answer_labeled_count']}")
    print(f"summary_path: {report['summary_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional Chinese event-memory prototype")
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "run-demo",
            "generate-dataset",
            "generate-grounded-dataset",
            "generate-long-context-dataset",
            "audit-long-context-dataset",
            "generate-public-grounded-dataset",
            "audit-public-grounded-dataset",
            "generate-public-grounded-hard-dataset",
            "audit-public-grounded-hard-dataset",
            "naturalize-dataset",
            "check-api",
            "extract-events",
            "judge-updates",
            "build-llm-memory",
            "filter-llm-relations",
            "refresh-cached-memory-text",
            "build-manual-audit",
            "auto-label-manual-audit",
            "summarize-manual-audit",
            "structure-llm-events",
            "run-experiment",
        ],
    )
    parser.add_argument("--output-dir", default=str(DATA_V0_DIR))
    parser.add_argument("--dataset-dir", default=str(DATA_V0_DIR))
    parser.add_argument("--audit-dir", default=str(ROOT / "outputs" / "manual_audit" / "realistic_llm_memory_sample"))
    parser.add_argument("--config", default="api_config.env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-qa", type=int, default=60)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--relation-limit", type=int, default=100)
    parser.add_argument("--answer-limit", type=int, default=50)
    parser.add_argument("--llm-turn-scope", choices=["user", "evidence"], default="user")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--method",
        choices=[
            "event_memory_gold",
            "event_memory_gold_v2",
            "event_memory_gold_v2_no_supersedes",
            "event_memory_gold_v2_no_time",
            "event_memory_gold_v2_no_entity",
            "event_memory_llm",
            "event_memory_gold_no_supersedes",
            "event_memory_gold_no_time",
            "event_memory_gold_no_entity",
            "chunk_rag",
            "full_context",
            "summary_memory",
        ],
        default="event_memory_gold",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "run-demo":
        run_demo()
    elif args.command == "generate-dataset":
        generate_dataset(Path(args.output_dir))
    elif args.command == "generate-grounded-dataset":
        output_dir = Path(args.output_dir)
        if output_dir == DATA_V0_DIR:
            output_dir = DATA_V1_GROUNDED_DIR
        generate_grounded_dataset(output_dir)
    elif args.command == "generate-long-context-dataset":
        output_dir = Path(args.output_dir)
        if output_dir == DATA_V0_DIR:
            output_dir = DATA_V2_LONG_CONTEXT_DIR
        generate_long_context_dataset(output_dir)
    elif args.command == "audit-long-context-dataset":
        audit_long_context_dataset(Path(args.dataset_dir))
    elif args.command == "generate-public-grounded-dataset":
        output_dir = Path(args.output_dir)
        if output_dir == DATA_V0_DIR:
            output_dir = DATA_PUBLIC_GROUNDED_DIR
        generate_public_grounded_dataset(output_dir)
    elif args.command == "audit-public-grounded-dataset":
        audit_public_grounded_dataset(Path(args.dataset_dir))
    elif args.command == "generate-public-grounded-hard-dataset":
        output_dir = Path(args.output_dir)
        if output_dir == DATA_V0_DIR:
            output_dir = DATA_PUBLIC_GROUNDED_HARD_DIR
        generate_public_grounded_hard_dataset(Path(args.dataset_dir), output_dir)
    elif args.command == "audit-public-grounded-hard-dataset":
        audit_public_grounded_hard_dataset(Path(args.dataset_dir))
    elif args.command == "naturalize-dataset":
        output_dir = Path(args.output_dir)
        if output_dir == DATA_V0_DIR:
            output_dir = DATA_V0_NATURALIZED_DIR
        naturalize_dataset_command(Path(args.dataset_dir), output_dir)
    elif args.command == "check-api":
        check_api(Path(args.config))
    elif args.command == "extract-events":
        extract_events(Path(args.config), Path(args.dataset_dir), args.limit or 3)
    elif args.command == "judge-updates":
        judge_updates(Path(args.config), Path(args.dataset_dir), args.limit or 3)
    elif args.command == "build-llm-memory":
        build_llm_memory_command(
            Path(args.dataset_dir),
            Path(args.output_dir),
            Path(args.config),
            args.limit_qa,
            args.force,
        )
    elif args.command == "filter-llm-relations":
        if args.memory_dir is None:
            raise SystemExit("--memory-dir is required for filter-llm-relations")
        filter_llm_relations_command(Path(args.memory_dir), Path(args.output_dir))
    elif args.command == "refresh-cached-memory-text":
        if args.memory_dir is None:
            raise SystemExit("--memory-dir is required for refresh-cached-memory-text")
        refresh_cached_memory_text_command(Path(args.dataset_dir), Path(args.memory_dir), Path(args.output_dir))
    elif args.command == "build-manual-audit":
        if args.memory_dir is None:
            raise SystemExit("--memory-dir is required for build-manual-audit")
        if args.experiment_dir is None:
            raise SystemExit("--experiment-dir is required for build-manual-audit")
        build_manual_audit_command(
            Path(args.dataset_dir),
            Path(args.memory_dir),
            Path(args.experiment_dir),
            Path(args.output_dir),
            args.relation_limit,
            args.answer_limit,
        )
    elif args.command == "summarize-manual-audit":
        summarize_manual_audit_command(Path(args.audit_dir), Path(args.output_dir))
    elif args.command == "auto-label-manual-audit":
        auto_label_manual_audit_command(Path(args.audit_dir))
    elif args.command == "structure-llm-events":
        if args.memory_dir is None:
            raise SystemExit("--memory-dir is required for structure-llm-events")
        structure_llm_events_command(Path(args.memory_dir), Path(args.output_dir))
    elif args.command == "run-experiment":
        output_dir = Path(args.output_dir)
        if output_dir == DATA_V0_DIR:
            output_dir = EXPERIMENTS_DIR
        memory_dir = Path(args.memory_dir) if args.memory_dir else None
        run_experiment_command(
            Path(args.dataset_dir),
            output_dir,
            args.method,
            args.limit,
            args.top_k,
            Path(args.config),
            memory_dir,
            args.llm_turn_scope,
        )


if __name__ == "__main__":
    main()
