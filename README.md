# Traditional Chinese Event-Centric Long-Term Memory Benchmark

[Traditional Chinese version](README.zh-TW.md)

This repository contains a research prototype for Traditional Chinese long-term conversational memory. It focuses on event-centric retrieval, conflict-aware memory updates, and evidence-grounded question answering across long multi-session dialogues.

The dataset is public-safe and synthetic, with factual scenario grounding from public sources. It does not contain private chat logs or API keys.

## Current Status

The project is ready as a research prototype and benchmark package. The offline test suite, public-grounded dataset, hard challenge split, cached realistic sample, and manual audit summaries are included so the main claims can be checked without calling an external API.

The `event_memory_gold_v2` method should be read as an oracle upper-bound, not as a deployed model. The cached `event_memory_llm` run is a user-scope realistic sample and diagnostic sample, generated with `--llm-turn-scope evidence`; it is not the full 360-question main result. This distinction matters for baseline fairness.

## Included Data And Results

| Area | Path | Role |
| --- | --- | --- |
| Main benchmark | `data/public_grounded_benchmark/` | 12 personas, 30 sessions each, 360 QA items, 120 update relations |
| Hard challenge | `data/public_grounded_hard_challenge/` | 72 harder QA items for update-chain, temporal, and distractor stress tests |
| Main results | `outputs/public_grounded_benchmark/` | Oracle and baseline metrics |
| Hard challenge results | `outputs/public_grounded_hard_challenge/` | Hard-split oracle metrics |
| Realistic LLM sample | `outputs/realistic_llm_memory_sample/` | Cached 60-question LLM-memory diagnostic sample |
| Manual audit | `outputs/manual_audit/realistic_llm_memory_sample/` | Auto-labeled audit package for relation and answer quality |

## Key Metrics

| Evaluation | Answer | Evidence@3 | Update | Temporal | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Main oracle upper-bound | 0.964 | 0.957 | 0.997 | 1.000 | Full public-grounded benchmark |
| Hard challenge oracle | 0.653 | 0.653 | 0.653 | 0.000 | Challenge split only |
| Realistic LLM memory sample | 0.817 | 0.759 | 0.800 | 1.000 | 60-question diagnostic sample |

For the realistic sample, additional quality indicators are `faithful = 0.433` and `strict_state = 0.360`. These are the main remaining weaknesses.

## Important Failure Modes

These numbers should not be read as a mature benchmark or production memory system.

- Hard challenge performance is uneven: `hard_conflict` is strong at 47/48 correct, but `hard_temporal` is 0/24. The overall hard-challenge score is therefore hiding a complete failure on the time-relevance subset.
- The cached LLM result covers 60 questions, not the full 360-question main benchmark. It is useful as a diagnostic sample, not as a statistically strong final model result.
- The realistic sample has low state-quality scores: `faithful = 0.433` and `strict_state = 0.360`. Answer accuracy alone overstates the quality of the memory state.
- The manual audit package is auto-labeled. It is useful for triage and error localization, but it is not independent human validation.
- The data is synthetic and public-grounded. It should not be treated as a verified distribution of real private long-term chat histories.

See `reports/failure_analysis.md` for the current error analysis and recommended next steps.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
PYTHONPATH=src python3 -m unittest discover tests
```

No API key is needed for the tests or for reading the cached results. If you want to run live LLM extraction, copy `api_config.example.env` to `api_config.env` and fill in local credentials. Do not commit `api_config.env`.

## Main Offline Checks

```bash
PYTHONPATH=src python3 -m event_memory.cli audit-public-grounded-dataset --dataset-dir data/public_grounded_benchmark
PYTHONPATH=src python3 -m event_memory.cli audit-public-grounded-hard-dataset --dataset-dir data/public_grounded_hard_challenge
PYTHONPATH=src python3 -m event_memory.cli run-experiment --method event_memory_gold_v2 --dataset-dir data/public_grounded_benchmark --output-dir outputs/public_grounded_benchmark
```

More details are in `REPRODUCIBILITY.md`.
