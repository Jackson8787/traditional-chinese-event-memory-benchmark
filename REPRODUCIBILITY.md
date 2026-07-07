# Reproducibility

## Environment

The project targets Python 3.11 or newer. The core tests use only the standard library and local project code.

```bash
python3 -m venv .venv
source .venv/bin/activate
PYTHONPATH=src python3 -m unittest discover tests
```

Optional packages for live LLM and retrieval experiments are listed in `requirements-optional.txt`.

## Dataset Integrity Checks

```bash
PYTHONPATH=src python3 -m event_memory.cli audit-public-grounded-dataset --dataset-dir data/public_grounded_benchmark
PYTHONPATH=src python3 -m event_memory.cli audit-public-grounded-hard-dataset --dataset-dir data/public_grounded_hard_challenge
```

## Recreate Public-Grounded Dataset Files

```bash
PYTHONPATH=src python3 -m event_memory.cli generate-public-grounded-dataset --output-dir data/public_grounded_benchmark
PYTHONPATH=src python3 -m event_memory.cli generate-public-grounded-hard-dataset --dataset-dir data/public_grounded_benchmark --output-dir data/public_grounded_hard_challenge
```

## Re-run Offline Oracle Experiments

```bash
PYTHONPATH=src python3 -m event_memory.cli run-experiment --method event_memory_gold_v2 --dataset-dir data/public_grounded_benchmark --output-dir outputs/public_grounded_benchmark
PYTHONPATH=src python3 -m event_memory.cli run-experiment --method event_memory_gold_v2 --dataset-dir data/public_grounded_hard_challenge --output-dir outputs/public_grounded_hard_challenge
```

## Inspect Cached Realistic Sample

The cached realistic sample is already under `outputs/realistic_llm_memory_sample/`. It was generated as a user-scope realistic sample for LLM-built memory and evaluated with evidence-scoped QA selection. Treat it as a diagnostic sample, not full 360-question main result. Use `event_memory_gold_v2` as the oracle upper-bound when comparing this sample to offline baselines.

## Live API Runs

Live API commands are optional and should only be run after creating a local `api_config.env` from `api_config.example.env`.

```bash
PYTHONPATH=src python3 -m event_memory.cli build-llm-memory --dataset-dir data/public_grounded_benchmark --output-dir outputs/realistic_llm_memory_sample/memory --limit-qa 60 --config api_config.env
PYTHONPATH=src python3 -m event_memory.cli run-experiment --method event_memory_llm --dataset-dir data/public_grounded_benchmark --memory-dir outputs/realistic_llm_memory_sample/memory --output-dir outputs/realistic_llm_memory_sample --llm-turn-scope evidence
```

Do not commit `api_config.env`.
