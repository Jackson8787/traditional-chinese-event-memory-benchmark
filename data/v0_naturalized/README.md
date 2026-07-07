# v0 Naturalized Seed Benchmark

Naturalized Traditional Chinese dialogue version of the seed benchmark.

## Source

- Source dataset: `data\v0`
- `dialogues.jsonl` has rewritten user-facing text.
- `personas.jsonl`, `gold_events.jsonl`, `gold_update_relations.jsonl`, and `qa.jsonl` are preserved from the source dataset.
- Turn ids and evidence ids are unchanged, so existing gold labels remain valid.

## Construction

```powershell
$env:PYTHONPATH="src"
C:\Users\o1000\anaconda3\envs\fuckyou\python.exe -m event_memory.cli naturalize-dataset --dataset-dir data\v0 --output-dir data\v0_naturalized
```
