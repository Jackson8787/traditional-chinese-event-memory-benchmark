# Package Manifest

## Included

- `README.md` and `README.zh-TW.md`: English and Traditional Chinese project entry points.
- `src/` and `event_memory/`: source code and import shim.
- `tests/`: offline regression tests and consistency checks.
- `data/`: sample, grounded, long-context, public-grounded benchmark, and hard challenge datasets.
- `outputs/public_grounded_benchmark/`: main oracle and baseline result artifacts.
- `outputs/public_grounded_ablation/`: ablation artifacts for the oracle memory design.
- `outputs/public_grounded_hard_challenge/`: hard challenge result artifacts.
- `outputs/public_grounded_hard_ablation/`: hard challenge ablation artifacts.
- `outputs/realistic_llm_memory_sample/`: cached LLM memory sample and evaluation artifacts.
- `outputs/manual_audit/realistic_llm_memory_sample/`: relation and answer audit artifacts.
- `reports/failure_analysis.md` and `reports/failure_analysis_zh.md`: current known failure modes and next-step analysis.

## Excluded

- `api_config.env`: local secret/config file.
- Internal work notes and transfer folders.
- Duplicate proposal folder with the same source files.
- Original proposal DOCX/PDF/PPTX archives; keep them outside the public GitHub package unless they are intentionally meant to be published.
- Python caches, local virtual environments, and editor artifacts.
