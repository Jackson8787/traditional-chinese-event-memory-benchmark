# Failure Analysis

This report summarizes the main weaknesses visible in the packaged artifacts. It is intentionally conservative: the current project should be read as a research prototype and diagnostic benchmark package, not as a mature benchmark or production-ready memory system.

## Hard Challenge

The hard challenge has 72 questions:

| Subset | Total | Answer | Evidence@3 | Update | Temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard_conflict | 48 | 0.979 | 0.979 | 0.979 | 0.000 |
| hard_temporal | 24 | 0.000 | 0.000 | 0.000 | 0.000 |
| overall | 72 | 0.653 | 0.653 | 0.653 | 0.000 |

The overall score hides an important asymmetry. The oracle upper-bound handles most explicit supersedes-expansion questions, but fails every time-relevance question in the hard split: `hard_temporal` is 0/24.

The error records show a repeated pattern: for time-scoped questions such as "only before March 2026, what was the old state and what did it later change into?", the retrieval answer often selects a different update chain, such as future activity or reminder updates, instead of the requested location/tool update chain. All 25 hard-split error rows are tagged as `update_error`; 24 of those are `hard_temporal`, matching the 0/24 subset result.

Interpretation: the current event/update representation can link many direct supersedes cases, but it does not yet robustly constrain update-chain retrieval by time scope and requested state dimension.

## Realistic LLM Sample

The realistic LLM memory result is a 60-question diagnostic sample, not the full 360-question benchmark:

| Metric | Value |
| --- | ---: |
| Answer | 0.817 |
| Evidence@3 | 0.759 |
| All Evidence@3 | 0.481 |
| Update | 0.800 |
| Faithful | 0.433 |
| Strict state | 0.360 |

Answer accuracy is therefore not enough to claim a robust memory system. The low faithful and strict-state scores mean the system can often produce plausible answers while still failing to preserve a complete and clean internal state.

## Audit Limitations

The included manual-audit package is auto-labeled. It is useful for finding likely failure modes, but it is not independent human validation.

Current audit summary:

- Relation rows labeled: 100.
- `supersedes` precision: 0.333.
- `supplements` precision: 0.541.
- Answer rows labeled: 50.
- Evidence complete rate: 0.440.
- Evidence complete-or-partial rate: 0.780.

These numbers should be treated as triage evidence. Strong claims require human review, preferably with at least two annotators and disagreement resolution.

## Dataset Scope

The dataset is synthetic and public-grounded. This makes it safer to release and easier to reproduce, but it does not prove that the same failure distribution will appear in real private long-term user histories.

The current scale is useful for a project benchmark:

- 12 personas.
- 30 sessions per persona.
- 360 main QA items.
- 72 hard-challenge QA items.
- 120 gold update relations.

It is not large enough to claim a general-purpose benchmark without additional datasets, external replication, and human annotation.

## Recommended Next Work

- Break down hard temporal failures by state dimension, retrieved chain, and missed evidence ids.
- Add a retrieval constraint that jointly uses update-chain relation, valid time, and target state dimension.
- Expand the realistic LLM run from 60 to the full 360 QA items only when API budget is available.
- Human-review the audit rows, especially low-confidence `supersedes`, temporal reasoning, and evidence completeness cases.
- Add external baselines or invite third-party reruns before presenting the benchmark as broadly validated.
