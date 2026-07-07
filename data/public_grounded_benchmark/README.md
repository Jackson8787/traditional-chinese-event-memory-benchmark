# Public-Grounded Long-Context Dataset

This split is a public-safe real-data-grounded synthetic Traditional Chinese
long-context memory benchmark. Public webpages and open data are used only as
short factual grounding for scenario construction. The released dialogues do
not copy long original text and do not include non-commercial sources.

Shape:

- 12 personas.
- 30 sessions per persona.
- 240 memory-bearing user turns per persona.
- 2880 memory-bearing user turns.
- 2880 assistant follow-up turns.
- 360 QA items.
- 120 gold update relations.

Surface variation policy: user memory text avoids the known high-frequency fake
templates checked by the unit tests. The current generated split has 2880 user
turns, 2875 unique user texts, and max exact duplicate count 2.

Evidence policy: QA evidence ids point to memory-bearing user turns only.
Assistant turns provide conversational context and distractors, not gold
evidence.
