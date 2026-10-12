# Archive

Earlier stages of the research, kept whole so every result can be traced. Nothing here is
used by the current code.

## track1_v1_experimental (September 2026)

"Compact Semantic Reasoning": a parser that turned controlled accounting and ordering problems
into typed events, answered by a small learned core or an exact executor. Very accurate inside
its narrow grammar; not a general language model.

- `README_ORIGINAL_SEPTEMBER_2026.md`: the repository's original README.
- `README.html`: the interactive overview of V1.
- `MASTER_PLAN.md`, `MASTER_LOG.md`: the V1 plan, status log and acceptance standard.
- `RESEARCH_HISTORY.md`: the chronology from V6 to V1, including the negative results.
- `CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md`: the design for learning after training
  (memory first, weights later), which the current exchange memory implements.
- `scripts/`: `semantic_model.py`, `semantic_tasks.py`, `semantic_experiment.py` and the
  108-test suite.
- `semantic_candidate.pt`, `semantic_model.pt`: the V1 checkpoints.
- `barnetts-model-v1-experimental.zip`: the V1 release as it was published.

Lessons carried forward: structure makes a small specialist precise; the language boundary is
the main risk; answer accuracy can hide structural failure; fair comparisons start from the
same raw text; exact tools (today's calculator) belong beside the network.

## track2_version1 (frozen 9 October 2026)

`version1.zip`: the deterministic text compiler and the first table-backed model, frozen when
the research moved to the memory-and-copy design. `docs/VERSION_1.md` in the main tree
describes what it contained and what was learned.
