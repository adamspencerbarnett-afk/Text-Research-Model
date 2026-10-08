# Master Plan

## Encoder research track

The repository now carries a second line of work beside V1: Compiler v0, a stateless single-pass text encoder with one frozen dictionary (`compiler/`). Its review and the phased plan for the next experiments (corpus and encoder fixes, the neural learnability ladder, lookup-table models, factorized IDs, the number channel, multi-unit prediction and a learned-chunking benchmark) are in [`docs/ENCODER_RESEARCH_PLAN.md`](docs/ENCODER_RESEARCH_PLAN.md). The V1 plan below is unchanged.

## Current status

V1 Experimental is the active semantic compiler and reasoning baseline. It accepts controlled natural language, predicts typed event programs, and answers through either a 346-parameter learned core or exact execution. The selected lexical attachment parser has 919,045 parameters and a 91-entry vocabulary.

The snapshot is intentionally experimental. Routine programs and answers reached 100% in three observed development screens, while full gates passed two screens and failed one. The reserved final evaluation remains unopened. No continuous-learning implementation exists yet.

## Repository map

- `scripts/semantic_model.py` — surface preprocessing, learned parsers, reasoning cores, exact-execution routing, checkpoint I/O, and the inference CLI.
- `scripts/semantic_tasks.py` — deterministic synthetic accounting and ordering tasks, controlled renderers, source annotations, split fingerprints, and exact execution.
- `scripts/semantic_experiment.py` — training protocols, controls, selection, strict gates, diagnostics, result recording, and checkpoint creation.
- `scripts/test_semantic_v1.py` — 108 runnable semantic regression and research-contract tests.
- `models/semantic_candidate.pt` — current experimental lexical attachment candidate; default inference checkpoint.
- `models/semantic_model.pt` — earlier typed incumbent retained as a frozen reference and core source for research protocols.
- `results/semantic_v1.json` — accumulated semantic experiments and validation evidence.
- `docs/RESEARCH_HISTORY.md` — consolidated V6–V1 chronology with evidence qualifications.
- `docs/CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md` — proposed next architecture; no implementation claim.
- `README.md` and `README.html` — public static and interactive project introductions.

## Supported contract

The current system supports one final question over controlled accounting or ordering text, with one to four visible proper names, no more than 24 clauses, and no more than 160 encoded positions per clause. The event schema contains initial balance, transfer, before relation, balance query, and before query. Transfer rows retain sender, recipient, signed quantity, and active/inactive status.

The parser sees only visible surface information. Gold semantic rows and source annotations are training labels or diagnostics and must never enter text inference. The `executor` path runs the predicted rows through deterministic balance or graph operations. The `parsed` path passes the same predicted rows through the small learned core. A correct answer never substitutes for a correct complete program in release gates.

## Current acceptance standard

- At least 99% complete-program and answer accuracy on IID, names, and retention slices for every declared seed.
- At least 95% on wording, length, numbers, composition, and combined program slices.
- 100% on known regressions and declared counterexample contrasts.
- Selection uses development data only. Confirmation seeds and the reserved final remain unavailable until the fixed development protocol qualifies.
- Report every seed and worst slice. Do not promote a checkpoint from an average that hides a failed program gate.
- Compare architectures from the same raw text and include parser, reasoning, execution, memory, and preprocessing costs as applicable.

The current candidate does not meet the multi-seed qualification standard because seed 433 failed the wording and combined accounting program gates. Keep the candidate frozen as the V1 Experimental reference.

## Next bounded research sequence

Do not start these experiments until the user asks to continue research.

1. Reproduce the fixed early/late encoder-role crossover on observed development data and complete the seed 433 error census.
2. Predeclare one architecture or training-recipe change using that diagnosis; keep fresh confirmation seeds and the reserved final sealed during design.
3. Require complete-program, role, polarity, and all-query gates across all development seeds before confirmation.
4. If the semantic baseline qualifies, implement continuous improvement as a separate versioned service: immutable foreground inference, scoped knowledge memory, verified replay, background candidate training, and gated publication.
5. Evaluate broader language and knowledge with a separately qualified language component; do not reinterpret specialist accuracy as general intelligence.

## Validation commands

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -m unittest discover -s scripts -p "test_semantic_v1.py" -v
.\.venv\Scripts\python.exe -B scripts\semantic_model.py --help
.\.venv\Scripts\python.exe -B scripts\semantic_experiment.py --help
```

Inference smoke test:

```powershell
$prompt = 'Alice starts with 5 coins. Bob starts with 3 coins. Clara starts with 4 coins. David starts with 2 coins. Alice gives Bob 2 coins. how many coins does Bob have now?'
.\.venv\Scripts\python.exe -B scripts\semantic_model.py --text $prompt --mode executor --device cpu
.\.venv\Scripts\python.exe -B scripts\semantic_model.py --text $prompt --mode parsed --device cpu
```

Both answer paths should return `5`. Passing these checks validates the retained software and checkpoint fixtures; it does not rerun training or score the reserved final.

## Research discipline

Keep only this plan and `MASTER_LOG.md` as root research notes. Add significant validated findings to the log and detailed numerical output to the existing semantic results file. Avoid per-run narrative logs, backup copies, and new phase files. Preserve failed hypotheses when they materially narrow the next experiment.

Keep facts, learned behavior, active context, and model capacity as separate resources in the proposed continual architecture. Use retrieval for prompt knowledge before weight updates, and require provenance and rollback for both memory and learned candidates. Device limits must cap storage, training, concurrency, and model versions explicitly.
