# Master Log

## Active issues

- V1 Experimental is a controlled-language specialist, not a general language model. Its 919,045-parameter lexical attachment parser plus 346-parameter learned core do not provide broad knowledge, conversation, or free-form generation.
- The preservation recipe repaired all known routine cases across seeds 419, 383, and 433, but seed 433 failed strict accounting program gates: 76% on unfamiliar wording and 38% on the combined shift. Correct base answers hid some inactive-transfer role errors.
- Only seeds 419 and 383 passed all observed development gates. The same development worlds were reused across seeds. Confirmation seeds 449, 461, and 479 remain unused, and the reserved final remains unscored.
- Continuous improvement, persistent memory, concurrent candidate training, and long-context retrieval remain design work only. The next architecture is documented in `docs/CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md`.

## Current verified baseline

- Public semantic closure: `semantic_model.py`, `semantic_tasks.py`, `semantic_experiment.py`, and `test_semantic_v1.py`; dependency closure is Python standard library plus PyTorch.
- Candidate checkpoint: lexical attachment parser, width 192, two local and two reasoning layers, 91 vocabulary entries, 919,045 parser parameters, and 346 learned-core parameters.
- Capacity: four entities, at most 24 clauses, at most 160 encoded positions per clause, and one scalar integer channel per clause.
- Preservation recipe: 1,500 parser updates; freeze after update 300; 909,072 protected parser parameters; 9,973 role-related trainable parameters; no optimizer reset or learning-rate-schedule change.
- Routine development result: 400 worlds per seed, 200 per task, with 100% complete programs and both answers for all three observed seeds.
- Release status: experimental research snapshot. No scientific promotion, fresh confirmation fit, final scoring, or continuous-learning update.

## Release cleanup — 2026-09-15

- Consolidated the complete architectural chronology and significant negative results into `docs/RESEARCH_HISTORY.md`.
- Reduced the release to the active semantic implementation, two required checkpoints, one semantic result record, two root notes, and the continuous-improvement proposal.
- Replaced the mixed test suite with a semantic-only suite containing 103 explicit test methods and five inherited contract tests, for 108 runnable tests total.
- Updated the experiment result path and test-source routing to the semantic release names.
- Set the inference CLI default to `models/semantic_candidate.pt`; the earlier incumbent remains available explicitly and is still used as a fixed research reference.
- Added a self-contained interactive HTML overview, a Markdown README, dependency metadata, repository text/binary rules, and automated GitHub test configuration.
- Release preparation changed documentation, routing, and checkpoint metadata only. It did not train weights, evaluate new data, or access the reserved final.

## Accumulated research findings

### Explicit state and specialist reasoning

- The early signed-delta arithmetic model reported 95.50% on its historical split with 88,430 parameters. Checkpoint selection reused the reported test split, so this is development evidence.
- Deterministic entity canonicalization preserved a reported 95.24% across new names/items; cross-item decomposition reported 94.62%. These gains came from normalization and routing around a frozen core.
- Handled snapshot, cancellation, between-time, and conditional operations reported 99.34–99.64% through compiler orchestration.
- A task-matched 19,351-parameter message-passing relation model reported 100% on its graph benchmark after a generic compact Transformer failed multi-hop propagation. The architecture and representation changed together, so this was not a representation-only control.
- Frozen arithmetic and graph specialists reported 94.74% on composed cross-domain queries. Held-out composed programs later reported 98.02%.

### Scaling and context limits

- Dynamic graphs reported 97.68% and event references 93.64%, while recursive large-world streaming fell to 19.3%, 14.3%, and 13.6% query accuracy at 8, 16, and 32 entities.
- A fixed-parameter recurrent workflow model accepted 640 events, but balanced accuracy declined to roughly 42–47%. Executable input length is not evidence of retained long-range reasoning.
- Width increases were not monotonic. New-policy adapters plateaued near 47–49%, explanation prediction near 25%, and planning remained weak. These failures narrowed the next work toward information preservation and verified interfaces.

### Compiler and parsing boundary

- A learned byte-level parser historically reached 36.52% exact parse and 51.28% answer accuracy. A deterministic parser reached 100% parse and 99.13% answers within its supported controlled grammar.
- The large jump established the value of correct decomposition; it did not demonstrate unrestricted text understanding. Once a compiler predicts the wrong operation or role, the downstream reasoner cannot recover reliably.
- Some historical compact/control comparisons supplied structured tuples to one path and raw language to the other. Those results cannot establish end-to-end text superiority.
- The current learned semantic parser restores a fair surface-text boundary. Oracle rows are labels and diagnostics only. Complete programs, contrasts, and all-query results remain primary because answers can mask structural errors.

### Workflow controls

- The historical compact workflow model slightly exceeded its subword control on several synthetic slices, while the control won one risk-prior shift. Alias results were confounded by unequal input processing.
- In the stronger long-context comparison, compact reported 63.85% balanced accuracy versus 62.02% for the control. The compact model had more parameters, and raw accuracy was close to the majority-class baseline; no universal advantage follows.
- Small missing-field and prior-shift adapters supported bounded adaptation without changing the core, motivating the future fixed-capacity learning design.

### Semantic V1 architecture

- Early direct text-to-answer models were near 31%, while typed/oracle paths reached 100% on IID tasks. This localized the main bottleneck to interpretation and role binding.
- Byte, typed, clause, pointer, relative-position, constrained-role, local-role, joint-role, evidence, and attachment variants were tested. Improvements that did not survive independent seeds or complete-program gates were not promoted.
- Preserving the encoder and kind/activity heads after update 300 fixed routine regressions in all three screens. Head-only continuation did not repair seed 433’s representation at the freeze boundary.
- Exact execution is the strongest answer path inside the supported schema, but its accuracy is bounded by parser fidelity. The 346-parameter learned core remains useful as a separately measured neural control.

## Decisions that remain in force

- Treat all V6–V12 numbers as historical reported results unless a future run reconstructs and reproduces them.
- Do not open the reserved final until one fixed recipe qualifies on every declared development and confirmation gate.
- Do not claim 100% intelligence from 100% bounded regression accuracy.
- Do not attribute causality to compactness when architecture, preprocessing, supervision, or input information differs.
- Do not update serving weights in place. A future continuous system must train a separate candidate and publish only a version that passes learning, retention, resource, and rollback gates.
- Keep long-term searchable knowledge separate from active neural context. Measure retrieval completeness and structured scans before pursuing extremely large local attention windows.
