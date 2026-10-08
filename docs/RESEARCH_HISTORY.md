# Research History: From Compact State Models to a Semantic Compiler

*Research concept and direction by Adam Barnett · September 15, 2026.*

This document consolidates the project’s experimental history into one public record. It preserves the findings that shaped the current architecture while removing superseded plans, duplicate logs, obsolete scripts, and unrelated branches.

The evidence has two levels:

- **Current reproducible evidence** comes from the retained semantic source, checkpoints, tests, and [`results/semantic_v1.json`](../results/semantic_v1.json). The workspace cleanup reran 108 semantic tests and exercised both retained answer paths.
- **Historical reported evidence** covers V6 through V12. Those values were copied from the original phase records before cleanup. They were reviewed for consistency but were not reproduced during this release preparation. Several old experiments lack the complete checkpoints, prepared datasets, or modules required for a faithful rerun.

That distinction matters. A recorded result explains why the architecture changed; it is not automatically a current benchmark or a scientifically independent replication.

## Original question

The research began with a practical local-compute goal: represent text compactly, train on less redundant input, and preserve relationships that ordinary token prediction might learn inefficiently. The hope was that a representation closer to the underlying meaning could produce a smaller, faster, and more accurate model.

The experiments gradually separated three ideas that had initially been treated as one:

1. **Storage compression** reduces bytes on disk or over a connection.
2. **Input representation** determines what distinctions a model can see and how efficiently it can learn them.
3. **Semantic compilation** converts language into explicit entities, roles, state changes, relations, and queries.

High precision consistently followed the third idea. Compactness helped some workloads, but correct decomposition and routing explained the strongest gains. This redirected the project from “compressed text should make a model intelligent” toward a testable pipeline: interpret visible text, produce a typed program, verify that program, and reason over it.

## V6: compact signed-state arithmetic

The first strong specialist used a signed-delta representation for short accounting stories. Transfers became changes to entity balances, which were processed by an 88,430-parameter model over 19 compact positions. The historical record reports 95.50% answer accuracy.

That result showed that a small model could solve a narrow task with high precision when the input already exposed the right state operations. It did not demonstrate open-language understanding. The experiment also selected checkpoints using its reported test split, so 95.50% should be treated as a development-era result rather than an untouched final estimate.

The useful architectural lesson was durable: arithmetic stories become easier after identity, direction, amount, and operation type are represented explicitly.

## Phase 7: canonicalization and decomposition

Phase 7 tested whether the arithmetic core could generalize beyond the exact names and items seen during training.

An attempt to retrain with explicit slots failed, reporting 27.64% answer accuracy and 13.83% state accuracy. The next approach kept the successful frozen core and moved generalization into a deterministic front end. It renamed visible entities into canonical local identities, selected the relevant item, and compiled transfers into signed deltas. This retained a reported 95.24% on 5,000 examples with new names and items.

Cross-item composition then decomposed a problem, ran the existing specialist on the needed parts, and combined its outputs. It reported 94.62% over 5,000 examples without retraining the arithmetic core.

This was the first clear compiler result. The model did not learn arbitrary language invariance by itself; the front end normalized irrelevant surface variation and preserved the variables the core needed. That was valuable, but it shifted the reliability requirement to the compiler.

## Phases 8–10: orchestration and specialist composition

The project next asked whether compiler transformations could add operations around a fixed core. Historical Phase 8 records report:

| Operation | Reported accuracy |
| --- | ---: |
| Snapshot queries | 99.58% |
| Cancellation | 99.56% |
| Between-time queries | 99.34% |
| Conditional execution | 99.64% |

These operations were largely achieved by transforming a request into calls the existing state model could handle, then composing the returned states. The result strengthened the case for orchestration, but it also meant that accuracy depended on handwritten task knowledge.

Phase 9 moved to relations. A generic compact Transformer reported 65.57% entity-distance accuracy and 18.56% exact-world accuracy on multi-hop ordering. A 19,351-parameter message-passing model then reported 100% entity-distance, reachability, and exact-world accuracy on its graph benchmark. A vectorized version was approximately 2.79 times faster in the recorded comparison.

The architectures were not matched controls. Message passing encoded the graph operation the task required, so this result supports an architectural fit between representation and reasoning. It does not isolate compact representation as the sole cause.

Phase 10 combined the frozen arithmetic and relation specialists. The 107,781-parameter system reported 94.74% overall accuracy on 5,000 cross-domain queries. It showed that a router could compose high-performing specialists while keeping their internal models small.

## Phases 11–12: dynamic graphs, references, and programs

Phase 11 increased the amount of state and control needed by each problem:

- Dynamic state-dependent graphs reported 97.68% accuracy.
- Event-reference tasks reported 93.64%.
- A recursive large-world streaming attempt failed, with query accuracy falling to 19.3%, 14.3%, and 13.6% at 8, 16, and 32 entities.

The streaming failure was important. Fixed-size recurrent state allowed the code to accept larger worlds, but the state did not preserve the distinctions required to answer them. Executable length was therefore separated from reasoning quality.

Phase 12 treated questions as composed programs over specialist operations. Held-out composed query programs reported 98.02% across 5,000 examples without retraining the specialists. This was a strong result for symbolic composition inside the supported grammar, while still depending on correct program construction.

## Phases 13–14: virtual machine and the parsing pivot

Phase 13 built a more general query virtual machine. Its historical sequence reported:

| Variant | Reported accuracy |
| --- | ---: |
| Raw generic VM | 91.30% |
| Invariant candidate handling | 93.07% |
| Four-permutation verification | 93.55% |
| Adaptive variant | 93.43% |

The more consequential comparison concerned language parsing. A learned byte-level convolutional parser reached only 36.52% exact parse and 51.28% end-answer accuracy. A deterministic controlled-language parser reached 100% exact parse and 99.13% answer accuracy on 6,000 supported paraphrases.

The deterministic result explained much of the earlier “excellent accuracy.” When language was known and compiled correctly, the downstream reasoning machinery worked very well. The parser did not generalize to unrestricted language; it precisely covered a designed grammar. The failed learned parser showed that text-to-structure learning was the unresolved problem.

Phase 14 shifted to hidden-rule workflow prediction, where labels could not be solved by an obvious exact arithmetic executor. A typed compact model with 72,835 parameters and 21 positions reported 72.89% balanced accuracy and approximately 0.25 ms model latency. A class-aware subword control with 78,083 parameters and 193 positions reported 71.55% balanced accuracy and approximately 0.666 ms. The comparison suggested an efficiency advantage for the compact representation on that synthetic workflow, but did not establish broad language or reasoning superiority.

## Phases 15–17: robustness, long context, and stronger controls

Phase 15 tested distribution shifts in the workflow task. The compact model reported 74.84% balanced accuracy on IID data versus 66.83% for its subword control. Under a risk-prior shift, the control slightly led, 67.71% to 66.95%. The compact model’s reported alias result was 75.80% versus 33.33%, but the inputs were not equivalent: the compact path received already-canonicalized event tuples while the control received rendered alias text. That comparison cannot establish raw-text alias understanding.

Phase 16 introduced a hierarchical model for 60-event inputs. Its 73,219-parameter version reported 81.40% raw accuracy and 61.88% balanced accuracy. A fair long-context control was still missing.

Phase 17 supplied a stronger shared test. The recorded results were:

| Model | Parameters | Raw accuracy | Balanced accuracy | Macro F1 | Model-only latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact hierarchy | about 73K | 83.83% | 63.85% | 0.611 | 0.647 ms |
| Stronger subword control | about 62K | 79.63% | 62.02% | 0.569 | 2.152 ms |

The compact model had roughly 17.3% more parameters. The task was heavily imbalanced, with an 83.67% majority-class raw baseline, so balanced accuracy and macro F1 are more informative than the raw headline. The latency measurement excluded preprocessing and represented model execution only.

Small frozen-core adapters were also tested. A 320-scalar missing-field adapter improved its selected condition, and a four-parameter prior calibrator addressed one distribution shift. These results supported bounded adaptation, while the records explicitly avoided claiming a universal ceiling or a general architecture win.

## Phases 18–30: multi-incident routing and fixed-capacity memory

The final historical branch explored workflow triage with target incidents mixed among distractors. It added query-relative routing, structured latent supervision, dual target/background streams, abstention, recurrent memory, adapters, explanations, and planning.

The structured multi-incident model reported 66.82% raw and 61.39% balanced accuracy. A 144,292-parameter dual-stream model reported 65.33% raw and 61.25% balanced accuracy on its recorded test, compared with 56.54% for a streaming baseline. A separate structured model achieved 95% answered-case accuracy at about 24% coverage; that selective result cannot be attributed to the dual-stream model without another evaluation.

A fixed 72,382-parameter recurrent model could execute 160-, 320-, and 640-event inputs. Its balanced accuracy declined from 57.26% at 160 events to roughly 50–52% at 320 and 42–47% at 640. Increased width was not consistently better.

Later experiments produced useful negative evidence:

- New-policy adapters plateaued around 47–49% balanced accuracy.
- Explanation prediction remained around 25%.
- The learned planner achieved about 3.1% exact match, versus 4.8% for an oracle-information condition, with regret about 2.82.
- Cascade and permutation variants did not resolve the main errors.

These failures narrowed the goal. A fixed-size state can process a long sequence without retaining every relevant detail. More routing, more width, or a longer accepted input does not guarantee better decisions.

## Semantic V1: text to typed events

The current branch returned to the core unresolved issue: can a learned model reliably bind visible language to an explicit program?

The retained system has four layers of responsibility:

1. Surface preprocessing identifies sentence boundaries, integer literals, and first-mention entity identities without consulting gold events.
2. A learned parser predicts event kind, two entity roles, activity/polarity, and associates visible quantities with clauses.
3. A 346-parameter learned core answers from predicted rows, providing a neural reasoning measurement.
4. An exact executor answers from the same predicted rows, isolating parser quality from reasoning quality.

Early raw text-to-answer paths were near 31% and showed shortcut learning. Structured inputs could reach 100% on IID tasks, confirming that the arithmetic and relation operations were solvable once represented correctly. Generalization then failed across unfamiliar wording, longer relation chains, and certain compositions. The architecture moved from byte and typed-position encoders toward a learned lexical parser with explicit relative positions, constrained roles, local and joint role variants, evidence diagnostics, and finally source attachment.

The selected lexical attachment parser has 919,045 parameters and a 91-entry vocabulary. Together with the learned core, the parsed neural path uses 919,391 parameters. It supports at most four entities, 24 clauses, and 160 encoded positions per clause.

The final preservation experiment froze 909,072 parser parameters after 300 updates and continued training 9,973 role-related parameters. All three observed seeds reached 100% on the routine regression set: 400 development worlds per seed, divided equally between accounting and relation tasks, with complete programs and both answer paths checked.

The full gate result was mixed:

| Seed | Routine regression | Full observed development gates |
| ---: | ---: | ---: |
| 419 | 100% | Pass |
| 383 | 100% | Pass |
| 433 | 100% | Fail |

Seed 433 reached 76% accounting program accuracy on the wording shift and 38% on the combined accounting shift. Its base answers could still be correct while inactive-transfer roles were wrong. This is why V1 remains experimental. The same development worlds were reused across seeds, three planned confirmation seeds remain unused, and the reserved final was not scored.

The selected candidate is a useful reproducible snapshot, not a scientifically promoted model. It captures the best current architecture, the strict gates, and the exact failure that should guide the next experiment.

## What the compiler taught us

Across the entire history, “compiler” referred to several forms of transformation: deterministic canonicalization, handwritten decomposition, program construction, and now learned semantic parsing. Their common value was an explicit interface between language and reasoning.

Five conclusions survived repeated testing:

1. **Structure can make a small specialist extremely precise.** Entity identity, semantic roles, polarity, state updates, and graph edges reduce the problem the reasoner must learn.
2. **The language boundary becomes the main risk.** A perfect executor cannot repair a wrong sender, recipient, event type, or negation decision.
3. **Answer accuracy can hide structural failure.** Complete programs, counterfactual contrasts, and all-query checks are necessary.
4. **Capacity and context length are not monotonic quality controls.** Wider models and longer executable streams sometimes performed worse.
5. **Fair controls must begin with the same raw information.** Results are confounded when one path receives structured events and another must infer them from text.

The current input is still numerical at the neural boundary. Lexical tokens become integer IDs and floating-point embeddings, and values use a numerical side channel. What remains “natural” is the semantic relationship represented by the typed program, not the physical datatype used by the hardware.

## What this work does and does not establish

The research supports a compact specialist architecture for controlled accounting and ordering language. It shows that semantic decomposition can yield high scoped precision and that a learned parser can be evaluated independently from a reasoner.

It does not establish that this method alone can create a competitive general-purpose language model. General conversation requires much broader language coverage, knowledge, generation, safety behavior, and transfer across domains. Those capabilities require suitable training data, capacity, and end-to-end evaluation. A small specialist and a large general model solve different problems even when the specialist wins on its own exact contract.

The historical accuracy values also should not be combined into one progress curve. The tasks, inputs, labels, architectures, and metrics changed between phases. Their value is causal and diagnostic: each experiment removed one mistaken assumption or exposed a new bottleneck.

## Current direction: bounded continuous improvement

The next architecture is a proposal, not an implemented result. It separates three update speeds:

- New facts enter a scoped, provenance-aware memory and become retrievable without changing weights.
- Repeated verified interpretation failures become bounded replay data for background candidate training.
- Model size or backbone changes occur only as explicit migrations with a new qualification cycle.

Serving uses immutable model and knowledge revisions, so an active request never sees partially updated state. A background candidate must pass new-skill, old-skill retention, resource, concurrency, and rollback tests before publication. Smaller devices can freeze parameter growth and cap stored knowledge, consolidating or evicting within policy rather than growing without limit.

Long-context work will initially target reliable access to a corpus larger than the active context through retrieval and structured scans. A million-token searchable collection is distinct from a million-token dense-attention window. The full design and proposed experiments are in [`CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md`](CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md).

## Release evidence and reproducibility

The GitHub-ready workspace intentionally retains only the current semantic implementation, its two checkpoints, one accumulated semantic results file, the consolidated history, and the continuous-improvement proposal. Historical metrics above are clearly labeled as reported because the old scripts and assets were removed after their material findings were merged here.

For the current branch, run:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s scripts -p "test_semantic_v1.py" -v
```

The suite contains 108 runnable tests: 103 explicitly defined methods plus five inherited model-contract tests. It validates code behavior and fixed research contracts; it does not rerun all training or open the reserved final set.

Future claims should continue the same discipline: declare the input contract, preserve disjoint evaluation data, compare end to end, report complete programs alongside answers, include resource cost, and state when a number is historical, developmental, or independently confirmed.
