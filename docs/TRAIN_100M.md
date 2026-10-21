# Training the 100M model from the C2b recipe

*20 October 2026. For continuing the research on a home computer. This replaces the 17 October plan in
`data_real/100m/plan.json`, which used a real-data recipe that lost at 5M.*

## Why the 100M, and what to expect

C2b shows the design works at 5M: memory, recall at any length, multi-part messages, "I don't know". What holds the 5M
back is understanding new wordings: it scores 92% on skills questions worded like its training and 39-47% on new
wordings; clarifying questions stay near 0% despite 10,410 training examples. A larger network generalizes better from
the same data, so the 100M should lift told personal facts, updates, real human phrasing and document reading. The
memory fronts are already at 85-100% and have little room.

The 100M is the same recipe on a larger core: same training kinds, same shares, same data, same tests. Only the core,
the position scaling and the batch split change.

## What you need

| item | value |
| --- | --- |
| GPU | 12 GB or more (measured peak 11.5 GB with `--accum 2`; 6.9 GB with `--accum 4` for 8-11 GB cards) |
| disk | about 5 GB free (data 2.75 GB, core 0.44 GB, checkpoints, caches) |
| time | about 10-11 hours per seed on an RTX 3090 (0.83 s/step at `--accum 2`, 0.93 s/step at `--accum 4`; 25,000-40,000 steps) |
| software | the same environment as the 5M (`docs/SETUP.md` section 1) |

## 1. Get the data and the core (two bundles)

```bash
git lfs pull --include="datasets/c2b_5m/*,datasets/100m/*"
tar -xJf datasets/c2b_5m/c2b_5m_training_data.tar.xz    # the C2b training data (2.75 GB), from the repo root
tar -xJf datasets/100m/core_P100_109M.tar.xz            # runs/core_P100/scale_custom_v0_8k.pt + its dictionary
```

Check the downloads against `datasets/SHA256SUMS` (`sha256sum -c datasets/SHA256SUMS` from inside `datasets/`).

The core: 109M parameters (1,024 wide, 8 layers, 16 heads, modern block), pre-trained with Muon and a warmup-stable-decay
schedule in bf16 on the P100 mix (Wikipedia, FineWeb-Edu, books, conversations, evidence pairs; 3.2 GB, 2 billion
codes), at 1,024 codes. It uses the same 8,192-entry dictionary as the 5M.

## 2. Train (seed 1)

```bash
SH='{"qa": 0.17, "skills": 0.135, "convo": 0.11, "recall": 0.12, "buried": 0.11, "docqa": 0.06, "story": 0.085, "rehearse": 0.1, "longconv": 0.11}'
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
python -B -m chat22.train --name c2b_100m_s1 --seed 1 \
    --core runs/core_P100/scale_custom_v0_8k.pt --rope-scale 2 --ctx 2048 \
    --data-dir data_real/5m_mixed --mix v22 --parts longconv,longdocs,alpacaqa,multi2 --shares "$SH" \
    --batch 16 --accum 2 --min-steps 25000 --max-steps 40000 --patience 10 \
    --memory-opts "$MO" --gpu-frac 0.75 > runs/chat22_train_c2b_100m_s1.log 2>&1
```

- `--rope-scale 2`: the core was trained at 1,024 codes; the chat window is 2,048.
- `--accum 2`: the 16-window batch runs in two parts with one update (same windows, half the memory). Use `--accum 4` on
  smaller cards.
- Steps: 25,000-40,000 gives about one pass over the real data at 100M (a larger model memorizes faster, so the floor is
  lower than the 5M's 30,000-48,000). Training stops on the held-out plateau.
- Resume: rerunning the same command continues from the last check (state every 500 steps).
- The log prints the resolved data sources and the effective shares at the start; every 500 steps it prints held-out
  loss per kind. An `ALARM` line means a kind's held-out loss rose while its training loss fell (overfitting).

Not yet tested at 100M (watch these): the learning rate (1e-4, the 5M's); the n-gram tables that the trust head mixes in
(at 20M, count tables made a core 1% worse; the chat stage's trust head can learn to ignore them, but check fluency).

## 3. Test and compare with C2b

```bash
EO='{"search": "v2", "hop2": true, "split": true}'
python -B -m chat22.tests.run_suite runs/chat22/c2b_100m_s1.pt --tier full --engine-opts "$EO" --tag _ev2full
python -B -m chat22.tests.run_suite runs/chat22/c2b_100m_s1.pt --tier full --engine-opts "$EO" \
    --sections afterno,phrasing,probe2,reason,twomem,open,multi --tag _ev2new
python research_side/compare_two_seeds.py c2b_100m_s1 --best combo2b_5m_s1
```

The full suite takes longer at 100M (about 1-1.5 hours). Chat with it: `python run_model.py --chat runs/chat22/c2b_100m_s1.pt`
(answers are slower than the 5M: a few milliseconds per code).

If it is stronger, train seed 2 (`--seed 2`, name `c2b_100m_s2`) before building on it, and record the result in
`recipes/BEST.json` and `LOG.md`.

## 4. After the first 100M

- The open 5M candidates apply at 100M too: counting and recaps (`--parts ...,reasoncr`), a training form for "told fact +
  stored fact" that does not teach answering from stored facts over the conversation (see the design page).
- More data variety for a larger model: more any-length sessions (`chat22.data.build_v23 --only longconv --facts
  data_real/shared/look/facts_pool_mixed.jsonl ...`, `docs/SETUP.md` section 3), more real phrasings.
- A 16k window as a short final extension stage (positions scaled again, long documents); long reach today comes from
  the conversation memory.
- Reinforcement learning on checked answers was planned for the 100M only (the 5M has no room for it).
