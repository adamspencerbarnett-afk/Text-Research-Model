# Datasets (Git LFS, fetched on request)

A normal `git clone` skips this folder (`.lfsconfig`: `fetchexclude = datasets/**`), so trying the model costs nobody
gigabytes. Fetch a bundle when you want to train, then unpack it **from the repository root**: every file lands at the
path the trainer reads.

| bundle | size | unpacked | what it is for |
| --- | ---: | ---: | --- |
| `c2b_5m/c2b_5m_training_data.tar.xz` | 590 MB | 2.75 GB | everything the C2b recipe trains on (5M and 100M) |
| `100m/core_P100_109M.tar.xz` | 405 MB | 436 MB | the 109M pre-trained core and its dictionary, for the 100M |

```bash
git lfs pull --include="datasets/c2b_5m/*"                 # or "datasets/c2b_5m/*,datasets/100m/*" for the 100M
tar -xJf datasets/c2b_5m/c2b_5m_training_data.tar.xz
tar -xJf datasets/100m/core_P100_109M.tar.xz              # 100M only
cd datasets && sha256sum -c SHA256SUMS                     # check the downloads
```

On Windows, Git Bash's `tar` unpacks `.tar.xz`; 7-Zip works too (unpack twice: `.xz`, then `.tar`).

## What the C2b bundle holds

| path | size | used for |
| --- | ---: | --- |
| `data_qa/train/qa_train.txt`, `data_qa/heldout/qa_heldout.txt` | 19 MB | the Q&A kind (Alpaca instruction Q&A) and its held-out windows; the open-ended test |
| `runs/chat5m_v2/dense_top4.npy` | 1.6 MB | MiniLM neighbours of the Q&A questions (look-alike facts in Q&A windows) |
| `data_real/shared/skills/train_v4c.jsonl` | 44 MB | personal facts told in conversation (skills v4c; the last 2,000 held out) |
| `data_real/shared/convo/dialogues_train.jsonl` | 1.2 GB | real dialogues (UltraChat, OASST), de-duplicated; filler turns for recall and multi-part windows |
| `data_chat22/rehearse_P100_whole_554MB_seed1.txt` | 555 MB | plain-text rehearsal (whole documents from the P100 core mix) |
| `data_core/mix_P100/train/books.txt` | 224 MB | long documents for story continuation and buried evidence |
| `data_real/shared/look/facts_pool_real.jsonl` | 61 MB | real look-alike facts for the search in training |
| `data_real/shared/look/facts_pool_mixed.jsonl` | 65 MB | real facts + 20,001 generated personal facts: the pool behind the any-length sessions (to build more) |
| `data_real/5m_mixed/longconv/longconv_s1_[0-3].jsonl` | 593 MB | any-length sessions (5,000 sessions, 67,099 question windows) |

Already in the repository (not in the bundle): the permanent store `data_real/shared/knowledge/knowledge_real.jsonl`
(311,010 real facts, LFS), the 5M core `runs/core_mt5m/` (LFS), and the held-out test files (`data_chat22/eval_clean/`,
`data_core/eval_clean/heldout/`, `data_mt/heldout/`, `data_skills/`).

## Sources and licences

Wikipedia (CC BY-SA), FineWeb-Edu (ODC-BY), Project Gutenberg books (public domain), UltraChat (MIT), OpenAssistant
(Apache 2.0), Alpaca (CC BY-NC 4.0: research use), Dolly (CC BY-SA 3.0), SQuAD (CC BY-SA 4.0), ARC (CC BY-SA 4.0),
OpenBookQA (Apache 2.0), CommonsenseQA (MIT), Natural Questions-style real Q&A in the store (CC BY-SA). Generated
material (personal facts, story worlds, sessions) is our own. Rebuilding everything from the sources: `docs/SETUP.md`
section 5 and `data_real/README.md`.
