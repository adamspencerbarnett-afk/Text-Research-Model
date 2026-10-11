"""The run ledger: every Q&A-design run and every scale run, settings beside results, in one
Markdown file, regenerated from the JSON results so it is never out of date.

    python compiler/experiments/run_ledger.py            # writes docs/RUN_LEDGER.md

The queue of planned runs and the findings per session are the two lists at the top; edit
them here. Everything else is read from results/.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- planned runs
QUEUE = [
    ("R1b", "Alpaca, 4.7M core adapted from the scale-run checkpoint (--init, --dict), window 256, 600 s GPU", "does a core that knows language raise the no-memory floor and the copy scores?", "done: yes, best run so far"),
    ("R1c", "same model, greedy decoding, evaluation only (--budget 0 --greedy)", "exact-answer rate without sampling drift", "done: +5-10 points exact"),
    ("R3b", "GSM8K, 4.7M core trained from scratch, copy from question + executor + check, 1200 s GPU", "set-up quality against depth (6 layers)", "done: 0/198 unseen"),
    ("R3c", "GSM8K, 4.7M core adapted from the scale-run checkpoint", "set-up quality with a core that knows language", "done: 1/198 unseen; stored 43-45%"),
    ("R1b-s2", "R1b with seed 2, greedy", "confirm the winner", "done: confirmed (seen 0.88 / 52% exact)"),
    ("R1d", "Alpaca, 19M core (scale run checkpoint) adapted, greedy", "does the floor keep rising with core size once the core knows language?", "done: better prediction, weaker copying (stored 8% exact vs ~50%)"),
    ("R0b", "Alpaca, 0.8M: retrieval similarity as a trust-head input; three window kinds in training", "learn when to copy; stop the prediction score worsening", "queued"),
    ("R0c", "Alpaca, 0.8M: top-3 retrieved exchanges in the window", "paraphrase group above the single-retrieval ceiling (0.27)", "queued"),
    ("R2b", "SQuAD with loss on answers and replies cut at the span", "copy-from-passage", "done: spans now short, but wrong (unseen F1 0.07)"),
    ("D1-D3r", "5M stage-1 data tests: clean modern text, + conversation, + evidence/math/rewordings, aligned vs random windows", "which data makes the core best at its three jobs", "done: big core gains, not visible after stage 2"),
    ("S2a", "stage 2 from the D3r core at lr 3e-4 and 1e-4, with and without 30% stage-1 rehearsal", "does a gentler stage 2 keep the core's gains?", "next"),
    ("K1", "contrastive question key: a projection head trained on Quora reworded pairs", "a learned memory key that beats the literal one", "next"),
    ("E1", "evidence builder: relaxed thresholds, multiple use, longer windows", "enough evidence pairs for the 50M mix (got 20.6k, needed 110k)", "next"),
    ("R2", "SQuAD 1.1: copy a span from a given passage, exact match", "the cleanest test of the copy head", "done (failed: rambling, memorised passages)"),
    ("R4", "GSM8K/SVAMP: retrieval by solution structure (operator sequence key), template copying", "reasoning by analogy over worked examples", "next after R2 (design written)"),
    ("R5", "number channel in the compiler (number unit → code + value)", "removes digit-run failures", "needs compiler change"),
    ("R6", "ARC / OpenBookQA / CommonsenseQA, one-letter answers with retrieved facts", "accuracy vs chance (25%)", "needs --format"),
    ("R7", "bAbI deduction / counting / path tasks", "V1's ground; expect the design to shine", "needs download mirror"),
    ("R8", "full check loop: verify, refine, restart, 'unknown'", "precision of 'unknown'", "after R4"),
    ("R9", "19M core on the winning design", "decides the 'smaller at equal quality' claim for reasoning", "after R4"),
    ("T1", "scale: domain feature for the trust head; per-domain tables (REASONING_PLAN 2.4)", "can the tables be made harmless at 19M?", "queued, low priority"),
]

# ----------------------------------------------------------------------------- findings per session
FINDINGS = [
    ("9 Oct, session 1 (GPU set-up)", [
        "CPU and GPU reproduce the same numbers to four decimals; bf16 reproduces them too; the GPU is 2x faster at 20M+ with the ragged tables.",
        "Windows CRLF in the corpus scripts changed every book's hash; fixed, corpora rebuilt from the manifests.",
    ]),
    ("9 Oct, session 2 (copy head)", [
        "The learned copy head is the missing mechanism: network answers on stored questions 0.13 -> 0.48 / 0.62 (two seeds); no change where memory holds nothing, so the gain is entirely from carrying what memory returns.",
        "An evidence bigram over the retrieved span and key folding add nothing; an answer-only copy span is no better than the full exchange.",
        "A target leak appeared when the copy span included the current position (pointer could point at itself); fixed by restricting the pointer to strictly earlier positions. Earlier copy-head results are unaffected.",
    ]),
    ("9 Oct, session 3 (scale run)", [
        "The exact-table gain reverses with core size: +10% at 0.8M, 0 at 4.7M, -1% at 19M (equal time), and 9-12% worse on Wikipedia. The tables are a crutch for a small core; the trust head does not reject them off-domain.",
        "The tables stay as the store that learns by reading; they are not the mechanism for 'lighter and faster at equal quality'. The memory and the copy head are.",
    ]),
    ("10 Oct, session 7 (what to train the core on)", [
        "Data deep dive: 82% of stage-1 text was 1800s fiction (14.4 archaic words per 10k, 0.2 modern), the rest raw wiki markup (half the bytes), no conversation, no evidence windows. Manual review found more: conversations 5x longer than the 256-code window, FineWeb paragraph breaks invisible, MetaMath in LaTeX, Gutenberg front matter. All rewritten (prepare_core_mix.py).",
        "Better data transforms the core itself (5M, equal steps, D0 -> best new): conversation fluency 2.02 -> 1.30 bits/byte (UltraChat) and 2.12 -> 1.49 (OpenAssistant), Wikipedia 2.06 -> 1.68, web 2.02 -> 1.66, math 2.23 -> 0.91; evidence gain 22.7% -> 31-32%; reworded-question retrieval by the core's own state 0.38 -> 0.50 at rank 1, 0.59 -> 0.78 in the top 10. Conversation data gave the largest single jump. Old books got worse (expected: D0 trained 82% on them).",
        "None of it shows in the Alpaca stage-2 scores: every core lands at stored 0.62-0.69 F1, reworded about 0.2, no-memory 0.09-0.14, within seed noise. Cause: stage 2 is a full fine-tune at lr 1e-3 for 15,565 steps on 19 MB, which overwrites stage-1 differences, and Alpaca F1 mostly measures copying. Stage 2 must be gentler (lower lr, rehearsal of stage-1 data) to keep what stage 1 built.",
        "Paragraph-aligned windows: mixed. Random windows give slightly better fluency (Wikipedia 1.68 vs 1.73, math 0.91 vs 1.17); aligned slightly better evidence gain (32.4% vs 31.2%) and reword retrieval (0.50 vs 0.49). Use random windows for prose and aligned windows only for structured units. The evidence test was incomplete: the builder found 20.6k pairs, so evidence was 3% of D3 instead of 18%.",
        "Learned memory key: the core's state alone finds 50% of reworded questions at rank 1 against 74% for the literal key; combined, 75%. A useful learned key needs training on rewordings (contrastive), not the language model's states as they are.",
        "S2a-1 (Q&A stage at lr 3e-4 instead of 1e-3, D3r core): validation 1.039 -> 0.971 and the no-memory floor doubles (0.10 -> 0.20 F1, equal to the 19M core), but exact copying drops (stored 30% -> 10%): the fresh copy and trust heads learn too slowly at the core's gentle rate. Fix adopted: two learning rates (--head-lr), core gentle, new heads full.",
        "S2a-2 (lr 3e-4 + 30% rehearsal of stage-1 text): best validation yet (0.925) and no-memory floor 0.22, but exact copying near zero (stored 3%, added 0%). The transcript shows why: answers do not stop ('William Shakespeare' then ': William Shockpeare : William Wordsworth ...'). Nothing in stage-2 training ever marked the end of an answer (the window after it is padding, outside the loss), and rehearsal text teaches that more text always follows. Fix adopted: an end-of-answer code trained as the last code of every exchange (--eos), the industry's end-of-turn token; generation stops at it.",
        "S2a-3 (lr 1e-4 + rehearsal): the best balance of the series. Validation 0.896 (from 1.039 at 1e-3), no-memory floor 0.23 (from 0.10), and copying back near the old level (stored 0.65 F1 / 27% exact, added 0.63 / 20%); probe 22/32 with held-out instructions 4/5. The 3e-4 + rehearsal run (3% exact) is the outlier, not the trend; the stage-2 recipe moves to 1e-4 for the core with rehearsal, plus the head rate, answer-only loss and the end code (S2b).",
        "19M with equal steps (R1f): no better than 4.7M on Q&A, and copies less (stored exact 8% against 22%). Reading comprehension (R2b): the copy head now returns short spans from the passage, but the wrong ones (unseen F1 0.07). Both point at the same limit: the core does not yet read well enough to choose what to copy.",
    ]),
    ("10 Oct, session 6 (power cut, seed check, reasoning with the adapted core)", [
        "Power cut lost R3c during scoring; the Q&A script now saves weights before scoring and run chains skip finished runs.",
        "R1b confirmed on a second seed: seen 0.79 / 0.88 F1, 47% / 52% exact; added 0.75 / 0.81, 53% / 48%; no-memory floor 0.20 / 0.15. The adapted 4.7M core is the base.",
        "GSM8K with the adapted core (R3c): 1 of 198 unseen. Three cores (0.8M scratch, 4.7M scratch, 4.7M adapted) all score 0-2%: core size and language knowledge do not fix arithmetic set-up. The replies are now well-formed step-by-step solutions whose individual steps are correct arithmetic (the executor computes them: 157 stated answers confirmed, 628 replaced) but whose choice of operations is wrong. This is the retrieval-of-procedure problem (R4), not a capacity problem.",
        "R1d (19M core adapted): best prediction yet (0.958 bits/byte against 0.98-1.05 at 4.7M) and a slightly higher no-memory floor (0.22 F1 against 0.15-0.20), but copying from memory is much weaker: stored questions 0.76 F1 / 8% exact against 0.79-0.88 / 47-52% at 4.7M. Cause: equal time is not equal training. The 19M core is 2.3x slower per step, so the same 600 s gave it 6,694 steps against 15,565, and the fresh copy head and trust head got less than half the updates. One seed; the fair re-run gives 19M the same number of steps (about 1,400 s).",
        "A small executor defect: after the executor writes a value the model can append digits to it ('17 - 17 = 0.5'); fix by closing the number after the executor writes it.",
    ]),
    ("9 Oct, session 5 (adapted core)", [
        "Adapting the scale run's 4.7M core (trained on 286 MB of books + Wikipedia) to the Q&A design for ten minutes beats every earlier run in every group: seen 0.80 F1 / 42% exact, added 0.74 / 43%, paraphrase 0.29, unseen 0.27, no-memory floor 0.19 (from 0.13). A core that knows language is the base from now on; training a core from scratch on Q&A data is dropped.",
        "Greedy decoding raises exact match by 5-10 points at no F1 cost (seen 47%, added 53%); adopted for scored answers. The 3-nat retrieval bias now lowers exact match (22% vs 42%): superseded by the copy head, dropped.",
        "GSM8K with a 4.7M core from scratch: 0 of 198 unseen; the executor only helps where the problem is stored (42-50%). The adapted core on GSM8K (R3c) is the next test.",
    ]),
    ("9 Oct, session 4 (reasoning baseline)", [
        "GSM8K unseen: 1% right at 0.8M. With copy-from-question, the executor and the answer check: 2% (inside noise). The machinery works (operands come from the question, results are computed); the 0.8M core cannot choose the operations. Set-up, not computing, is the problem.",
        "A bigger window (512, whole exchange) did not help the copy head (truncation hypothesis dropped).",
        "A 4.7M core trained from scratch on 19 MB of Q&A memorises it in minutes (validation flat from minute one, no-memory floor unchanged at 0.13). The core must learn language from the big text and be adapted to Q&A; plateau stopping with best-weights restore added to m2_qa.",
    ]),
]


def num(v, d=3):
    return "" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def core_of(setup: dict) -> tuple[str, int]:
    w, l, h = setup.get("width", 128), setup.get("layers", 4), setup.get("heads", 4)
    params = l * 12 * w * w
    return f"{w}w {l}L {h}h", params


def flags_of(s: dict) -> str:
    on = []
    for k, label in (("copy_head", "copy"), ("evidence", "evid"), ("fold_keys", "fold"), ("copy_answer_only", "ans-only"),
                     ("copy_question", "copy-q"), ("calculator", "calc"), ("greedy", "greedy")):
        if s.get(k):
            on.append(label)
    if s.get("init"):
        on.append("init:" + Path(s["init"]).stem)
    return " ".join(on) or "none"


def qa_rows() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results" / "m2_*.json")) + glob.glob(str(ROOT / "results" / "r[0-9]*.json")),
                       key=os.path.getmtime):
        j = json.load(open(path))
        if "groups" not in j:
            continue
        s, g, t = j["setup"], j["groups"], j["training"]
        core, params = core_of(s)
        data = "GSM8K" if "gsm" in s.get("data", "") else "Alpaca"
        seen, added = g["seen_network_with_own_exchange_in_context"], g["added_network_with_added_exchange_in_context"]
        para, unseen, nomem = g["paraphrase_network_with_nearest"], g["unseen_network_with_nearest"], g["heldout_network_without_memory"]
        rows.append({
            "run": Path(path).stem.replace("m2_qa_", "").replace("m2_", ""), "data": data, "core": core, "params": f"{params/1e6:.2f}M",
            "ctx": s.get("ctx"), "span": s.get("neighbor_codes"), "batch": s.get("batch"), "budget": int(s.get("budget", 0)),
            "steps": t["steps"], "device": s.get("device", "cpu"), "seed": s.get("seed"), "self": s.get("self_context_p"),
            "flags": flags_of(s), "valid": t.get("best_valid") or (t["history"][-1]["valid_bits_per_byte"] if t["history"] else None),
            "seen_f1": seen["f1"], "seen_ex": seen.get("exact"), "seen_bias": seen.get("f1_bias"),
            "added_f1": added["f1"], "para_f1": para["f1"], "para_n": para["n"], "unseen_f1": unseen["f1"], "nomem_f1": nomem["f1"],
            "num_unseen": unseen.get("num_exact"), "num_seen": seen.get("num_exact"), "stop": t.get("stop_reason", "budget"),
        })
    return rows


def scale_rows() -> list[dict]:
    out = []
    for path in [ROOT / "results" / "scale_run.json", ROOT / "results" / "scale_check.json"]:
        if not path.exists():
            continue
        j = json.load(open(path))
        runs = j.get("runs") or j.get("configs") or {}
        for key, r in runs.items():
            if "model" not in r:
                continue
            m, t, ev = r["model"], r["training"], r["eval"]
            wiki = ev.get("ood", {}).get("enwik_head_2MiB", {}).get("bits_per_byte")
            out.append({"file": path.stem, "run": key, "core": f"{m['width']}w {m['layers']}L {m['heads']}h", "params": f"{m['params']['total']/1e6:.1f}M",
                        "tables": "exact 2,3-gram x%d" % m["ngram_tables"][0]["topk"] if m.get("ngram_tables") else "none",
                        "ctx": m["ctx"], "batch": m["batch"], "budget": int(t["budget_s"]), "steps": t["steps"], "seen_gb": t["bytes_seen"] / 1e9,
                        "amp": t.get("amp", False), "books": ev["heldout_overall"]["bits_per_byte"], "valid": ev.get("validation", {}).get("bits_per_byte"),
                        "wiki": wiki, "ood": ev.get("ood_overall", {}).get("bits_per_byte"), "gen": r["generation"]["bytes_per_s"], "stop": t.get("stop_reason", "budget")})
    return out


def table(headers: list[tuple[str, str]], rows: list[dict]) -> str:
    head = "| " + " | ".join(h for _, h in headers) + " |\n|" + "|".join(" ---: " if k not in ("run", "data", "core", "flags", "device", "stop", "tables", "file") else " --- " for k, _ in headers) + "|\n"
    body = "".join("| " + " | ".join(num(r.get(k)) for k, _ in headers) + " |\n" for r in rows)
    return head + body


def main() -> int:
    qa = qa_rows()
    sc = scale_rows()
    md = ["# Run ledger\n", "*Regenerated by `compiler/experiments/run_ledger.py` from `results/`. Settings on the left, results on the right. "
          "F1 is word overlap with the reference answer (0 to 1); `ex` is the exact-match rate; `num` is the final-number exact-match rate on GSM8K; "
          "`valid` is bits per byte on held-out exchanges with the nearest exchange in the window (lower is better). Seed noise on the F1 columns is about ±0.07 (60 sampled answers).*\n",
          "\n## Q&A design runs (M2 memory + M1 tables + core + trust head, with the flags listed)\n",
          table([("run", "run"), ("data", "data"), ("core", "core"), ("params", "core params"), ("ctx", "ctx"), ("span", "span"), ("batch", "batch"),
                 ("budget", "budget s"), ("steps", "steps"), ("device", "dev"), ("seed", "seed"), ("self", "self-ctx p"), ("flags", "flags"),
                 ("valid", "valid b/B"), ("seen_f1", "seen F1"), ("seen_ex", "seen ex"), ("seen_bias", "seen F1 +bias"), ("added_f1", "added F1"),
                 ("para_f1", "para F1"), ("para_n", "para n"), ("unseen_f1", "unseen F1"), ("nomem_f1", "no-mem F1"), ("num_seen", "num seen"), ("num_unseen", "num unseen"), ("stop", "stop")], qa),
          "\nFlags: `copy` learned copy head over the retrieved exchange; `evid` bigram table over the retrieved span; `fold` inflection folding in the key; "
          "`ans-only` copy span limited to the answer; `copy-q` copy head may point into the current question; `calc` executor writes `a op b =` results and checks the final answer; "
          "`greedy` argmax decoding; `init:` core initialised from that checkpoint. Runs 2 and 3 on GSM8K carry the target leak (fixed in 2b/3b).\n",
          "\n## Scale runs (language modelling, books + Wikipedia; traditional BPE against compiler + exact tables)\n",
          table([("file", "file"), ("run", "run"), ("core", "core"), ("params", "params"), ("tables", "tables"), ("ctx", "ctx"), ("batch", "batch"), ("budget", "budget s"),
                 ("steps", "steps"), ("seen_gb", "text seen GB"), ("amp", "bf16"), ("books", "books b/B"), ("valid", "valid b/B"), ("wiki", "wiki b/B"), ("ood", "ood b/B"), ("gen", "reply B/s"), ("stop", "stop")], sc),
          "\n## Queue of planned runs\n", "| id | run | question it answers | status |\n| --- | --- | --- | --- |\n",
          "".join(f"| {i} | {r} | {q} | {st} |\n" for i, r, q, st in QUEUE),
          "\n## Key findings by session\n"]
    for title, items in FINDINGS:
        md.append(f"\n**{title}**\n\n" + "".join(f"- {it}\n" for it in items))
    out = ROOT / "docs" / "RUN_LEDGER.md"
    out.write_bytes("".join(md).encode("utf-8"))
    print(f"wrote {out} with {len(qa)} Q&A runs and {len(sc)} scale runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
