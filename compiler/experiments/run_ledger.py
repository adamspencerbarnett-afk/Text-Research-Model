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
    ("R1b", "Alpaca, 4.7M core adapted from the scale-run checkpoint (--init, --dict), window 256, 600 s GPU", "does a core that knows language raise the no-memory floor and the copy scores?", "running"),
    ("R1c", "same model, greedy decoding, evaluation only (--budget 0 --greedy)", "exact-answer rate without sampling drift", "queued"),
    ("R3b", "GSM8K, 4.7M core trained from scratch, copy from question + executor + check, 1200 s GPU", "set-up quality against depth (6 layers)", "running"),
    ("R3c", "GSM8K, 4.7M core adapted from the scale-run checkpoint", "set-up quality with a core that knows language", "queued"),
    ("R0b", "Alpaca, 0.8M: retrieval similarity as a trust-head input; three window kinds in training", "learn when to copy; stop the prediction score worsening", "queued"),
    ("R0c", "Alpaca, 0.8M: top-3 retrieved exchanges in the window", "paraphrase group above the single-retrieval ceiling (0.27)", "queued"),
    ("R2", "SQuAD 1.1: copy a span from a given passage, exact match", "the cleanest test of the copy head", "needs --format squad"),
    ("R4", "GSM8K/SVAMP: retrieval by solution structure (operator sequence key), template copying", "reasoning by analogy over worked examples", "needs the second index"),
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
    for path in sorted(glob.glob(str(ROOT / "results" / "m2_*.json")) + glob.glob(str(ROOT / "results" / "r1*.json")),
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
            "flags": flags_of(s), "valid": t["history"][-1]["valid_bits_per_byte"] if t["history"] else None,
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
