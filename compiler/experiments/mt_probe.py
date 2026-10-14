"""Multi-turn probe: does the model use the conversation? (14 October 2026; docs/MULTITURN_PLAN.md)

    python compiler/experiments/mt_probe.py MODEL.pt --device cuda --graph [--precision bf16] [--heldout data_mt]

1. Scripted conversations, five groups, keyword-graded on the last answer: recall of a stated fact,
   reference resolution, follow-ups, corrections, returning to an earlier topic. The earlier turns are played
   live (the model's own replies go into the history, as in a real chat), and the final question is also asked
   with no history, so the score shows what the conversation itself contributed.
2. History gain on held-out conversations (prepare_dialogue.py): bits per byte of each thread's final reply with
   the earlier turns in the window and without them; the share of bits the history saves.
Writes results/mtprobe_<model><tag>.json and docs/chat_probes/mt_<model><tag>.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import m2_qa  # noqa: E402

# (group, earlier user turns, final question, keyword groups: any of each, all groups)
CONVERSATIONS = [
    ("recall", ["My sister is called Ana and she lives in Lisbon."], "Where does my sister live?", [["lisbon"]]),
    ("recall", ["I have a dog named Max. He is three years old."], "How old is my dog?", [["three", "3"]]),
    ("recall", ["My favourite colour is green."], "What is my favourite colour?", [["green"]]),
    ("recall", ["I work as a nurse at a hospital in Chicago."], "What is my job?", [["nurse"]]),
    ("recall", ["My car is a red Toyota."], "What brand is my car?", [["toyota"]]),
    ("recall", ["I was born in 1990 in a small town."], "What year was I born?", [["1990"]]),
    ("reference", ["Here are three cities: Paris, Tokyo and Cairo."], "Which country is the second one in?", [["japan"]]),
    ("reference", ["My children are Tom, Sara and Leo."], "What is the name of the last child I mentioned?", [["leo"]]),
    ("reference", ["I bought apples, bread and milk today."], "What was the second thing I bought?", [["bread"]]),
    ("reference", ["The meeting is on Monday at 3 pm in room 12."], "What room is it in?", [["12"]]),
    ("reference", ["Think about the planets Mars and Venus."], "Which of the two is closer to the Sun?", [["venus"]]),
    ("followup", ["What is the capital of France?"], "What river runs through it?", [["seine"]]),
    ("followup", ["Who wrote Romeo and Juliet?"], "Name another play he wrote.", [["hamlet", "macbeth", "othello", "lear", "tempest"]]),
    ("followup", ["What is photosynthesis?"], "What gas do plants release during it?", [["oxygen"]]),
    ("followup", ["Tell me about the Eiffel Tower."], "In which city is it?", [["paris"]]),
    ("correction", ["My meeting is on Tuesday.", "Sorry, I meant Thursday, not Tuesday."], "What day is my meeting?", [["thursday"]]),
    ("correction", ["I live in Boston.", "Actually, I moved to Denver last week."], "Where do I live now?", [["denver"]]),
    ("correction", ["Please call me Sam.", "On second thought, call me Alex instead."], "What should you call me?", [["alex"]]),
    ("correction", ["My order number is 4417.", "Correction: the order number is 4471."], "What is my order number?", [["4471"]]),
    ("return", ["My cat is called Luna.", "What is the capital of Italy?"], "What is my cat called?", [["luna"]]),
    ("return", ["I am planning a trip to Spain in May.", "What is a good book to read on a plane?"], "Which country am I travelling to?", [["spain"]]),
    ("return", ["My brother plays the guitar.", "Name a famous painter."], "What instrument does my brother play?", [["guitar"]]),
    # one paragraph-style message with several parts; correct only if every part is answered (14 October)
    ("multipart", [], "I need two things. First, what is the capital of Canada? Second, what is the boiling point of water in Celsius?",
     [["ottawa"], ["100"]]),
    ("multipart", [], "Here is a note from my courier: 'Order 5521 will be delivered on Friday at noon.' When will it arrive, and what is the order number?",
     [["friday"], ["5521"]]),
    ("multipart", [], "I have three meetings tomorrow: budget at 9, design at 11 and hiring at 2. Which meeting is at 11, and what time is the hiring meeting?",
     [["design"], ["2"]]),
    ("multipart", [], "My name is Priya and I am learning to cook. I have chicken, rice and spinach at home. What could I make with them, and what is my name?",
     [["chicken", "rice", "spinach"], ["priya"]]),
    ("multipart", ["I am planning a party for 12 people next Saturday. It will be in my garden, and two of my guests are vegetarian."],
     "How many guests are vegetarian, and where is the party?", [["two", "2"], ["garden"]]),
]


def check(answer: str, groups) -> bool:
    low = answer.lower()
    return all(any(k in low for k in g) for g in groups)


@torch.no_grad()
def history_gain(eng, dialogues, n: int = 300) -> dict:
    """Bits per byte of the final reply with and without the earlier turns (same window layout as training)."""
    a, enc, model, dev = eng.a, eng.enc, eng.model, eng.dev
    sep = np.asarray(eng.sep, dtype=np.int64)
    bits = {"with": 0.0, "without": 0.0}; nbytes = 0.0; used = 0
    for t in dialogues[:n]:
        k = len(t) - 1 if t[-1][0] == "assistant" else len(t) - 2
        if k < 3:
            continue
        q, ans = t[k - 1][1].encode("utf-8")[-350:], t[k][1].encode("utf-8")
        own = np.append(enc.encode(m2_qa.render(q, ans))[0], m2_qa.EOS)[: a.ctx // 2]
        ql = min(len(enc.encode(m2_qa.render(q, None))[0]), len(own) - 1)
        hist = b"".join(m2_qa.render(t[j][1].encode("utf-8"), t[j + 1][1].encode("utf-8")) for j in range(0, k - 1, 2))
        hc = enc.encode(hist)[0][-min(getattr(a, "history_codes", 400), a.ctx + 1 - len(sep) - len(own)):]
        scored = own[ql:]
        nbytes += float(enc.lens[scored].sum()); used += 1
        for mode, h in (("with", hc), ("without", np.array([], dtype=np.int64))):
            seq = np.concatenate([h, sep, own])
            x = torch.from_numpy(seq[None, :-1]).to(dev); y = torch.from_numpy(seq[None, 1:]).to(dev)
            src = torch.zeros_like(x, dtype=torch.bool); src[0, :len(h)] = True
            nll = model.nll(x, y, src, torch.zeros(1, device=dev))[0, len(h) + len(sep) + ql - 1:]
            bits[mode] += float(nll.sum()) / math.log(2)
    w, wo = bits["with"] / max(1, nbytes), bits["without"] / max(1, nbytes)
    return {"threads": used, "bpb_with_history": round(w, 4), "bpb_without_history": round(wo, 4),
            "history_gain_pct": round(100 * (1 - w / max(1e-9, wo)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--graph", action="store_true"); ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "int8", "int4"])
    ap.add_argument("--max-codes", type=int, default=80); ap.add_argument("--no-repeat", type=int, default=3)
    ap.add_argument("--heldout", default=str(ROOT / "data_mt"), help="prepare_dialogue.py folder for the history gain ('' to skip)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    native = ROOT / "data" / "cv0.exe"
    eng = m2_qa.QAEngine(a.model, native=str(native) if native.exists() else None, device=a.device, precision=a.precision)
    fast = "graph" if a.graph else True
    rows, t0 = [], time.perf_counter()
    for group, earlier, final, keys in CONVERSATIONS:
        history = []
        for u in earlier:   # played live: the model's own replies go into the history
            r, _ = eng.ask(u.encode(), a.max_codes, a.no_repeat, fast=fast, history=list(history) or None)
            history.append((u.encode(), r))
        ans, _ = eng.ask(final.encode(), a.max_codes, a.no_repeat, fast=fast, history=history)
        alone, _ = eng.ask(final.encode(), a.max_codes, a.no_repeat, fast=fast)
        text, text_alone = ans.decode("utf-8", "replace"), alone.decode("utf-8", "replace")
        rows.append({"group": group, "turns": earlier, "question": final, "answer": text, "answer_without_history": text_alone,
                     "ok": check(text, keys), "ok_without_history": check(text_alone, keys),
                     "history": [[q.decode("utf-8", "replace"), r.decode("utf-8", "replace")] for q, r in history]})
    summary = {}
    for r in rows:
        g = summary.setdefault(r["group"], {"n": 0, "ok": 0, "ok_without_history": 0})
        g["n"] += 1; g["ok"] += r["ok"]; g["ok_without_history"] += r["ok_without_history"]
    total = {"n": len(rows), "ok": sum(r["ok"] for r in rows), "ok_without_history": sum(r["ok_without_history"] for r in rows)}
    hg = None
    if a.heldout and Path(a.heldout, "heldout", "dialogues.jsonl").exists():
        dl = [json.loads(l)["turns"] for l in open(Path(a.heldout, "heldout", "dialogues.jsonl"), encoding="utf-8")]
        hg = history_gain(eng, dl)
    stem = Path(a.model).stem + a.tag
    res = {"model": a.model, "precision": a.precision, "graph": a.graph, "summary": summary, "total": total,
           "history_gain": hg, "seconds": round(time.perf_counter() - t0, 1), "rows": rows}
    (ROOT / "results" / f"mtprobe_{stem}.json").write_text(json.dumps(res, indent=1))
    md = [f"# Multi-turn probe: `{a.model}`\n\n*{time.strftime('%Y-%m-%d %H:%M')}. Earlier turns played live; the final "
          "question asked with the conversation and without it. Keyword check on the final answer.*\n\n",
          "| group | with history | without |\n| --- | ---: | ---: |\n"]
    md += [f"| {k} | {v['ok']}/{v['n']} | {v['ok_without_history']}/{v['n']} |\n" for k, v in summary.items()]
    md.append(f"| **all** | **{total['ok']}/{total['n']}** | {total['ok_without_history']}/{total['n']} |\n")
    if hg:
        md.append(f"\nHistory gain on {hg['threads']} held-out conversations: {hg['history_gain_pct']}% "
                  f"({hg['bpb_without_history']} -> {hg['bpb_with_history']} bits per byte of the final reply).\n")
    for r in rows:
        md.append(f"\n**{r['group']}**: " + " / ".join(f"*{h[0]}* -> {h[1][:120]}" for h in r["history"])
                  + f"\n\n> **{r['question']}** {r['answer'][:300] or '(empty)'} ({'yes' if r['ok'] else 'no'}; "
                  f"without history: {'yes' if r['ok_without_history'] else 'no'})\n")
    out = ROOT / "docs" / "chat_probes" / f"mt_{stem}.md"
    out.write_bytes("".join(md).encode("utf-8"))
    print(json.dumps({"total": total, "summary": summary, "history_gain": hg}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
