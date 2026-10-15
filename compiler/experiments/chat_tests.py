"""Tests for the 5M chat model v2 (15 October 2026): speed and limits of the long window, remembering a whole
conversation, short stories, and the conversation probes played as real chats.

    python compiler/experiments/chat_tests.py runs/chat5m_v2/chat5m_v2.pt --device cuda [--cpu-threads 16] [--tag x]

  1 speed     first-code time (the wait before streaming starts) and ms per code as the window fills, 256 -> 2,000 codes,
              on the GPU and on the CPU
  2 position  bits per code of held-out text by position, out to twice the trained window (where long context breaks)
  3 recall    a fact told in the first turn, asked after 2-40 real filler turns, with the session memory on and off
              (names and phrasings reserved for tests, never trained)
  4 stories   eight short-story prompts, up to 600 codes, streamed; length, repetition, whether the named character is
              still there in the second half, speed; texts saved for reading
  5 probes    mt_probe (27) and mt_probe2 (120) conversations played live through the chat engine
Writes results/chat_tests_<model><tag>.json and docs/chat_probes/stories_<model><tag>.md.
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
import chat_engine as CE  # noqa: E402
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import m2_qa  # noqa: E402
import mt_probe  # noqa: E402
import mt_probe2  # noqa: E402

STORIES = ["Write a short story about a lighthouse keeper named Ada who finds a message in a bottle.",
           "Write a short story about a boy named Theo who builds a robot out of old bicycle parts.",
           "Tell me a story about a dragon named Ember who is afraid of the dark.",
           "Write a short story about two friends, Mira and Jonas, who get lost in a forest.",
           "Write a story about a baker named Lucia who bakes a cake that grants wishes.",
           "Write a short story about an old cat named Biscuit who travels across the city.",
           "Tell me a short story about a girl named Nora who learns to sail.",
           "Write a story about a detective named Hale who solves a mystery at a train station."]
NAMES = ["Ada", "Theo", "Ember", "Mira", "Lucia", "Biscuit", "Nora", "Hale"]


def heldout_pairs():
    out = []
    for l in open(ROOT / "data_mt" / "heldout" / "dialogues.jsonl", encoding="utf-8"):
        t = json.loads(l)["turns"]
        out += [(t[j][1].encode("utf-8", "replace")[:600], t[j + 1][1].encode("utf-8", "replace")[:700]) for j in range(0, len(t) - 1, 2)
                if t[j][0] == "user" and t[j + 1][0] == "assistant"]
    return out


def speed(eng, pairs, lengths, label):
    rows = []
    for L in lengths:
        hist, n = [], 0
        for p in pairs:                      # fill the window with real turns to about L codes
            c = len(eng.enc.encode(CE.render(*p))[0])
            if n + c > L:
                break
            hist.append(p); n += c
        eng.reply(b"Thanks. Can you summarize what we talked about?", 8, history=hist, use_memory=False)   # warm-up
        _, info = eng.reply(b"Thanks. Can you summarize what we talked about?", 64, history=hist, use_memory=False)
        per = (info["seconds"] - info["first_code_ms"] / 1000) / max(1, info["codes"] - 1)
        rows.append({"device": label, "window_codes": info["window_used"] - info["codes"], "first_code_ms": info["first_code_ms"],
                     "ms_per_code": round(1000 * per, 2), "codes": info["codes"]})
        print("speed", rows[-1], flush=True)
    return rows


@torch.no_grad()
def position(eng, n_windows=12, length=4096):
    enc, model, dev = eng.enc, eng.model, eng.dev
    text = b"".join(p.read_bytes() for p in sorted((ROOT / "data_core" / "eval_clean" / "heldout").glob("*.txt")))
    codes = enc.encode(text[:3_000_000])[0]
    edges = [0, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    sums, cnt = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    starts = np.linspace(0, len(codes) - length - 2, n_windows).astype(int)
    for s0 in starts:
        seq = torch.from_numpy(codes[s0:s0 + length + 1].astype(np.int64))[None].to(dev)
        x, y = seq[:, :-1], seq[:, 1:]
        nll = model.nll(x, y, torch.zeros_like(x, dtype=torch.bool), torch.zeros(1, device=dev))[0].float().cpu().numpy()
        for b in range(len(edges) - 1):
            seg = nll[edges[b]:edges[b + 1]]
            sums[b] += seg.sum(); cnt[b] += len(seg)
    return [{"positions": f"{edges[b]}-{edges[b + 1]}", "bits_per_code": round(float(sums[b] / max(1, cnt[b]) / math.log(2)), 3),
             "beyond_trained_window": edges[b] >= eng.a.ctx} for b in range(len(edges) - 1)]


def recall(eng, pairs, rng, distances=(2, 5, 10, 20, 40), n=15):
    G3.setup(test=True)                       # reserved names and phrasings only
    rows = []
    for d in distances:
        for mem in (True, False):
            ok = 0
            for _ in range(n):
                f = G.Fact(rng)
                eng.forget()
                eng.remember(f.stored()[1].encode(), b"Got it. I'll remember that.")
                for j in rng.choice(len(pairs), d, replace=False):
                    eng.remember(*pairs[int(j)])
                ans, info = eng.reply(f.ask().encode(), 60, use_memory=mem)
                ok += f.value.lower() in ans.decode("utf-8", "replace").lower()
            rows.append({"filler_turns": d, "session_memory": mem, "recalled": f"{ok}/{n}", "share": round(ok / n, 3),
                         "fact_in_window": info["turns_in_window"] > d})
            print("recall", rows[-1], flush=True)
    eng.forget()
    return rows


def stories(eng, max_codes=600):
    rows, md = [], []
    for prompt, name in zip(STORIES, NAMES):
        eng.forget()
        ans, info = eng.reply(prompt.encode(), max_codes, use_memory=False)
        text = ans.decode("utf-8", "replace")
        words = text.lower().split()
        tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        rows.append({"prompt": prompt, "chars": len(text), "words": len(words), "codes": info["codes"], "stopped": info["stopped"],
                     "distinct_3grams": round(len(set(tri)) / max(1, len(tri)), 3), "name_in_second_half": name.lower() in " ".join(words[len(words) // 2:]),
                     "seconds": info["seconds"], "ms_per_code": round(1000 * info["seconds"] / max(1, info["codes"]), 1)})
        md.append(f"## {prompt}\n\n{text}\n\n*{len(words)} words, {info['codes']} codes, {info['seconds']} s, stopped: {info['stopped']}*\n")
        print("story", {k: v for k, v in rows[-1].items() if k != "prompt"}, flush=True)
    return rows, md


def probes(eng):
    res = {}
    rows = []
    for group, earlier, final, keys in mt_probe.CONVERSATIONS:
        eng.forget()
        for u in earlier:
            r, _ = eng.reply(u.encode(), 80); eng.remember(u.encode(), r)
        ans, _ = eng.reply(final.encode(), 80)
        rows.append((group, mt_probe.check(ans.decode("utf-8", "replace"), keys)))
    res["mt_probe"] = {"total": f"{sum(o for _, o in rows)}/{len(rows)}", **{g: f"{sum(o for gg, o in rows if gg == g)}/{sum(gg == g for gg, _ in rows)}" for g in dict.fromkeys(g for g, _ in rows)}}
    rows = []
    for group, turns, final, keys, wrong in mt_probe2.build():
        eng.forget()
        for u in turns:
            r, _ = eng.reply(u.encode(), 60); eng.remember(u.encode(), r)
        ans, _ = eng.reply(final.encode(), 60)
        low = ans.decode("utf-8", "replace").lower()
        rows.append((group, all(any(k in low for k in g) for g in keys)))
    res["mt_probe2"] = {"total": f"{sum(o for _, o in rows)}/{len(rows)}", **{g: f"{sum(o for gg, o in rows if gg == g)}/{sum(gg == g for gg, _ in rows)}" for g in dict.fromkeys(g for g, _ in rows)}}
    eng.forget()
    print("probes", res, flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--device", default="cuda"); ap.add_argument("--cpu-threads", type=int, default=16)
    ap.add_argument("--tag", default=""); ap.add_argument("--quick", action="store_true", help="tiny version of every test (checks the script)")
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    native = str(ROOT / "data" / "cv0.exe")
    eng = CE.ChatEngine(str(ROOT / a.model), native=native, device=a.device, log=lambda *x: None)
    pairs = heldout_pairs()
    res = {"model": a.model, "window": eng.a.ctx, "date": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    lengths = (256, 1024) if a.quick else (256, 512, 1024, 1536, 1900)
    res["speed"] = speed(eng, pairs, lengths, a.device)
    res["position"] = position(eng, 2 if a.quick else 12, 1024 if a.quick else 2 * eng.a.ctx)
    print("position", res["position"], flush=True)
    res["recall"] = recall(eng, pairs, rng, (2, 10) if a.quick else (2, 5, 10, 20, 40), 2 if a.quick else 15)
    srows, md = stories(eng, 120 if a.quick else 600)
    res["stories"] = srows
    if not a.quick:
        res["probes"] = probes(eng)
    if a.device != "cpu" and not a.quick:   # the same speed test on the CPU
        torch.set_num_threads(a.cpu_threads)
        cpu = CE.ChatEngine(str(ROOT / a.model), native=native, device="cpu", log=lambda *x: None)
        res["speed"] += speed(cpu, pairs, lengths, f"cpu-{a.cpu_threads}")
    res["seconds"] = round(time.time() - t0)
    stem = Path(a.model).stem + a.tag
    (ROOT / "results" / f"chat_tests_{stem}.json").write_text(json.dumps(res, indent=1))
    (ROOT / "docs" / "chat_probes").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "chat_probes" / f"stories_{stem}.md").write_text(f"# Short stories: `{a.model}`\n\n" + "\n".join(md), encoding="utf-8")
    print("done", res["seconds"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
