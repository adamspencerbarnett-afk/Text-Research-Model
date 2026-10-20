"""How far can the context be pushed on this machine? (15 October 2026, Adam: "compare to a million tokens")

Two different things are measured, for the 5M chat model and the 100M model:

1. THE RAW WINDOW (attention over everything at once, what large models mean by a 1M-token context)
   For lengths 2k -> 1M codes of unseen held-out text (~3.5 characters per code; a typical token is ~4):
     read time   the whole context goes through the network once, in chunks (= the wait before the first word)
     ms / code   writing 64 codes after it, each attending to everything before
     memory      peak GPU memory; where it runs out
     quality     bits per code by position band (trained windows: 1,024 for the 100M core, 2,048 for the chat model)
     needle      a fact planted at 10% depth ("the secret code word is ..."), asked at the end: found by attention alone?
2. SESSION MEMORY (our design: the window stays 2,048; every turn is stored in RAM and retrieved when relevant)
   A fact told in turn 1, then 10 -> 5,000 real filler turns (up to ~3M codes), then the question; lookup time,
   answer time, recall.

    python compiler/experiments/context_stress.py --device cuda [--max-5m 1048576] [--max-100m 524288] [--quick]
Writes results/context_stress.json.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import core_eval as C  # noqa: E402

MODELS = {"5M": "runs/chat5m_v2/chat5m_v2_A.pt", "100M": "runs/s2b_100m/qa_s2b_100m.pt"}
NEEDLE = b"Remember this for later: the secret code word is tangerine-47."
QUESTION = b"\n\nUser: What is the secret code word?\nAssistant:"
BANDS = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]


def held_text_codes(enc, n_codes):
    files = ["books.txt", "fineweb.txt", "ultrachat.txt", "qqp.txt", "evidence.txt", "oasst.txt"]
    text = b"\n\n".join((ROOT / "data_core" / "eval_clean" / "heldout" / f).read_bytes() for f in files)
    codes = enc.encode(text)[0]
    return codes[:n_codes]


def peak_ram_gb() -> float:
    """This process's peak working set (Windows), in GB."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32"); k32.GetCurrentProcess.restype = wintypes.HANDLE
    gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    gpmi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**30


def total_ram_gb() -> float:
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30


@torch.no_grad()
def raw_window(model, enc, dev, L, budget_s):
    """Read L codes (needle at 10% depth, question at the end) in chunks with a KV cache, then write 64 codes."""
    hay = held_text_codes.cache[: L]
    nd, qc = enc.encode(NEEDLE)[0], enc.encode(QUESTION)[0]
    pos = int(0.1 * L)
    seq = np.concatenate([hay[:pos], nd, hay[pos:L - len(nd) - len(qc)], qc]).astype(np.int64)
    x_all = torch.from_numpy(seq).to(dev)
    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    t0 = time.perf_counter(); cache = None; done = 0
    sums = np.zeros(len(BANDS) - 1); cnts = np.zeros(len(BANDS) - 1)
    while done < len(seq):
        # chunk size keeps the attention mask (chunk x everything so far) near 64 MB
        chunk = int(max(64, min(2048, 2 ** 26 // max(1, done + 2048))))
        x = x_all[done:done + chunk][None]
        h, cache = model.hidden_cached(x, done, cache)
        tgt = x_all[done + 1:done + 1 + x.shape[1]]
        n = len(tgt)
        if n:
            lp = F.log_softmax((h[0, :n].float() @ model.emb.weight.float().T), -1).gather(-1, tgt[:, None])[:, 0]
            nll = (-lp).cpu().numpy() / math.log(2)
            idx = np.arange(done, done + n)
            for b in range(len(BANDS) - 1):
                m = (idx >= BANDS[b]) & (idx < BANDS[b + 1])
                if m.any():
                    sums[b] += nll[m].sum(); cnts[b] += m.sum()
        done += x.shape[1]
        if time.perf_counter() - t0 > budget_s:
            return {"length": L, "status": f"stopped: reading took over {budget_s} s", "read_codes": done}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    read_s = time.perf_counter() - t0
    last = h[:, -1]
    out, times = [], []
    for _ in range(64):
        t1 = time.perf_counter()
        logits = (last.float() @ model.emb.weight.float().T)
        if model.no_case_flags:
            logits[:, 256:258] = float("-inf")
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        h, cache = model.hidden_cached(torch.tensor([[nxt]], device=dev), done, cache); done += 1
        last = h[:, -1]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
    text = enc.decode(out).decode("utf-8", "replace")
    res = {"length": L, "status": "ok", "read_s": round(read_s, 2), "first_word_wait_s": round(read_s + times[0], 2),
           "ms_per_code": round(1000 * float(np.median(times[1:])), 2),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None,
           "peak_ram_gb": round(peak_ram_gb(), 2) if dev.type == "cpu" else None,
           "needle_found": "tangerine" in text.lower(), "reply": text[:100],
           "bits_per_code_by_position": {f"{BANDS[b]}-{BANDS[b + 1]}": round(float(sums[b] / cnts[b]), 3) for b in range(len(BANDS) - 1) if cnts[b]}}
    del cache; torch.cuda.empty_cache() if dev.type == "cuda" else None
    return res


def session_memory(path, dev, turns_list, n_needles=10):
    """A fact in turn 1, N real filler turns stored in session memory, then the question: recall and timing."""
    import chat_engine as CE
    import m2_qa
    ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    chat = bool((ck.get("args") or {}).get("chat_v2"))
    eng = (CE.ChatEngine if chat else m2_qa.QAEngine)(str(ROOT / path), native=str(ROOT / "data" / "cv0.exe"), device=str(dev), log=lambda *x: None)
    pairs = []
    for f in ("heldout", "train"):
        for i, l in enumerate(open(ROOT / "data_mt" / f / "dialogues.jsonl", encoding="utf-8")):
            t = json.loads(l)["turns"]
            pairs += [(t[j][1].encode("utf-8", "replace")[:900], t[j + 1][1].encode("utf-8", "replace")[:1500]) for j in range(0, len(t) - 1, 2)
                      if t[j][0] == "user" and t[j + 1][0] == "assistant"]
            if len(pairs) > max(turns_list) + 100:
                break
        if len(pairs) > max(turns_list) + 100:
            break
    facts = [(f"My locker code at the swimming pool is {4000 + 37 * k}.", "What is my locker code at the swimming pool?", str(4000 + 37 * k)) for k in range(n_needles)]
    base = len(eng.mem.exchanges)
    rows = []
    for n in turns_list:
        ok, look, ans_s, add_s, codes = 0, 0.0, 0.0, 0.0, 0
        for k, (stmt, q, val) in enumerate(facts):
            while len(eng.mem.exchanges) > base:            # start each conversation empty
                if chat:
                    eng.base_n = base; eng.forget(); break
                eng.mem.exchanges.pop(); kk = eng.mem.keys.pop()
                if eng.mem.exact.get(kk) == len(eng.mem.exchanges):
                    del eng.mem.exact[kk]
                for w in kk:
                    eng.mem.index[w].pop(); eng.mem.df[w] -= 1
                if eng.mem.learned is not None:
                    eng.mem.learned.vectors = eng.mem.learned.vectors[:-1]
            t0 = time.perf_counter()
            eng.add(stmt.encode(), b"Got it. I'll remember that.")
            off = (k * 997) % max(1, len(pairs) - n)
            for u, r in pairs[off:off + n]:
                eng.add(u, r)
            add_s += time.perf_counter() - t0
            if k == 0:
                codes = sum(len(eng.enc.encode(m2_qa.render(u, r))[0]) for u, r in pairs[off:off + min(n, 300)]) * n / max(1, min(n, 300))
            t1 = time.perf_counter()
            r_ = eng.mem.retrieve(q.encode(), 3)
            look += time.perf_counter() - t1
            t2 = time.perf_counter()
            if chat:
                a, _ = eng.reply(q.encode(), 60, history=[], verify=0)
            else:
                a, _ = eng.ask(q.encode(), 60, 3, fast=True)
            ans_s += time.perf_counter() - t2
            ok += val in a.decode("utf-8", "replace")
        rows.append({"filler_turns": n, "approx_codes_in_memory": int(codes), "approx_characters": int(codes * 3.5),
                     "recalled": f"{ok}/{n_needles}", "store_s_per_conversation": round(add_s / n_needles, 2),
                     "lookup_ms": round(1000 * look / n_needles, 1), "answer_s": round(ans_s / n_needles, 3)})
        print("session memory", path, rows[-1], flush=True)
    del eng; torch.cuda.empty_cache() if dev.type == "cuda" else None
    return rows


def cpu_stress(a) -> int:
    """Raw window on the CPU, both models, doubling from 2,048 codes: stops at the time budget per length, or before a
    length whose memory (about 2.5x the last) would pass 3/4 of the PC's RAM."""
    torch.set_num_threads(a.cpu_threads)
    cap = 0.75 * total_ram_gb()
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "device": f"cpu-{a.cpu_threads}", "ram_cap_gb": round(cap, 1), "raw_window": {}}
    native = str(ROOT / "data" / "cv0.exe")
    for name, path in MODELS.items():
        if a.only and name != a.only:
            continue
        enc, model = C.load_core(str(ROOT / path), native, "cpu")
        model = model.float().eval()
        held_text_codes.cache = held_text_codes(enc, a.max_cpu + 10)
        rows, L, base = [], 2048, peak_ram_gb()
        while L <= a.max_cpu:
            r = raw_window(model, enc, torch.device("cpu"), L, a.budget_s)
            rows.append(r); print("raw window CPU", name, {k: v for k, v in r.items() if k != "bits_per_code_by_position"}, flush=True)
            if r["status"] != "ok":
                break
            if base + (r["peak_ram_gb"] - base) * 2.5 > cap:
                rows.append({"length": L * 2, "status": f"skipped: would pass 3/4 of RAM ({cap:.0f} GB)"}); break
            L *= 2
        res["raw_window"][name] = rows
        del model
        (ROOT / "results" / f"context_stress_cpu{'_' + a.only if a.only else ''}.json").write_text(json.dumps(res, indent=1))
    print("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--max-5m", type=int, default=1048576)
    ap.add_argument("--max-100m", type=int, default=524288); ap.add_argument("--budget-s", type=float, default=900)
    ap.add_argument("--cpu-threads", type=int, default=16); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--memory-only", action="store_true", help="keep the saved raw-window results; run only the session-memory part")
    ap.add_argument("--cpu-stress", action="store_true", help="raw window on the CPU for both models (results/context_stress_cpu.json)")
    ap.add_argument("--max-cpu", type=int, default=262144)
    ap.add_argument("--only", default="", help="one model (5M or 100M); the CPU test runs each in its own process (peak RAM never resets)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":      # Adam, 16 October: stress tests use at most 3/4 of the GPU (fail cleanly, never page to RAM)
        torch.cuda.set_per_process_memory_fraction(0.75)
    if a.cpu_stress:
        return cpu_stress(a)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "device": a.device, "raw_window": {}, "session_memory": {}}
    native = str(ROOT / "data" / "cv0.exe")
    out = ROOT / "results" / "context_stress.json"
    if a.memory_only and out.exists():
        res = json.loads(out.read_text()); res["session_memory"] = {}
    for name, path in (MODELS.items() if not a.memory_only else []):
        enc, model = C.load_core(str(ROOT / path), native, a.device)
        model = model.to(torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
        top = (8192 if a.quick else (a.max_5m if name == "5M" else a.max_100m))
        held_text_codes.cache = held_text_codes(enc, top + 10)
        print(name, f"{len(held_text_codes.cache):,} held-out codes ready", flush=True)
        rows, L = [], 2048
        while L <= top:
            try:
                r = raw_window(model, enc, dev, L, a.budget_s)
            except torch.cuda.OutOfMemoryError:
                r = {"length": L, "status": "out of GPU memory"}; torch.cuda.empty_cache()
            rows.append(r); print("raw window", name, {k: v for k, v in r.items() if k != "bits_per_code_by_position"}, flush=True)
            if r["status"] != "ok":
                break
            # 16 October: past the GPU's memory Windows pages to system RAM instead of failing, and the 100M froze for
            # 9.5 h at 131,072 codes. Stop when the next length (about 2.5x the memory) would not fit comfortably.
            if dev.type == "cuda" and r.get("peak_gpu_gb", 0) * 2.5 > 0.75 * torch.cuda.get_device_properties(0).total_memory / 2**30:
                rows.append({"length": L * 2, "status": "skipped: would not fit in GPU memory"}); break
            L *= 2
        res["raw_window"][name] = rows
        if dev.type == "cuda" and name == "5M" and not a.quick:          # the same on the CPU, short lengths
            torch.set_num_threads(a.cpu_threads)
            enc_c, m_c = C.load_core(str(ROOT / path), native, "cpu")
            cpu_rows = []
            for L in (2048, 8192, 32768):
                r = raw_window(m_c.float().eval(), enc_c, torch.device("cpu"), L, a.budget_s)
                cpu_rows.append(r); print("raw window CPU", {k: v for k, v in r.items() if k != "bits_per_code_by_position"}, flush=True)
            res["raw_window"]["5M_cpu"] = cpu_rows
        del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
        (ROOT / "results" / "context_stress.json").write_text(json.dumps(res, indent=1))
    turns = [10, 100] if a.quick else [10, 100, 1000, 5000]
    for name, path in MODELS.items():
        res["session_memory"][name] = session_memory(path, dev, turns, 3 if a.quick else 10)
        (ROOT / "results" / "context_stress.json").write_text(json.dumps(res, indent=1))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
