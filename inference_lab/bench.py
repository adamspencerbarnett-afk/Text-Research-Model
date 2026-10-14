"""Inference speed lab, phase A: CPU only (14 October 2026, Adam's request).

Runs from copies of the research scripts in inference_lab/lib, so the running research queue never imports
anything changed here. Threads are capped at 8 so the GPU queue's data loading is not disturbed.

    .venv/Scripts/python.exe -B inference_lab/bench.py core      # core models: decoding path, threads, precision, prompt length
    .venv/Scripts/python.exe -B inference_lab/bench.py qa        # Q&A models: recompute vs KV cache, threads
    .venv/Scripts/python.exe -B inference_lab/bench.py compile   # torch.compile on the CPU (needs a C++ compiler inductor accepts)

Measures, greedy decoding so every path should give the same codes:
  first_ms   time to the first new code (prompt processed + one pick): what a streaming chat shows as the wait
  ms_code    mean time per further code
  cps        characters per second of the generated text
  same       share of generated codes equal to the reference path (fp32, full recompute)
Writes inference_lab/results/<part>.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB / "lib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import ladder_encodings as LE  # noqa: E402
import train_lm  # noqa: E402

assert Path(train_lm.__file__).resolve().parent == LAB / "lib", "must import the lab copies"
NATIVE = str(ROOT / "data" / "cv0.exe")
OUT = LAB / "results"
MAX_THREADS = 8

CORES = [("5M", "runs/core_modern512/scale_5M_v0_8k_ng64.pt"),
         ("50M", "runs/core_P50/scale_50M_v0_8k.pt"),
         ("100M", "runs/core_P100/scale_custom_v0_8k.pt")]
QAS = [("5M chat", "models/qa_best_5M_dense.pt"), ("50M chat", "runs/s2b_50m/qa_s2b_50m.pt")]


def load_core(path: Path):
    """As core_eval.load_core: the network only (count tables and Q&A heads are not used by the core chat)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info, shape = ck["encoding"], ck["model"]
    dict_path = Path(info["dictionary"])
    for p in (dict_path, ROOT / dict_path, path.parent / dict_path.name):
        if p.exists():
            dict_path = p; break
    enc = LE.V0Dict(str(dict_path), NATIVE)
    model = train_lm.LM(enc, shape["width"], shape["layers"], shape["heads"], shape["ctx"], 0, None, (2,), [],
                        block=shape.get("block", "legacy"))
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith(("mix_head", "table", "copy_q", "copy_k"))}
    model.load_state_dict(sd, strict=False)
    return enc, model.eval()


def prompt_codes(enc, n: int) -> list[int]:
    text = b"".join(p.read_bytes() for p in sorted((ROOT / "data_core" / "eval_clean" / "heldout").glob("*.txt"))[:2])
    return [int(c) for c in enc.encode(text[:200_000])[0][1000:1000 + n]]


def logits_of(model, h):
    lg = (h.to(model.emb.weight.dtype) @ model.emb.weight.T).float()
    if model.no_case_flags:
        lg[..., 256:258] = float("-inf")
    return lg


@torch.no_grad()
def gen_recompute(model, prompt: list[int], n: int):
    seq, times, dev = list(prompt), [], model.emb.weight.device
    for _ in range(n):
        t0 = time.perf_counter()
        x = torch.tensor([seq[-model.ctx:]], device=dev)
        seq.append(int(logits_of(model, model.hidden(x)[:, -1]).argmax(-1)))
        times.append(time.perf_counter() - t0)
    return seq[len(prompt):], times


@torch.no_grad()
def gen_cached(model, prompt: list[int], n: int):
    """KV cache: the prompt goes through once, then one code per step (requires prompt + n <= ctx)."""
    assert len(prompt) + n <= model.ctx
    times, out, dev = [], [], model.emb.weight.device
    t0 = time.perf_counter()
    h, cache = model.hidden_cached(torch.tensor([prompt], device=dev), 0, None)
    nxt = int(logits_of(model, h[:, -1]).argmax(-1)); out.append(nxt)
    times.append(time.perf_counter() - t0)
    for i in range(n - 1):
        t0 = time.perf_counter()
        h, cache = model.hidden_cached(torch.tensor([[nxt]], device=dev), len(prompt) + i, cache)
        nxt = int(logits_of(model, h[:, -1]).argmax(-1)); out.append(nxt)
        times.append(time.perf_counter() - t0)
    return out, times


def summarise(enc, codes, times, ref):
    text = enc.decode(codes)
    rest = times[1:] or times
    same = sum(int(a == b) for a, b in zip(codes, ref)) / max(1, len(ref)) if ref else 1.0
    return {"first_ms": round(1000 * times[0], 1), "ms_code": round(1000 * float(np.mean(rest)), 2),
            "cps": round(len(text) / sum(times), 1), "codes": len(codes), "same": round(same, 3),
            "text": text[:80].decode("utf-8", "replace")}


def run_core(quick: bool = False):
    rows = []
    for name, rel in CORES:
        path = ROOT / rel
        if not path.exists():
            print("missing", path); continue
        enc, model = load_core(path)
        q8 = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        lengths = [32, 256, model.ctx - 64]
        for L in lengths:
            prompt = prompt_codes(enc, L)
            torch.set_num_threads(MAX_THREADS)
            n_rec = 24 if name == "100M" else 48
            ref, t = gen_recompute(model, prompt, n_rec)          # reference path: full window each code (today's core chat)
            r = summarise(enc, ref, t, None); r.update(model=name, prompt=L, path="recompute", precision="fp32", threads=MAX_THREADS)
            rows.append(r); print(r, flush=True)
            for threads in ([MAX_THREADS] if quick else [1, 2, 4, 8]):
                torch.set_num_threads(threads)
                for prec, m in (("fp32", model), ("int8", q8)):
                    gen_cached(m, prompt, 4)                          # warm-up
                    codes, t = gen_cached(m, prompt, 64)
                    r = summarise(enc, codes, t, ref); r.update(model=name, prompt=L, path="cache", precision=prec, threads=threads)
                    rows.append(r); print(r, flush=True)
            if L == lengths[1]:                                       # bf16 compute on a CPU without bf16 units, once per model
                torch.set_num_threads(MAX_THREADS)
                mb = load_core(path)[1].to(torch.bfloat16)
                gen_cached(mb, prompt, 4)
                codes, t = gen_cached(mb, prompt, 64)
                r = summarise(enc, codes, t, ref); r.update(model=name, prompt=L, path="cache", precision="bf16", threads=MAX_THREADS)
                rows.append(r); print(r, flush=True)
                del mb
        del model, q8
    return rows


QUESTIONS = ["Name a food that is traditionally served during Thanksgiving in the US.",
             "What food do Americans usually eat on Thanksgiving?",
             "Create a list of 10 verbs related to cooking",
             "What is the capital of France?",
             "Explain why the sky is blue.",
             "Give me three tips for staying healthy."]


def run_qa():
    import m2_qa
    assert Path(m2_qa.__file__).resolve().parent == LAB / "lib"
    rows = []
    for name, rel in QAS:
        torch.set_num_threads(MAX_THREADS)
        t0 = time.perf_counter()
        eng = m2_qa.QAEngine(str(ROOT / rel), native=NATIVE, device="cpu", log=lambda *a: None)
        load_s = time.perf_counter() - t0
        variants = [("recompute", False, "fp32"), ("cache", True, "fp32")]
        ref = {}
        for threads in [1, 2, 4, 8]:
            torch.set_num_threads(threads)
            for label, fast, prec in variants:
                if label == "recompute" and threads not in (1, 8):
                    continue
                eng.ask(QUESTIONS[0].encode(), 8, 3, fast=fast)        # warm-up
                tot, first, chars, codes, same = 0.0, 0.0, 0, 0, 0
                for q in QUESTIONS:
                    t1 = time.perf_counter(); eng.ask(q.encode(), 1, 3, fast=fast); first += time.perf_counter() - t1
                    t1 = time.perf_counter(); out, info = eng.ask(q.encode(), 120, 3, fast=fast); tot += time.perf_counter() - t1
                    chars += len(out); codes += info["codes"]
                    ref.setdefault(q, out); same += int(out == ref[q])
                n = len(QUESTIONS)
                r = {"model": name, "path": label, "precision": prec, "threads": threads, "load_s": round(load_s, 1),
                     "first_ms": round(1000 * first / n, 1), "s_answer": round(tot / n, 3), "ms_code": round(1000 * tot / max(1, codes), 2),
                     "cps": round(chars / tot, 1), "same_answers": f"{same}/{n}"}
                rows.append(r); print(r, flush=True)
        # real int8 kernels (dynamic quantisation of every Linear) on the cached path, 8 threads
        torch.set_num_threads(MAX_THREADS)
        eng.model = torch.ao.quantization.quantize_dynamic(eng.model, {nn.Linear}, dtype=torch.qint8)
        eng.ask(QUESTIONS[0].encode(), 8, 3, fast=True)
        tot, first, chars, codes, same = 0.0, 0.0, 0, 0, 0
        for q in QUESTIONS:
            t1 = time.perf_counter(); eng.ask(q.encode(), 1, 3, fast=True); first += time.perf_counter() - t1
            t1 = time.perf_counter(); out, info = eng.ask(q.encode(), 120, 3, fast=True); tot += time.perf_counter() - t1
            chars += len(out); codes += info["codes"]; same += int(out == ref[q])
        n = len(QUESTIONS)
        r = {"model": name, "path": "cache", "precision": "int8 dynamic", "threads": MAX_THREADS, "load_s": round(load_s, 1),
             "first_ms": round(1000 * first / n, 1), "s_answer": round(tot / n, 3), "ms_code": round(1000 * tot / max(1, codes), 2),
             "cps": round(chars / tot, 1), "same_answers": f"{same}/{n}"}
        rows.append(r); print(r, flush=True)
        del eng
    return rows


def run_compile():
    rows = []
    name, rel = CORES[0]
    enc, model = load_core(ROOT / rel)
    torch.set_num_threads(MAX_THREADS)
    prompt = prompt_codes(enc, 256)
    ref, t = gen_cached(model, prompt, 64)
    base = summarise(enc, ref, t, None); base.update(model=name, path="cache eager"); rows.append(base); print(base)
    try:
        t0 = time.perf_counter()
        model.hidden_cached = torch.compile(model.hidden_cached, dynamic=True)
        gen_cached(model, prompt, 4)
        comp_s = time.perf_counter() - t0
        codes, t = gen_cached(model, prompt, 64)
        r = summarise(enc, codes, t, ref); r.update(model=name, path="cache compiled", compile_s=round(comp_s, 1))
    except Exception as e:  # noqa: BLE001
        r = {"model": name, "path": "cache compiled", "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
    rows.append(r); print(r)
    return rows


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "core"
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    rows = {"core": run_core, "qa": run_qa, "compile": run_compile}[part]()
    meta = {"part": part, "date": time.strftime("%Y-%m-%d %H:%M"), "device": "cpu", "torch": torch.__version__,
            "max_threads": MAX_THREADS, "seconds": round(time.time() - t0)}
    (OUT / f"{part}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print("wrote", OUT / f"{part}.json", meta)
