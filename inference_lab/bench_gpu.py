"""Inference speed lab, phase B: GPU against CPU (runs at the end of the research queue, on an idle GPU).

    .venv/Scripts/python.exe -B inference_lab/bench_gpu.py

1. Core models on the GPU: full recompute, KV cache, fp32 weights and bf16 compute (autocast), prompt lengths.
2. Q&A models on the GPU: recompute, KV cache and the CUDA-graph decoder, fp32 and bf16 kernels, with joules.
3. The CPU with every thread (22 and 44), now that nothing else needs them: KV cache, fp32 and int8.
Writes inference_lab/results/gpu.json. Same measures as bench.py.
"""
from __future__ import annotations

import contextlib
import json
import time

import torch
import torch.nn as nn

import bench
from bench import ROOT, OUT, NATIVE, CORES, QAS, QUESTIONS, load_core, prompt_codes, gen_recompute, gen_cached, summarise
import cost  # lab copy (bench put lib first on the path)


def autocast(prec):
    return torch.autocast("cuda", dtype=torch.bfloat16) if prec == "bf16" else contextlib.nullcontext()


def core_gpu(rows):
    for name, rel in CORES:
        enc, model = load_core(ROOT / rel)
        model.cuda()
        for L in (32, 256, model.ctx - 64):
            prompt = prompt_codes(enc, L)
            ref, t = gen_recompute(model, prompt, 48)
            r = summarise(enc, ref, t, None); r.update(model=name, device="cuda", prompt=L, path="recompute", precision="fp32"); rows.append(r); print(r, flush=True)
            for prec in ("fp32", "bf16"):
                with autocast(prec):
                    gen_cached(model, prompt, 4)
                    codes, t = gen_cached(model, prompt, 64)
                    if prec == "bf16":
                        _, tr = gen_recompute(model, prompt, 24)
                r = summarise(enc, codes, t, ref); r.update(model=name, device="cuda", prompt=L, path="cache", precision=prec); rows.append(r); print(r, flush=True)
                if prec == "bf16":
                    r = summarise(enc, _, tr, ref); r.update(model=name, device="cuda", prompt=L, path="recompute", precision="bf16"); rows.append(r); print(r, flush=True)
        del model; torch.cuda.empty_cache()


def qa_gpu(rows):
    import m2_qa
    for name, rel in QAS:
        ref = {}
        for kernels in ("fp32", "bf16"):
            eng = m2_qa.QAEngine(str(ROOT / rel), native=NATIVE, device="cuda", kernels=kernels, log=lambda *a: None)
            for label, fast in (("recompute", False), ("cache", True), ("graph", "graph")):
                eng.ask(QUESTIONS[0].encode(), 8, 3, fast=fast)
                tot, first, chars, codes, same = 0.0, 0.0, 0, 0, 0
                with cost.Meter("cuda", period_s=0.5) as m:
                    for q in QUESTIONS * 3:
                        t1 = time.perf_counter(); eng.ask(q.encode(), 1, 3, fast=fast); first += time.perf_counter() - t1
                        t1 = time.perf_counter(); out, info = eng.ask(q.encode(), 120, 3, fast=fast); tot += time.perf_counter() - t1
                        chars += len(out); codes += info["codes"]; ref.setdefault(q, out); same += int(out == ref[q])
                n = 3 * len(QUESTIONS)
                r = {"model": name, "device": "cuda", "path": label, "precision": kernels, "first_ms": round(1000 * first / n, 1),
                     "s_answer": round(tot / n, 3), "ms_code": round(1000 * tot / max(1, codes), 2), "cps": round(chars / tot, 1),
                     "same_answers": f"{same}/{n}", "j_answer": m.summary(tot + first, 2 * n).get("j_per_unit")}
                rows.append(r); print(r, flush=True)
            del eng; torch.cuda.empty_cache()


def cpu_all_threads(rows):
    for name, rel in CORES:
        enc, model = load_core(ROOT / rel)
        q8 = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        prompt = prompt_codes(enc, 256)
        ref, _ = gen_cached(model, prompt, 64)
        for threads in (8, 16, 22, 44):
            torch.set_num_threads(threads)
            for prec, m in (("fp32", model), ("int8", q8)):
                gen_cached(m, prompt, 4)
                codes, t = gen_cached(m, prompt, 64)
                r = summarise(enc, codes, t, ref); r.update(model=name, device="cpu", prompt=256, path="cache", precision=prec, threads=threads)
                rows.append(r); print(r, flush=True)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    rows, t0 = [], time.time()
    core_gpu(rows); qa_gpu(rows); cpu_all_threads(rows)
    meta = {"part": "gpu", "date": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "seconds": round(time.time() - t0)}
    (OUT / "gpu.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print("wrote", OUT / "gpu.json", meta)
