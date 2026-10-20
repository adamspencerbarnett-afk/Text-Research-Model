"""Disk speed test for the attention cache (16 October 2026, Adam: "run some tests using hard drive just to gauge speed,
no need for huge tests"). Where can the KV cache of a long window live, and what does each place cost per written code?

  gpu    the cache stays in GPU memory (session 36's setting)
  ram    the cache lives in system RAM and is copied to the GPU for every code
  disk   the cache lives in a file on the NVMe drive (outside OneDrive) and is read back for every code, unbuffered
         (FILE_FLAG_NO_BUFFERING, so Windows' file cache cannot turn the test into a RAM test)

Small sizes only: 5M chat model 8k -> 256k codes (cache 0.05 -> 1.6 GB), 100M 8k -> 64k (0.27 -> 2.1 GB); 16 codes
written per setting (median). Plus the drive's own unbuffered read speed on a 4 GB file. GPU use capped at 3/4.

    python compiler/experiments/disk_kv_bench.py [--dir C:/stress_cache]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import context_stress as CS  # noqa: E402
import core_eval as C  # noqa: E402

LENGTHS = {"5M": [8192, 32768, 131072, 262144], "100M": [8192, 16384, 32768, 65536]}
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


class Unbuffered:
    """Reads a whole file past the Windows file cache into one page-aligned buffer (reused between reads)."""

    def __init__(self, path: str, size: int):
        self.path, self.size = path, size
        self.cap = (size + 4095) // 4096 * 4096
        self.buf = k32.VirtualAlloc(None, self.cap, 0x3000, 0x04)        # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE

    def read(self) -> np.ndarray:
        h = k32.CreateFileW(self.path, 0x80000000, 1, None, 3, 0x20000000 | 0x08000000, None)   # NO_BUFFERING | SEQUENTIAL
        if h == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), "CreateFileW")
        got, pos, n = wintypes.DWORD(0), 0, 64 << 20
        while pos < self.cap:
            k32.ReadFile(h, self.buf + pos, min(n, self.cap - pos), ctypes.byref(got), None)
            if got.value == 0:
                break
            pos += got.value
        k32.CloseHandle(h)
        return np.ctypeslib.as_array((ctypes.c_uint8 * self.size).from_address(self.buf))

    def free(self):
        k32.VirtualFree(self.buf, 0, 0x8000)


def write_padded(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data); f.write(b"\0" * ((-len(data)) % 4096))


def drive_speed(d: Path, gb: int = 4) -> dict:
    p = str(d / "speed.bin"); size = gb << 30
    with open(p, "wb") as f:
        block = os.urandom(64 << 20)
        for _ in range(size // len(block)):
            f.write(block)
    r = Unbuffered(p, size); t = time.perf_counter(); r.read(); dt = time.perf_counter() - t; r.free()
    os.remove(p)
    return {"file_gb": gb, "unbuffered_read_gb_per_s": round(size / dt / 1e9, 2)}


@torch.no_grad()
def bench(model, enc, L, d: Path, steps: int = 16) -> dict:
    dev = torch.device("cuda")
    x = torch.from_numpy(CS.held_text_codes.cache[:L].astype(np.int64)).to(dev)
    cache, done = None, 0
    while done < L:                                          # read the context once, on the GPU, in chunks
        chunk = int(max(64, min(2048, 2 ** 26 // max(1, done + 2048))))
        h, cache = model.hidden_cached(x[done:done + chunk][None], done, cache); done += chunk
    nxt = int((h[:, -1].float() @ model.emb.weight.float().T).argmax(-1))
    shapes = [(k.shape, v.shape) for k, v in cache]
    row = {"length": L, "cache_gb": round(sum(k.numel() + v.numel() for k, v in cache) * 2 / 1e9, 3)}

    def step(past, pos, tok):
        torch.cuda.synchronize(); t = time.perf_counter()
        h_, _ = model.hidden_cached(torch.tensor([[tok]], device=dev), pos, past()); torch.cuda.synchronize()
        return time.perf_counter() - t

    # gpu: the cache stays where it is
    row["gpu_ms_per_code"] = round(1000 * float(np.median([step(lambda: cache, L, nxt) for _ in range(steps)])), 2)
    # ram: the cache in system RAM (pinned), copied to the GPU for every code
    ram = [(k.cpu().pin_memory(), v.cpu().pin_memory()) for k, v in cache]
    row["ram_ms_per_code"] = round(1000 * float(np.median([step(lambda: [(k.to(dev, non_blocking=True), v.to(dev, non_blocking=True)) for k, v in ram], L, nxt)
                                                           for _ in range(steps)])), 2)
    # disk: the cache in a file, read back past the file cache for every code
    flat = torch.cat([t.reshape(-1) for kv in ram for t in kv]).view(torch.int16).numpy().tobytes()
    p = str(d / f"kv_{L}.bin"); write_padded(p, flat); del flat, ram
    sizes = [int(np.prod(s)) for kv in shapes for s in kv]
    reader = Unbuffered(p, sum(sizes) * 2)

    def from_disk():
        a = torch.from_numpy(reader.read().view(np.int16)).view(torch.bfloat16).to(dev)
        parts, o = [], 0
        for kv in shapes:
            pair = []
            for s in kv:
                n = int(np.prod(s)); pair.append(a[o:o + n].view(s)); o += n
            parts.append(tuple(pair))
        return parts
    row["disk_ms_per_code"] = round(1000 * float(np.median([step(from_disk, L, nxt) for _ in range(steps)])), 2)
    reader.free(); os.remove(p)
    del cache; torch.cuda.empty_cache()
    for k in ("ram", "disk"):
        row[f"{k}_vs_gpu"] = round(row[f"{k}_ms_per_code"] / row["gpu_ms_per_code"], 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/stress_cache"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    assert "onedrive" not in str(d.resolve()).lower(), "keep the cache files outside OneDrive"
    torch.cuda.set_per_process_memory_fraction(0.75)        # Adam: stress tests use at most 3/4 of the GPU
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "dir": str(d), "drive": drive_speed(d, 1 if a.quick else 4), "rows": {}}
    print("drive", res["drive"], flush=True)
    native = str(ROOT / "data" / "cv0.exe")
    for name, path in CS.MODELS.items():
        enc, model = C.load_core(str(ROOT / path), native, "cuda")
        model = model.to(torch.bfloat16).eval()
        lengths = LENGTHS[name][:2] if a.quick else LENGTHS[name]
        CS.held_text_codes.cache = CS.held_text_codes(enc, max(lengths) + 10)
        res["rows"][name] = []
        for L in lengths:
            r = bench(model, enc, L, d); res["rows"][name].append(r); print(name, r, flush=True)
        del model; torch.cuda.empty_cache()
    out = ROOT / "results" / f"disk_kv_bench{'_quick' if a.quick else ''}.json"
    out.write_text(json.dumps(res, indent=1)); print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
