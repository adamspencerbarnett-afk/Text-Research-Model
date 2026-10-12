"""Cost meter: GPU power, energy and peak memory over a span of work, so every run records what it
cost beside what it scored.

    with cost.Meter(device) as m:
        ... train or answer ...
    m.summary(seconds, units)   -> {"gpu_avg_w", "gpu_kj", "peak_mem_mb", "j_per_unit"}

Power is sampled from nvidia-smi every two seconds in a background thread (board power draw,
the whole card). Peak memory is torch's allocator high-water mark. On the CPU, or when
nvidia-smi is absent, the power fields are None and the rest still works.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time


def read_power_w() -> float | None:
    """Current board power draw of GPU 0 in watts, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


class Meter:
    def __init__(self, device: str = "cpu", period_s: float = 2.0):
        self.gpu = str(device).startswith("cuda")
        self.period, self.samples, self._stop, self._thread = period_s, [], threading.Event(), None
        self.t0 = self.seconds = 0.0

    def _poll(self):
        while not self._stop.is_set():
            w = read_power_w()
            if w is not None:
                self.samples.append(w)
            self._stop.wait(self.period)

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.gpu:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            if read_power_w() is not None:
                self._thread = threading.Thread(target=self._poll, daemon=True)
                self._thread.start()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.period + 1)
        return False

    def summary(self, seconds: float | None = None, units: int | None = None) -> dict:
        """seconds: the span to charge (default: the whole metered span); units: steps or answers."""
        s = self.seconds if seconds is None else seconds
        avg = sum(self.samples) / len(self.samples) if self.samples else None
        peak = None
        if self.gpu:
            try:
                import torch
                peak = round(torch.cuda.max_memory_allocated() / 2**20)
            except Exception:
                pass
        out = {"seconds": round(s, 1), "gpu_avg_w": None if avg is None else round(avg, 1),
               "gpu_kj": None if avg is None else round(avg * s / 1000, 2), "peak_mem_mb": peak, "power_samples": len(self.samples)}
        if units:
            out["units"] = units
            out["s_per_unit"] = round(s / units, 4)
            out["j_per_unit"] = None if avg is None else round(avg * s / units, 2)
        return out
