# Inference speed lab

*14 October 2026, at Adam's request: how fast can replies get on this machine, on the CPU or the GPU, and
with which settings? Run from copies of the research scripts (`lib/`) so the research queue never imports
anything changed here. Phase A ran on the CPU at no more than 8 threads beside the GPU queue. Phase B
(GPU) runs after session 31 on an idle GPU.*

Machine: Xeon E5-2699 v4 (22 cores, 44 threads, AVX2, no bf16 or int8 dot-product units), L3 55 MB,
DDR4-2133; RTX 3090 24 GB. PyTorch 2.11.

| file | what |
| --- | --- |
| `bench.py core` | core models: full recompute against KV cache, threads 1-8, fp32 / int8 / bf16, prompt 32 / 256 / window-64 codes |
| `bench.py qa` | Q&A models through the engine: recompute against KV cache, threads, int8 |
| `bench.py compile` | torch.compile on the CPU |
| `bench_gpu.py` | phase B: GPU recompute / cache / CUDA graph, fp32 and bf16, joules; CPU at 8-44 threads |
| `chain_gpu.sh` | waits for session 31's last report card and an idle GPU, then runs phase B |
| `run_model_lab.py`, `test_chat_lab.py` | the chat changes, tested here before they went into `run_model.py` |
| `results/*.json` | every measurement |

Greedy decoding throughout, so every path should give the same codes. *same* is the share of generated
codes equal to the reference (fp32, full recompute). *first* is the time to the first new code, the wait
a streaming chat shows.

## Phase A results (CPU)

**1. The KV cache: same codes, much faster, more so with longer prompts.** Per code, 8 threads:

| model | prompt (codes) | recompute ms/code | cache ms/code | speed-up | same |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M core | 32 | 13.1 | 6.7 | 2.0x | 1.0 |
| 5M core | 448 | 46.4 | 7.8 | 5.9x | 1.0 |
| 50M core | 32 | 35.5 | 13.3 | 2.7x | 1.0 |
| 50M core | 960 | 468.0 | 21.1 | 22x | 1.0 |
| 100M core | 32 | 60.8 | 22.1 | 2.8x | 1.0 |
| 100M core | 256 | 258.0 | 25.3 | 10x | 1.0 |
| 100M core | 960 | 923.3 | 32.4 | 28x | 1.0 |

Q&A models through the engine (6 questions, up to 120 codes; the engine already had the cache, the
chat used it):

| model | threads | recompute s/answer | cache s/answer | first code ms | same answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M chat | 1 | 1.75 | 0.51 | 42 | 6/6 |
| 5M chat | 8 | 0.66 | 0.38 | 20 | 6/6 |
| 50M chat | 1 | 12.29 | 1.78 | 223 | 6/6 |
| 50M chat | 8 | 2.50 | 0.78 | 52 | 6/6 |

**2. Threads.** Cached, ms per code (prompt 960 codes for 50M and 100M, 448 for 5M):

| model | 1 | 2 | 4 | 8 | first code at 1 / 8 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M core | 13.1 | 9.9 | 8.2 | 7.8 | 139 / 47 ms |
| 50M core | 51.5 | 37.7 | 21.7 | 21.1 | 2.4 / 0.55 s |
| 100M core | 82.8 | 52.6 | 36.0 | 32.4 | 5.5 / 0.87 s |

Small models stop gaining past 2-4 threads (the work per code is too small to share). The prompt pass
keeps gaining, because it is one large batch of matrix work. Phase B measures 16, 22 and 44 threads.

**3. int8 and bf16 compute on this CPU: not worth it.**

| model | prompt | fp32 ms/code | int8 ms/code | bf16 ms/code | bf16 first code |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M core | 256 | 7.3 | 15.7 | 8.3 | 227 ms (fp32 31) |
| 50M core | 256 | 14.5 | 25.7 | 14.5 | 1.15 s (fp32 0.12) |
| 100M core | 256 | 25.3 | 42.7 | 24.5 | 2.09 s (fp32 0.28) |

Real int8 kernels (dynamic quantisation) are 1.5-2.5x slower and change greedy text, keeping 2-38% of the
codes and 3-4 of 6 Q&A answers. The research int8 result, "free at 5M and 50M", was weight rounding,
measured for quality, not int8 arithmetic. The Xeon has no int8 dot-product instructions (VNNI), so the
kernels convert back and forth. bf16 has no hardware support here either: the prompt pass is 7-10x
slower. Both stay GPU options.

**4. torch.compile on the CPU** fails: PyTorch's CPU compiler needs Microsoft's C++ compiler (`cl`),
which is not installed (MinGW g++ is not accepted). Not tested. Installing Visual Studio Build Tools
would allow it. The cache gives most of the gain a compiler would.

## Adopted in run_model.py (phase C, CPU findings)

| setting | default | what |
| --- | --- | --- |
| `CORE_DECODE` | `"cache"` | core chat with the KV cache. When the window is full, its newest half is processed again (one prompt pass every half window). `"recompute"` is the old path |
| `STREAM` | `True` | the core chat prints its reply as it is generated; the status line shows the time to the first code |
| `CPU_INT8` | `False` | int8 kernels for the core chat on the CPU (slower here, kept for CPUs with VNNI) |
| menu option 4 | | switches device (auto / cpu / cuda) and CPU threads while running |

Tested: cached and recomputed replies are identical, with the same seed and sampling, on the 50M and
100M cores. The streamed text equals the reply. Past the window the two paths slide differently, so the
texts part after that point (expected).

Phase B results (GPU) will be added here when the queue finishes.
