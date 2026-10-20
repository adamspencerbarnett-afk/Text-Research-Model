"""The chat22 engine: one conversation with a chat22 model.

Each reply: tools (arithmetic, calendar) -> which turns fit the window -> search both memories (skipping turns already in
the window) -> build the window (window.build, the same function training uses) -> generate. The turns stay in the
attention cache between replies, so only the memory block and the message are read again (when the window is full,
the oldest turns are dropped a quarter of the window at a time, so the cache is rebuilt rarely, not every turn).

Answer-time rules (16 October decisions): no code ban (the old engine banned 186 codes absent from the Q&A text, which
blocked copying rare words); the repeat block (no 3-code phrase twice) does not apply while the reply copies text that
is in the window; EOS or a new "User"/"Assistant" line ends the reply."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import torch

from . import encoder as E
from . import tools as T
from . import window as W
from .memory import Memory
from .model import ChatLM
from .tables import load_tables

ROOT = Path(__file__).resolve().parents[1]
ABSTAIN = (b"don't have that", b"do not have that", b"don't know", b"not in my memory", b"haven't told me")


class Engine:
    def __init__(self, checkpoint: str, device: str = "cuda", dense: bool | None = None, gate=(0.15, 0.55), log=print, memory_opts: dict | None = None,
                 search: str = "v1", judge_path: str | None = None, facts_path: str | None = None):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = cfg = ck["config"]
        self.dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.enc = E.load(ROOT / cfg["dict"].replace("\\", "/"))   # saved on Windows; also opens on Linux and macOS
        self.sep = self.enc.encode(E.SEP_TEXT)[0]
        tables = load_tables(Path(checkpoint).with_suffix(".tables.npz"))
        self.model = ChatLM(cfg["vocab"], cfg["width"], cfg["layers"], cfg["heads"], cfg["ctx"], tables, cfg.get("rope_scale", 1.0))
        self.model.load(ck["state_dict"]); self.model.to(self.dev).eval()
        use_dense = cfg.get("dense", 0.0) if dense is None else (cfg.get("dense", 2.0) if dense else 0.0)
        dk = None
        if use_dense:
            import dense_keys                          # MiniLM: the outside reference key, labelled in results
            dk = dense_keys.DenseKeys(str(self.dev), weight=use_dense)
        opts = dict(cfg.get("memory_opts") or {})
        opts.update(memory_opts or {})                         # settings given here override those saved with the model
        if "gate" in opts:
            gate = tuple(opts.pop("gate"))
        if opts.get("gate_facts") is not None:
            opts["gate_facts"] = tuple(opts["gate_facts"])
        self.memory = Memory(dk, gate, **opts)
        fp = facts_path or cfg.get("facts")                   # facts_path: another permanent store (e.g. real facts only)
        if fp and fp.endswith(".jsonl"):                       # v2.3: the uploaded-facts pool (Q&A, stored facts, document pieces)
            import json as _json
            rows = [_json.loads(l) for l in open(ROOT / fp, encoding="utf-8")]
            vecs = None
            vp = ROOT / (fp[:-len(".jsonl")] + "_minilm.npy")    # meaning vectors of the keys, computed once and kept
            if dk is not None and vp.exists() and len(np.load(vp, mmap_mode="r")) == len(rows):
                vecs = torch.from_numpy(np.load(vp)).to(dk.dev)
            elif dk is not None:
                vecs = dk.embed([r["key"].encode("utf-8", "replace") for r in rows]); np.save(vp, vecs.cpu().numpy())
            self.memory.add_facts([(r["q"].encode("utf-8", "replace"), r["a"].encode("utf-8", "replace"), r["key"].encode("utf-8", "replace")) for r in rows], vecs)
        else:
            facts = E.parse_exchanges((ROOT / fp).read_bytes()) if fp else []
            self.memory.add_facts([(q, a) for q, a in facts if a.strip()])
        self.ctx, self.mem_codes, self.reply_room = cfg["ctx"], cfg["mem_codes"], cfg["reply_room"]
        # search engine v2 (16 Oct, docs/SEARCH_ENGINE_V2.md): fact table + fast lane + decide (+ the judge if given)
        self.search = None
        if search == "v2":
            from .search_v2 import JudgeScorer, SearchV2
            judge = JudgeScorer(ROOT / judge_path, str(self.dev)) if judge_path else None
            self.search = SearchV2(self.memory, judge)
        self.history: list[tuple[bytes, bytes]] = []
        self.first_kept = 0                               # turns before this index are out of the window
        self._pc = None                                   # (codes, cache, states) of the last reply
        self.reuse = True
        log(f"chat22 engine: {Path(checkpoint).name}, {self.model.param_counts()['total']:,} parameters, window {self.ctx}, "
            f"{len(self.memory.facts.items):,} stored facts, dense key {'MiniLM x%.1f' % use_dense if dk else 'off'}")

    # ------------------------------------------------------------------ conversation
    def remember(self, user: bytes, reply: bytes) -> None:
        self.history.append((user, reply))
        (self.search or self.memory).add_turn(user, reply, len(self.history))

    def forget(self) -> None:
        self.history, self.first_kept, self._pc = [], 0, None
        (self.search or self.memory).forget_session()

    def undo_last(self) -> None:
        """Take back the last remembered turn (history and its session entries)."""
        if self.history:
            (self.search or self.memory).drop_turn(len(self.history)); self.history.pop()
            self.first_kept = min(self.first_kept, len(self.history)); self._pc = None

    def add_fact(self, q: bytes, a: bytes) -> None:
        (self.search or self.memory).add_facts([(q.strip(), a.strip())])

    # ------------------------------------------------------------------ one reply
    def _window(self, items, msg, max_codes):
        """Window over the turns from first_kept on; when they no longer fit, drop the oldest so that a quarter of
        the window is free again (the cache then changes once per quarter, not every turn)."""
        room = min(max_codes, self.reply_room)
        turns = self.history[self.first_kept:]
        codes, copy_end, pl, kept = W.build(self.enc, self.sep, items, turns, msg, None, self.ctx, self.mem_codes, room)
        if kept < len(turns):
            bare = W.build(self.enc, self.sep, items, [], msg, None, self.ctx, self.mem_codes, room)[0]
            free = self.ctx - len(bare) - room                      # room for turns in this window
            sizes = [len(self.enc.encode(E.render(*t))[0]) for t in turns]
            drop = 0
            while drop < len(turns) and sum(sizes[drop:]) > max(0, free - self.ctx // 4):
                drop += 1
            self.first_kept += drop
            turns = self.history[self.first_kept:]
            codes, copy_end, pl, kept = W.build(self.enc, self.sep, items, turns, msg, None, self.ctx, self.mem_codes, room)
        # the first turn (1-based) wholly in the window: the oldest kept turn may be only its end, so it stays searchable
        first_in = len(self.history) - kept + (2 if kept else 1)
        return codes, copy_end, kept, first_in

    @torch.no_grad()
    def reply(self, msg: bytes, max_codes: int = 300, use_memory: bool = True, use_tools: bool = True, verify: int = 0,
              stream=None) -> tuple[bytes, dict]:
        t0 = time.perf_counter()
        tool = T.run(msg) if use_tools else None
        _, _, kept0, first_in = self._window([], msg, max_codes)
        items, found = [], {}
        if use_memory:
            items, found = (self.search or self.memory).search(msg, [t[0] for t in self.history[-2:]], first_in)
        if tool:
            items = [W.Item(tool[0], tool[1], "fact")] + items
        prompt, copy_end, kept, first_in2 = self._window(items, msg, max_codes)
        if use_memory and first_in2 != first_in:               # the memory block pushed turns out: search again
            items, found = (self.search or self.memory).search(msg, [t[0] for t in self.history[-2:]], first_in2)
            if tool:
                items = [W.Item(tool[0], tool[1], "fact")] + items
            prompt, copy_end, kept, _ = self._window(items, msg, max_codes)
        evidence = b" ".join(it.q + b" " + it.a for it in items)
        if verify > 1 and items:
            outs = []
            g = torch.Generator().manual_seed(len(msg) + len(self.history))
            for j in range(verify):
                o = self._generate(prompt, copy_end, max_codes, None, 0.0 if j == 0 else 0.7, g)
                outs.append((_support(o[0], evidence, msg), o[2], o))
                if j == 0 and _abstains(o[0]):
                    break
            chosen = max(outs, key=lambda t: (round(t[0], 2), t[1]))[2]
            if stream is not None:
                for piece in re.findall(rb"\S+\s*", chosen[0]):
                    stream(piece)
        else:
            chosen = self._generate(prompt, copy_end, max_codes, stream)
        text, n_codes, mean_lp, stop, first = chosen
        return text, {"seconds": round(time.perf_counter() - t0, 3), "first_code_ms": round(1000 * first),
                      "window_codes": len(prompt), "window": self.ctx, "turns_in_window": kept, "turns_total": len(self.history),
                      "items": [it.text().decode("utf-8", "replace")[:160] for it in items], "found": found,
                      "reused_codes": self.last_reused, "codes": n_codes, "stopped": stop, "mean_logprob": round(mean_lp, 3)}

    # ------------------------------------------------------------------ generation
    def _generate(self, prompt, copy_end, max_codes, stream=None, temperature=0.0, gen=None):
        m, dev, enc = self.model, self.dev, self.enc
        t0 = time.perf_counter()
        seq = [int(c) for c in prompt]; start = len(seq)
        cache, keys, done = None, None, 0
        if self.reuse and self._pc is not None:                 # the part read last time is not read again
            old, oc, ok_ = self._pc
            n = min(len(old), start - 1)
            same = np.asarray(old[:n]) == np.asarray(seq[:n])
            p = int(n if same.all() else same.argmin())
            if p > 0:
                cache = [(k[:, :, :p], v[:, :, :p]) for k, v in oc]; keys = ok_[:, :p]; done = p
        self.last_reused = done
        x_dev = torch.tensor(seq, device=dev)
        copy_text = seq[:copy_end]
        first, stop, lp, shown = None, None, [], b""
        while len(seq) - start < max_codes and len(seq) < self.ctx:
            h, cache = m.hidden_cached(x_dev[done:][None], done, cache)
            keys = h if keys is None else torch.cat([keys, h], 1)
            done = len(seq)
            logp = m.next_logprobs(keys[:, -1], keys, x_dev[None], copy_end)[0].float()
            banned = _repeat_banned(seq[start:], copy_text, 3)
            if banned:
                logp[banned] = float("-inf")
            if temperature > 0:
                lg = logp / temperature
                kth = torch.topk(lg, 20).values[-1]
                nxt = int(torch.multinomial(torch.softmax(lg.masked_fill(lg < kth, float("-inf")), -1).cpu(), 1, generator=gen).item())
            else:
                nxt = int(logp.argmax())
            lp.append(float(logp[nxt]))
            if first is None:
                first = time.perf_counter() - t0
            if nxt == E.EOS:
                stop = "end"; break
            seq.append(nxt); x_dev = torch.cat([x_dev, torch.tensor([nxt], device=dev)])
            text = enc.decode(seq[start:])
            cut = min([q for q in (text.find(b"\nUser"), text.find(b"\nAssistant")) if q >= 0], default=-1)
            if cut >= 0:
                text = text[:cut]; stop = "turn"
            safe = text if stop else text[:max(0, len(text) - 12)]
            if stream is not None and len(safe) > len(shown):
                stream(safe[len(shown):]); shown = safe
            if stop:
                break
        full = enc.decode(seq[start:])
        cut = min([q for q in (full.find(b"\nUser"), full.find(b"\nAssistant")) if q >= 0], default=len(full))
        if stream is not None and cut > len(shown):
            stream(full[len(shown):cut])
        if self.reuse and cache is not None:
            self._pc = (seq[:done], cache, keys)
        return full[:cut].strip(), len(seq) - start, sum(lp) / max(1, len(lp)), stop or ("window" if len(seq) >= self.ctx else "length"), first or 0.0


def _repeat_banned(gen: list[int], window_codes: list[int], n: int) -> list[int]:
    """Codes that would repeat a 3-code phrase already in the reply, unless the reply is copying window text that
    continues that way (lists and quoted text may repeat)."""
    if len(gen) < n - 1:
        return []
    prefix = tuple(gen[len(gen) - n + 1:])
    banned = {gen[i + n - 1] for i in range(len(gen) - n + 1) if tuple(gen[i:i + n - 1]) == prefix}
    if not banned:
        return []
    ok = {window_codes[i + n - 1] for i in range(len(window_codes) - n + 1) if tuple(window_codes[i:i + n - 1]) == prefix}
    return sorted(banned - ok)


WORD = re.compile(rb"[a-z0-9][a-z0-9@.\-#']*")


def _words(b: bytes) -> set:
    return {w for w in WORD.findall(b.lower()) if len(w) > 2}


def _abstains(text: bytes) -> bool:
    t = text.lower().replace("’".encode(), b"'")
    return any(p in t for p in ABSTAIN)


def _support(text: bytes, evidence: bytes, msg: bytes) -> float:
    if _abstains(text):
        return 0.8
    aw = _words(text)
    return sum(w in _words(evidence + b" " + msg) for w in aw) / max(1, len(aw)) if aw else 0.0
