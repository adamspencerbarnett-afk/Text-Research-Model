"""Compiler v0: a stateless, single-pass text encoder for model-training research.

The encoder turns bytes into integer IDs using one fixed dictionary. It keeps no state
between units, so the same text always produces the same IDs, whatever file it is in or
wherever it appears. Every input, including arbitrary binary data, decodes back exactly.

ID layout
    0-255   single bytes: the fallback for anything the dictionary does not cover
    256     CAP flag: the next word is capitalised ("The" -> CAP + "the")
    257     UPPER flag: the next word is all capitals ("NASA" -> UPPER + "nasa")
    258+    dictionary entries, in dictionary-file order

Scanner units, chosen left to right (first rule that matches wins)
    1. a space followed by letters      " word"
    2. letters                          "word"
    3. a run of spaces that is followed by " word" (all but the space the word keeps)
    4. a run of spaces, newlines or tabs
    5. any other single byte             ".", "7", a UTF-8 byte, ...
Letters are ASCII A-Z and a-z. Digits are single-byte units, so numbers are spelled
digit by digit.

Encoding, unit by unit
    1. Case flag (if enabled): a Title-case or ALL-CAPS word is lowercased and preceded by
       CAP or UPPER. Mixed-case words ("iPhone") are left as they are.
    2. Phrase: the longest run of 2..max_phrase_units consecutive flag-free units whose
       bytes form a phrase entry becomes a single ID.
    3. Otherwise the unit is split greedily, longest match first: the first piece comes
       from the start table, later pieces from the continuation table, and a single byte
       is used when no entry matches.

This module is the reference implementation: it defines the format, fits dictionaries
and is the oracle for the native encoder in native/cv0.cpp, which must produce
byte-identical ID files.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import re
import struct
import sys
import time
from typing import Iterable, Iterator, Sequence

FORMAT_NAME = "compiler-v0 dictionary"
FORMAT_VERSION = 1
CAP_ID = 256
UPPER_ID = 257
FIRST_ENTRY_ID = 258
MAX_ENTRY_LEN = 32
KINDS = ("S", "C", "P")  # start piece (includes whole words), continuation piece, phrase
ID_MAGIC = b"CV0I"
ID_HEADER = struct.Struct("<4sBB2xQQ")  # magic, version, id width, pad, dict hash, count

UNIT = re.compile(rb" [A-Za-z]+|[A-Za-z]+| +(?= [A-Za-z])| +|\n+|\t+|[\x00-\xff]")


# --------------------------------------------------------------------------- scanner

def scan(data: bytes) -> Iterator[bytes]:
    """Yield the scanner units of ``data``; their concatenation is exactly ``data``."""
    for match in UNIT.finditer(data):
        yield match.group()


def case_split(unit: bytes) -> tuple[int, bytes]:
    """Return (flag, unit to encode) under the case-flag rule; flag 0 means no flag."""
    letters = unit[1:] if unit[:1] == b" " else unit
    if not letters or not letters.isalpha():
        return 0, unit
    if len(letters) >= 2 and letters.isupper():
        return UPPER_ID, unit.lower()
    if letters[:1].isupper() and (len(letters) == 1 or letters[1:].islower()):
        return CAP_ID, unit.lower()
    return 0, unit


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# --------------------------------------------------------------------------- dictionary

def _escape(value: bytes) -> str:
    out = []
    for b in value:
        if b == 0x5C:
            out.append("\\\\")
        elif 0x20 <= b <= 0x7E:
            out.append(chr(b))
        else:
            out.append(f"\\x{b:02x}")
    return "".join(out)


def _unescape(text: str) -> bytes:
    out, i = bytearray(), 0
    while i < len(text):
        c = text[i]
        if c != "\\":
            out.append(ord(c))
            i += 1
        elif text[i + 1:i + 2] == "\\":
            out.append(0x5C)
            i += 2
        elif text[i + 1:i + 2] == "x" and i + 4 <= len(text):
            out.append(int(text[i + 2:i + 4], 16))
            i += 4
        else:
            raise ValueError(f"bad escape in dictionary entry: {text!r}")
    return bytes(out)


class Dictionary:
    """A frozen ID table. Entries keep their position for the life of the dictionary."""

    def __init__(self, entries: Sequence[tuple[str, bytes]], case_flags: bool = True,
                 max_phrase_units: int = 1):
        if not 1 <= max_phrase_units <= 8:
            raise ValueError("max_phrase_units must be 1..8")
        self.entries = [(kind, bytes(value)) for kind, value in entries]
        self.case_flags = bool(case_flags)
        self.max_phrase_units = int(max_phrase_units)
        self.tables: dict[str, dict[bytes, int]] = {kind: {} for kind in KINDS}
        self.id_bytes: list[bytes] = [bytes([b]) for b in range(256)] + [b"", b""]
        for offset, (kind, value) in enumerate(self.entries):
            if kind not in KINDS:
                raise ValueError(f"unknown entry kind {kind!r}")
            if kind in "SC" and not 2 <= len(value) <= MAX_ENTRY_LEN:
                raise ValueError(f"piece length must be 2..{MAX_ENTRY_LEN}: {value!r}")
            if kind == "P" and len(value) < 2:
                raise ValueError(f"phrase too short: {value!r}")
            if value in self.tables[kind]:
                raise ValueError(f"duplicate {kind} entry {value!r}")
            self.tables[kind][value] = FIRST_ENTRY_ID + offset
            self.id_bytes.append(value)
        self.max_piece_len = max([len(v) for k, v in self.entries if k in "SC"], default=1)

    @property
    def vocab_size(self) -> int:
        return FIRST_ENTRY_ID + len(self.entries)

    def serialize(self) -> bytes:
        lines = [FORMAT_NAME, f"version {FORMAT_VERSION}", f"case_flags {int(self.case_flags)}",
                 f"max_phrase_units {self.max_phrase_units}", f"entries {len(self.entries)}"]
        lines += [f"{kind}\t{_escape(value)}" for kind, value in self.entries]
        return ("\n".join(lines) + "\n").encode("ascii")

    @property
    def hash(self) -> int:
        return fnv1a64(self.serialize())

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.serialize())

    @classmethod
    def parse(cls, raw: bytes) -> "Dictionary":
        lines = raw.decode("ascii").split("\n")
        if lines[-1] != "":
            raise ValueError("dictionary must end with a newline")
        lines.pop()
        if len(lines) < 5 or lines[0] != FORMAT_NAME:
            raise ValueError("not a compiler-v0 dictionary")
        header = dict(line.split(" ", 1) for line in lines[1:5])
        if int(header["version"]) != FORMAT_VERSION:
            raise ValueError("unsupported dictionary version")
        count = int(header["entries"])
        body = lines[5:]
        if len(body) != count:
            raise ValueError("entry count mismatch")
        entries = []
        for line in body:
            kind, _, text = line.partition("\t")
            entries.append((kind, _unescape(text)))
        d = cls(entries, bool(int(header["case_flags"])), int(header["max_phrase_units"]))
        if d.serialize() != raw:
            raise ValueError("dictionary is not in canonical form")
        return d

    @classmethod
    def load(cls, path: str) -> "Dictionary":
        with open(path, "rb") as f:
            return cls.parse(f.read())


# --------------------------------------------------------------------------- encoder

class Encoder:
    def __init__(self, dictionary: Dictionary, cache_size: int = 1 << 18):
        self.d = dictionary
        self.start = dictionary.tables["S"]
        self.cont = dictionary.tables["C"]
        self.phrase = dictionary.tables["P"]
        self.k = dictionary.max_phrase_units if self.phrase else 1
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._cache_size = cache_size

    def pieces(self, unit: bytes) -> tuple[int, ...]:
        """Greedy longest-match split of one (already case-normalised) unit."""
        hit = self._cache.get(unit)
        if hit is not None:
            return hit
        out, pos, n, table = [], 0, len(unit), self.start
        longest = self.d.max_piece_len
        while pos < n:
            length = min(longest, n - pos)
            found = None
            while length >= 2:
                found = table.get(unit[pos:pos + length])
                if found is not None:
                    break
                length -= 1
            if found is None:
                found, length = unit[pos], 1
            out.append(found)
            pos += length
            table = self.cont
        result = tuple(out)
        if len(self._cache) < self._cache_size:
            self._cache[unit] = result
        return result

    def normalised_units(self, data: bytes) -> list[tuple[int, bytes]]:
        units = scan(data)
        if self.d.case_flags:
            return [case_split(u) for u in units]
        return [(0, u) for u in units]

    def encode(self, data: bytes) -> list[int]:
        units = self.normalised_units(data)
        out: list[int] = []
        i, n, k_max = 0, len(units), self.k
        while i < n:
            flag, unit = units[i]
            if flag:
                out.append(flag)
            elif k_max > 1:
                matched = False
                for k in range(min(k_max, n - i), 1, -1):
                    if any(units[i + m][0] for m in range(1, k)):
                        continue
                    pid = self.phrase.get(b"".join(units[i + m][1] for m in range(k)))
                    if pid is not None:
                        out.append(pid)
                        i += k
                        matched = True
                        break
                if matched:
                    continue
            out.extend(self.pieces(unit))
            i += 1
        return out

    def decode(self, ids: Iterable[int]) -> bytes:
        table, out = self.d.id_bytes, bytearray()
        mode, seen_letter = 0, False
        for x in ids:
            if x == CAP_ID or x == UPPER_ID:
                if not self.d.case_flags:
                    raise ValueError("case flag in a stream encoded without case flags")
                mode, seen_letter = x, False
                continue
            if not 0 <= x < len(table):
                raise ValueError(f"ID {x} is outside the dictionary")
            piece = table[x]
            if not mode:
                out += piece
                continue
            for c in piece:
                if mode:
                    if 97 <= c <= 122 or 65 <= c <= 90:
                        if (mode == UPPER_ID or not seen_letter) and 97 <= c <= 122:
                            c -= 32
                        seen_letter = True
                    elif seen_letter or c != 32:
                        mode = 0
                out.append(c)
        return bytes(out)


# --------------------------------------------------------------------------- ID files

def write_ids(path: str, ids: Sequence[int], dictionary: Dictionary) -> None:
    width = 2 if dictionary.vocab_size <= 1 << 16 else 4
    with open(path, "wb") as f:
        f.write(ID_HEADER.pack(ID_MAGIC, FORMAT_VERSION, width, dictionary.hash, len(ids)))
        f.write(struct.pack(f"<{len(ids)}{'H' if width == 2 else 'I'}", *ids))


def read_ids(path: str, dictionary: Dictionary | None = None) -> list[int]:
    with open(path, "rb") as f:
        raw = f.read()
    magic, version, width, dict_hash, count = ID_HEADER.unpack_from(raw)
    if magic != ID_MAGIC or version != FORMAT_VERSION or width not in (2, 4):
        raise ValueError("not a compiler-v0 ID file")
    if dictionary is not None and dict_hash != dictionary.hash:
        raise ValueError("ID file was encoded with a different dictionary")
    if len(raw) != ID_HEADER.size + count * width:
        raise ValueError("ID file length does not match its header")
    return list(struct.unpack_from(f"<{count}{'H' if width == 2 else 'I'}", raw, ID_HEADER.size))


# --------------------------------------------------------------------------- fitting

def count_units(texts: Iterable[bytes], case_flags: bool) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for data in texts:
        for unit in scan(data):
            if len(unit) >= 2:
                if case_flags:
                    unit = case_split(unit)[1]
                counts[unit] += 1
    return counts


def learn_pieces(unit_counts: dict[bytes, int], budget: int, positional: bool = True,
                 min_count: int = 2, max_len: int = MAX_ENTRY_LEN,
                 progress=None) -> tuple[list[tuple[str, bytes]], list[tuple[bytes, bytes, bool]]]:
    """Byte-pair merges inside units until ``budget`` distinct entries exist.

    With ``positional`` the first symbol of a unit is tracked separately, giving start
    pieces (S) and continuation pieces (C). Without it this is ordinary BPE and every
    entry is reported as S; that variant is only used as a baseline.
    Returns (entries in creation order, merges in order).
    """
    words: list[list[bytes]] = []
    weights: list[int] = []
    for unit, count in sorted(unit_counts.items()):
        if len(unit) >= 2:
            words.append([unit[i:i + 1] for i in range(len(unit))])
            weights.append(count)

    def pairs(w: list[bytes]):
        for i in range(len(w) - 1):
            if len(w[i]) + len(w[i + 1]) <= max_len:
                yield (w[i], w[i + 1], positional and i == 0)

    pair_count: dict = collections.defaultdict(int)
    where: dict = collections.defaultdict(set)
    for idx, w in enumerate(words):
        for p in pairs(w):
            pair_count[p] += weights[idx]
            where[p].add(idx)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    entries: list[tuple[str, bytes]] = []
    seen: set = set()
    merges: list = []
    while len(entries) < budget and heap:
        neg, p = heapq.heappop(heap)
        current = pair_count.get(p, 0)
        if current != -neg:
            if current > 0:
                heapq.heappush(heap, (-current, p))
            continue
        if current < min_count:
            break
        a, b, at_start = p
        merged = a + b
        merges.append(p)
        key = ("S" if (at_start or not positional) else "C", merged)
        if key not in seen:
            seen.add(key)
            entries.append(key)
        touched = set()
        for idx in list(where[p]):
            w, wt = words[idx], weights[idx]
            for q in pairs(w):
                pair_count[q] -= wt
                where[q].discard(idx)
                touched.add(q)
            nw, i = [], 0
            while i < len(w):
                if (i + 1 < len(w) and w[i] == a and w[i + 1] == b
                        and (not positional or (i == 0) == at_start)):
                    nw.append(merged)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            words[idx] = nw
            for q in pairs(nw):
                pair_count[q] += wt
                where[q].add(idx)
                touched.add(q)
        for q in touched:
            c = pair_count[q]
            if c > 0:
                heapq.heappush(heap, (-c, q))
            else:
                pair_count.pop(q, None)
                where.pop(q, None)
        if progress and len(merges) % 2000 == 0:
            progress(f"  {len(merges):,} merges, {len(entries):,} entries")
    return entries, merges


def _phrase_candidates(texts_factory, encoder: Encoder, max_units: int, min_count: int,
                       keep: int) -> list[tuple[int, bytes]]:
    """Score runs of flag-free units that each encode to one ID; returns (saved, bytes)."""
    def runs():
        for data in texts_factory():
            run: list[bytes] = []
            for flag, unit in encoder.normalised_units(data):
                if flag == 0 and len(encoder.pieces(unit)) == 1:
                    run.append(unit)
                    continue
                if len(run) >= 2:
                    yield run
                run = []
            if len(run) >= 2:
                yield run

    scores: dict[bytes, int] = {}
    allowed: set = set()
    for k in range(2, max_units + 1):
        counts: collections.Counter = collections.Counter()
        for run in runs():
            for i in range(len(run) - k + 1):
                if k == 2 or tuple(run[i:i + k - 1]) in allowed:
                    counts[tuple(run[i:i + k])] += 1
        top = [(c, g) for g, c in counts.items() if c >= min_count]
        top.sort(key=lambda t: (-t[0] * (k - 1), b"".join(t[1])))
        top = top[:keep]
        allowed = {g for _, g in top}
        for c, g in top:
            scores[b"".join(g)] = c * (k - 1)
    ranked = sorted(scores.items(), key=lambda t: (-t[1], t[0]))
    return [(s, v) for v, s in ranked]


def fit(texts_factory, vocab_size: int, phrase_entries: int = 0, max_phrase_units: int = 3,
        case_flags: bool = True, min_count: int = 2, progress=None) -> tuple[Dictionary, list]:
    """Fit a dictionary. ``texts_factory()`` must return a fresh iterable of bytes each call."""
    budget = vocab_size - FIRST_ENTRY_ID - phrase_entries
    if budget < 0:
        raise ValueError("vocab_size too small for the requested phrase entries")
    say = progress or (lambda *_: None)
    t0 = time.time()
    counts = count_units(texts_factory(), case_flags)
    say(f"counted {sum(counts.values()):,} units, {len(counts):,} distinct ({time.time() - t0:.1f}s)")
    entries, merges = learn_pieces(counts, budget, positional=True, min_count=min_count, progress=say)
    say(f"learned {len(entries):,} pieces ({time.time() - t0:.1f}s)")
    if phrase_entries and max_phrase_units >= 2:
        base = Encoder(Dictionary(entries, case_flags, 1))
        cands = _phrase_candidates(texts_factory, base, max_phrase_units, min_count, phrase_entries * 2)
        entries = entries + [("P", v) for _, v in cands[:phrase_entries]]
        say(f"added {min(phrase_entries, len(cands)):,} phrases ({time.time() - t0:.1f}s)")
    return Dictionary(entries, case_flags, max_phrase_units if phrase_entries else 1), merges


# --------------------------------------------------------------------------- BPE baseline

class BPEBaseline:
    """Ordinary BPE (merge-order application) on the same scanner units, for comparison."""

    def __init__(self, merges: Sequence[tuple[bytes, bytes, bool]], case_flags: bool):
        self.rank = {(a, b): r for r, (a, b, _) in enumerate(merges)}
        self.case_flags = case_flags
        vocab = sorted({a + b for a, b, _ in merges})
        self.ids = {v: FIRST_ENTRY_ID + i for i, v in enumerate(vocab)}
        self._cache: dict[bytes, tuple[int, ...]] = {}

    def _unit(self, unit: bytes) -> tuple[int, ...]:
        hit = self._cache.get(unit)
        if hit is not None:
            return hit
        sym = [unit[i:i + 1] for i in range(len(unit))]
        while len(sym) > 1:
            best = min(((self.rank.get((sym[i], sym[i + 1]), 1 << 60), i) for i in range(len(sym) - 1)))
            if best[0] == 1 << 60:
                break
            a, b = sym[best[1]], sym[best[1] + 1]
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            sym = out
        result = tuple(self.ids[s] if len(s) > 1 else s[0] for s in sym)
        self._cache[unit] = result
        return result

    def encode(self, data: bytes) -> list[int]:
        out: list[int] = []
        for unit in scan(data):
            if self.case_flags:
                flag, unit = case_split(unit)
                if flag:
                    out.append(flag)
            out.extend(self._unit(unit))
        return out


# --------------------------------------------------------------------------- CLI

def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compiler v0 reference encoder")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="fit a dictionary on training files")
    f.add_argument("train", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--vocab", type=int, default=32768)
    f.add_argument("--phrases", type=int, default=0)
    f.add_argument("--max-phrase-units", type=int, default=3)
    f.add_argument("--no-case-flags", action="store_true")
    f.add_argument("--min-count", type=int, default=2)
    for name in ("encode", "decode"):
        p = sub.add_parser(name)
        p.add_argument("--dict", required=True)
        p.add_argument("source")
        p.add_argument("destination")
    s = sub.add_parser("stats", help="IDs per KB and round-trip check")
    s.add_argument("--dict", required=True)
    s.add_argument("files", nargs="+")
    a = ap.parse_args(argv)

    if a.command == "fit":
        d, _ = fit(lambda: (_read(p) for p in a.train), a.vocab, a.phrases, a.max_phrase_units,
                   not a.no_case_flags, a.min_count, progress=lambda m: print(m, file=sys.stderr))
        d.save(a.out)
        print(f"{a.out}: {d.vocab_size:,} IDs, hash {d.hash:016x}")
        return 0
    d = Dictionary.load(a.dict)
    enc = Encoder(d)
    if a.command == "encode":
        write_ids(a.destination, enc.encode(_read(a.source)), d)
    elif a.command == "decode":
        with open(a.destination, "wb") as out:
            out.write(enc.decode(read_ids(a.source, d)))
    else:
        for path in a.files:
            raw = _read(path)
            ids = enc.encode(raw)
            ok = enc.decode(ids) == raw
            print(f"{path}: {len(raw):,} bytes -> {len(ids):,} IDs "
                  f"({1000 * len(ids) / max(1, len(raw)):.1f} per KB), round trip {'exact' if ok else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
