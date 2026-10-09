"""Encodings for the training ladder, behind one interface.

Every encoding turns bytes into a 1-D array of model tokens and back, and says how many
original bytes each token stands for, so every result can be stated in bits per original
byte whatever the encoding.

    Bytes       raw bytes, the baseline
    V0Dict      a fitted Compiler v0 dictionary (compiler_v0.py), encoded with the C++ tool
    HashCodes   no dictionary at all: each scanner unit gets a (group, member) code from a
                hash of its bytes, so encoding is a pure function; decoding uses a reverse
                map of the units seen in training
"""
from __future__ import annotations

import collections
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_v0 as cv  # noqa: E402

REPLACEMENT = "�".encode("utf-8")


class Encoding:
    name: str
    vocab: int          # flat vocabulary: tokens are 0..vocab-1
    groups: int = 0     # >0: tokens are coordinate codes g * members + m
    members: int = 0

    def encode(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Return (tokens int64, bytes-per-token int64)."""
        raise NotImplementedError

    def encode_file(self, path: str, work: Path) -> tuple[np.ndarray, np.ndarray]:
        return self.encode(Path(path).read_bytes())

    def decode(self, tokens) -> bytes:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "vocab": self.vocab, "groups": self.groups, "members": self.members}


class Bytes(Encoding):
    name, vocab = "bytes", 256

    def encode(self, data):
        tokens = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        return tokens, np.ones(len(tokens), dtype=np.int64)

    def decode(self, tokens):
        return np.asarray(list(tokens), dtype=np.int64).astype(np.uint8).tobytes()


class V0Dict(Encoding):
    def __init__(self, dict_path: str, native: str | None = None):
        self.path, self.native = str(dict_path), native
        self.d = cv.Dictionary.load(self.path)
        self.enc = cv.Encoder(self.d)
        self.name = Path(dict_path).stem
        self.vocab = self.d.vocab_size
        self.lens = np.array([len(b) for b in self.d.id_bytes], dtype=np.int64)

    def encode(self, data):
        tokens = np.array(self.enc.encode(data), dtype=np.int64)
        return tokens, self.lens[tokens]

    def encode_file(self, path, work):
        if not self.native:
            return super().encode_file(path, work)
        out = Path(work) / f"{self.name}__{Path(path).stem}.ids"
        if not out.exists():
            subprocess.run([self.native, "encode", self.path, str(path), str(out)], check=True, capture_output=True)
        raw = out.read_bytes()
        magic, version, width, dict_hash, count = struct.unpack_from("<4sBB2xQQ", raw)
        if magic != b"CV0I" or dict_hash != self.d.hash:
            raise ValueError(f"{out}: wrong ID file or dictionary")
        tokens = np.frombuffer(raw, dtype="<u2" if width == 2 else "<u4", offset=24).astype(np.int64)
        assert len(tokens) == count
        return tokens, self.lens[tokens]

    def decode(self, tokens):
        return self.enc.decode(int(t) for t in tokens)

    def describe(self):
        d = super().describe()
        d.update(dictionary=self.path, dictionary_hash=f"{self.d.hash:016x}", entries=len(self.d.entries),
                 phrases=sum(1 for k, _ in self.d.entries if k == "P"), case_flags=self.d.case_flags,
                 max_phrase_units=self.d.max_phrase_units)
        return d


class HashCodes(Encoding):
    """Dictionary-free coordinate codes.

    A scanner unit's code is (fnv1a64(unit) mod groups, (fnv1a64(unit) >> 32) mod members),
    packed as g * members + m. Encoding needs nothing but the hash. Decoding looks the code up
    in a reverse map built from the training text: a collision decodes to the more frequent
    unit and an unseen code to U+FFFD, and both are counted.
    """

    def __init__(self, groups: int = 4096, members: int = 4096):
        self.groups, self.members = int(groups), int(members)
        self.vocab = self.groups * self.members
        self.name = f"hash{self.groups}x{self.members}"
        self.reverse: dict[int, bytes] = {}
        self.unit_count: collections.Counter = collections.Counter()
        self._cache: dict[bytes, int] = {}
        self.collisions = {"distinct_units": 0, "codes_used": 0, "units_lost": 0, "unit_mass_lost": 0.0}

    def code(self, unit: bytes) -> int:
        c = self._cache.get(unit)
        if c is None:
            h = cv.fnv1a64(unit)
            c = (h % self.groups) * self.members + ((h >> 32) % self.members)
            self._cache[unit] = c
        return c

    def fit(self, texts) -> None:
        for data in texts:
            self.unit_count.update(cv.scan(data))
        by_code: dict[int, list] = collections.defaultdict(list)
        for unit, n in self.unit_count.items():
            by_code[self.code(unit)].append((n, unit))
        total = sum(self.unit_count.values())
        lost_units, lost_mass = 0, 0
        for c, items in by_code.items():
            items.sort(key=lambda t: (-t[0], t[1]))
            self.reverse[c] = items[0][1]
            lost_units += len(items) - 1
            lost_mass += sum(n for n, _ in items[1:])
        self.collisions = {"distinct_units": len(self.unit_count), "codes_used": len(by_code),
                           "units_lost": lost_units, "unit_mass_lost": lost_mass / max(1, total)}

    def encode(self, data):
        units = list(cv.scan(data))
        tokens = np.fromiter((self.code(u) for u in units), dtype=np.int64, count=len(units))
        lens = np.fromiter((len(u) for u in units), dtype=np.int64, count=len(units))
        return tokens, lens

    def decode(self, tokens):
        return b"".join(self.reverse.get(int(t), REPLACEMENT) for t in tokens)

    def fidelity(self, data: bytes) -> dict:
        """Share of bytes that decode back exactly, unit by unit."""
        exact = total = unseen = 0
        for u in cv.scan(data):
            total += len(u)
            back = self.reverse.get(self.code(u))
            if back is None:
                unseen += len(u)
            elif back == u:
                exact += len(u)
        return {"bytes": total, "exact_share": exact / max(1, total), "unseen_share": unseen / max(1, total),
                "collided_share": (total - exact - unseen) / max(1, total)}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"groups": self.groups, "members": self.members,
                                          "reverse": {str(c): cv._escape(u) for c, u in self.reverse.items()},
                                          "collisions": self.collisions}))

    @classmethod
    def load(cls, path: str) -> "HashCodes":
        j = json.loads(Path(path).read_text())
        h = cls(j["groups"], j["members"])
        h.reverse = {int(c): cv._unescape(u) for c, u in j["reverse"].items()}
        h.collisions = j["collisions"]
        return h

    def describe(self):
        d = super().describe()
        d.update(self.collisions)
        return d
