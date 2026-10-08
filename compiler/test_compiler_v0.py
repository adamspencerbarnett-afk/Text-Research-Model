"""Contract tests for the Compiler v0 encoder (reference and native)."""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compiler_v0 as cv  # noqa: E402

WORDS = ("the of and to a in that it was he his for as with on be at by had not but from "
         "this which or have an they were all one their there been we would when who will "
         "more if no out so said what up its about into than them can only other new some "
         "could time these two may then do first any my now such like our over man me even "
         "most made after also did many before must through back years where much your way "
         "well down should because each just those people how too little state good very "
         "make world still own see men work long get here between both life being under never "
         "day same another know while last might us great old year off come since against go "
         "came right used take three holmes watson baker street london evening morning").split()


SYLLABLES = ("ka ri to me sa lo ne vi da ru pe so mi ta ko li ba de fu gi "
             "an er in on ar el or us en is at ul").split()
_lexicon_rng = random.Random(1234)
LEXICON = list(WORDS) + sorted({"".join(_lexicon_rng.choice(SYLLABLES) for _ in range(_lexicon_rng.randint(2, 4)))
                                for _ in range(4000)})
ZIPF = [1.0 / (rank + 1) for rank in range(len(LEXICON))]


def synthetic_text(seed: int, sentences: int = 400) -> bytes:
    """Deterministic English-like text with capitals, ALL-CAPS, numbers and punctuation."""
    rng = random.Random(seed)
    out = []
    for s in range(sentences):
        words = rng.choices(LEXICON, weights=ZIPF, k=rng.randint(4, 14))
        words[0] = words[0].capitalize()
        if rng.random() < 0.1:
            words[rng.randrange(len(words))] = rng.choice(WORDS).upper()
        if rng.random() < 0.2:
            words.insert(rng.randrange(len(words)), str(rng.randint(0, 99999)))
        sentence = " ".join(words) + rng.choice([".", ".", "?", "!", ";"])
        out.append(sentence + ("\n\n" if s % 7 == 6 else " "))
    return "".join(out).encode()


TRAIN = [synthetic_text(seed) for seed in range(4)]


def fitted(vocab=900, phrases=0, case_flags=True, units=3):
    d, _ = cv.fit(lambda: iter(TRAIN), vocab, phrases, units, case_flags)
    return d


ADVERSARIAL = [
    b"",
    b"a",
    b" ",
    b"   ",
    b"\n",
    b"I",
    b" I am",
    b"A AB Ab aB ABc abC",
    b"THE The the tHe",
    b"  leading and trailing  ",
    b"one\r\ntwo\r\n",
    b"tabs\tand\t\tmore\ttabs",
    b"x" * 100 + b" " + b"Y" * 70,
    b"caf\xc3\xa9 na\xc3\xafve \xe2\x80\x9cquoted\xe2\x80\x9d",
    b"12345 67.89 -0.5e10",
    bytes(range(256)),
    b"\xff\xfe\x00\x01 The END",
    b"Holmes said, \"Watson!\" -- and LONDON slept.\n\n\n\n    Indented.",
]


class ScannerTests(unittest.TestCase):
    def test_units_follow_the_rules(self):
        cases = {
            b"The cat sat.": [b"The", b" cat", b" sat", b"."],
            b"a  b": [b"a", b" ", b" b"],
            b"a   .": [b"a", b"   ", b"."],
            b"x\n\n\ty": [b"x", b"\n\n", b"\t", b"y"],
            b"abc123": [b"abc", b"1", b"2", b"3"],
            b" \xc3\xa9": [b" ", b"\xc3", b"\xa9"],
        }
        for text, expected in cases.items():
            self.assertEqual(list(cv.scan(text)), expected, text)

    def test_units_concatenate_to_input(self):
        rng = random.Random(7)
        for _ in range(200):
            data = bytes(rng.choice(b"aZ  \n\t.,9\xc3") for _ in range(rng.randint(0, 60)))
            self.assertEqual(b"".join(cv.scan(data)), data)

    def test_case_split(self):
        self.assertEqual(cv.case_split(b"The"), (cv.CAP_ID, b"the"))
        self.assertEqual(cv.case_split(b" I"), (cv.CAP_ID, b" i"))
        self.assertEqual(cv.case_split(b" NASA"), (cv.UPPER_ID, b" nasa"))
        self.assertEqual(cv.case_split(b"iPhone"), (0, b"iPhone"))
        self.assertEqual(cv.case_split(b"ABc"), (0, b"ABc"))
        self.assertEqual(cv.case_split(b"  "), (0, b"  "))
        self.assertEqual(cv.case_split(b"."), (0, b"."))


class RoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = {
            "flags+phrases": fitted(phrases=60, case_flags=True),
            "flags": fitted(case_flags=True),
            "plain+phrases": fitted(phrases=60, case_flags=False, units=4),
            "plain": fitted(case_flags=False),
        }

    def test_adversarial_inputs_round_trip(self):
        for name, d in self.variants.items():
            enc = cv.Encoder(d)
            for data in ADVERSARIAL + TRAIN[:1]:
                with self.subTest(variant=name, data=data[:30]):
                    self.assertEqual(enc.decode(enc.encode(data)), data)

    def test_random_inputs_round_trip(self):
        rng = random.Random(11)
        alphabet = b"aeiouAEIOUtThHsSnN  ..,\n\n\t'\"0123\xc3\xa9\xff"
        for name, d in self.variants.items():
            enc = cv.Encoder(d)
            for _ in range(300):
                data = bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
                self.assertEqual(enc.decode(enc.encode(data)), data, (name, data))

    def test_ids_are_within_the_vocabulary(self):
        for d in self.variants.values():
            ids = cv.Encoder(d).encode(TRAIN[2])
            self.assertTrue(all(0 <= x < d.vocab_size for x in ids))
            if not d.case_flags:
                self.assertNotIn(cv.CAP_ID, ids)
                self.assertNotIn(cv.UPPER_ID, ids)

    def test_dictionary_shortens_text(self):
        d = self.variants["flags+phrases"]
        held_out = synthetic_text(99)
        self.assertLess(len(cv.Encoder(d).encode(held_out)), 0.45 * len(held_out))

    def test_flags_cannot_be_decoded_without_case_flags(self):
        with self.assertRaises(ValueError):
            cv.Encoder(self.variants["plain"]).decode([cv.CAP_ID, ord("a")])
        with self.assertRaises(ValueError):
            cv.Encoder(self.variants["plain"]).decode([10 ** 6])


class StatelessTests(unittest.TestCase):
    def test_encoding_splits_at_unit_boundaries_without_phrases(self):
        enc = cv.Encoder(fitted(case_flags=True))
        rng = random.Random(3)
        data = synthetic_text(42, 60)
        units = list(cv.scan(data))
        for _ in range(50):
            cut = rng.randrange(len(units) + 1)
            a, b = b"".join(units[:cut]), b"".join(units[cut:])
            self.assertEqual(enc.encode(a + b), enc.encode(a) + enc.encode(b))

    def test_same_word_same_ids_in_any_context(self):
        enc = cv.Encoder(fitted(case_flags=True))
        word = enc.encode(b" watson")
        for context in (b"Holmes and%s said", b"\n\n%s", b"(%s)", b"%s?" ):
            ids = enc.encode(context.replace(b"%s", b" watson"))
            joined = ",".join(map(str, ids))
            self.assertIn(",".join(map(str, word)), joined)

    def test_capitalised_word_reuses_lowercase_ids(self):
        enc = cv.Encoder(fitted(case_flags=True))
        self.assertEqual(enc.encode(b" Street"), [cv.CAP_ID] + enc.encode(b" street"))
        self.assertEqual(enc.encode(b" STREET"), [cv.UPPER_ID] + enc.encode(b" street"))


class DictionaryTests(unittest.TestCase):
    def test_serialization_is_canonical_and_hash_is_stable(self):
        d = cv.Dictionary([("S", b" the"), ("C", b"ing"), ("P", b" of the")], True, 2)
        raw = d.serialize()
        self.assertEqual(raw, b"compiler-v0 dictionary\nversion 1\ncase_flags 1\nmax_phrase_units 2\n"
                              b"entries 3\nS\t the\nC\ting\nP\t of the\n")
        self.assertEqual(d.hash, 0xBB9493B204346E78)
        self.assertEqual(cv.Dictionary.parse(raw).entries, d.entries)
        self.assertEqual([d.tables["S"][b" the"], d.tables["C"][b"ing"], d.tables["P"][b" of the"]], [258, 259, 260])

    def test_escaping_round_trips_every_byte(self):
        value = bytes(range(256))
        d = cv.Dictionary([("P", value)])
        self.assertEqual(cv.Dictionary.parse(d.serialize()).entries[0][1], value)

    def test_invalid_dictionaries_are_rejected(self):
        with self.assertRaises(ValueError):
            cv.Dictionary([("S", b"ab"), ("S", b"ab")])
        with self.assertRaises(ValueError):
            cv.Dictionary([("S", b"a")])
        with self.assertRaises(ValueError):
            cv.Dictionary([("S", b"x" * (cv.MAX_ENTRY_LEN + 1))])
        with self.assertRaises(ValueError):
            cv.Dictionary([("Q", b"ab")])
        good = cv.Dictionary([("S", b"ab")]).serialize()
        with self.assertRaises(ValueError):
            cv.Dictionary.parse(good.replace(b"entries 1", b"entries 2"))
        with self.assertRaises(ValueError):
            cv.Dictionary.parse(good + b"\n")

    def test_fitting_is_deterministic_and_sized(self):
        a = fitted(vocab=700, phrases=40)
        b = fitted(vocab=700, phrases=40)
        self.assertEqual(a.serialize(), b.serialize())
        self.assertEqual(a.vocab_size, 700)
        self.assertEqual(sum(1 for k, _ in a.entries if k == "P"), 40)
        self.assertTrue(any(k == "C" for k, _ in a.entries))


class IdFileTests(unittest.TestCase):
    def test_write_read_and_dictionary_check(self):
        d = fitted()
        ids = cv.Encoder(d).encode(TRAIN[1])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.ids")
            cv.write_ids(path, ids, d)
            self.assertEqual(os.path.getsize(path), cv.ID_HEADER.size + 2 * len(ids))
            self.assertEqual(cv.read_ids(path, d), ids)
            with self.assertRaises(ValueError):
                cv.read_ids(path, fitted(vocab=800))

    def test_large_vocabularies_use_four_byte_ids(self):
        letters = "abcdefghijklmnopqrstuvwxyz"
        entries = [("S", (a + b + c + e).encode()) for a in letters for b in letters for c in letters for e in "abcd"]
        d = cv.Dictionary(entries[:70000])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.ids")
            ids = cv.Encoder(d).encode(b"zzzd aaaa")
            cv.write_ids(path, ids, d)
            self.assertEqual(os.path.getsize(path), cv.ID_HEADER.size + 4 * len(ids))
            self.assertEqual(cv.read_ids(path, d), ids)


def find_compiler():
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@unittest.skipUnless(find_compiler(), "no C++ compiler available")
class NativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        exe = os.path.join(cls.tmp, "cv0.exe" if os.name == "nt" else "cv0")
        subprocess.run([find_compiler(), "-O2", "-std=c++17", "-o", exe, str(HERE / "native" / "cv0.cpp")],
                       check=True, capture_output=True)
        cls.exe = exe

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_native(self, *args):
        return subprocess.run([self.exe, *args], check=True, capture_output=True)

    def test_native_matches_reference_byte_for_byte(self):
        rng = random.Random(5)
        alphabet = b"aeiouAEIOUtThHsSnN  ..,\n\n\t'\"0123\xc3\xa9\xff"
        inputs = ADVERSARIAL + [synthetic_text(77), bytes(rng.choice(alphabet) for _ in range(5000))]
        variants = [fitted(phrases=60), fitted(case_flags=False), fitted(phrases=80, case_flags=False, units=4)]
        for vi, d in enumerate(variants):
            dpath = os.path.join(self.tmp, f"d{vi}.cv0d")
            d.save(dpath)
            enc = cv.Encoder(d)
            for ii, data in enumerate(inputs):
                src, ref, out, back = (os.path.join(self.tmp, f"{n}{vi}_{ii}") for n in ("src", "ref", "out", "back"))
                Path(src).write_bytes(data)
                cv.write_ids(ref, enc.encode(data), d)
                self.run_native("encode", dpath, src, out)
                self.assertEqual(Path(out).read_bytes(), Path(ref).read_bytes(), (vi, data[:40]))
                self.run_native("decode", dpath, ref, back)
                self.assertEqual(Path(back).read_bytes(), data)

    def test_native_rejects_other_dictionaries(self):
        a, b = fitted(), fitted(vocab=800)
        pa, pb = os.path.join(self.tmp, "a.cv0d"), os.path.join(self.tmp, "b.cv0d")
        a.save(pa)
        b.save(pb)
        ids = os.path.join(self.tmp, "a.ids")
        cv.write_ids(ids, cv.Encoder(a).encode(TRAIN[0]), a)
        result = subprocess.run([self.exe, "decode", pb, ids, os.path.join(self.tmp, "x")], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
