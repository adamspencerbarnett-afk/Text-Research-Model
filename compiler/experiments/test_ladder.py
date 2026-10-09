"""Contract tests for the ladder encodings and the training harness (small and fast)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import compiler_v0 as cv  # noqa: E402
import ladder_encodings as LE  # noqa: E402

TEXT = (b"The cat sat on the mat. The dog sat on the log!\nWe went to the park, and the park was green.\n"
        b"Numbers 123 and 4,567 stay digits; UTF-8 \xe2\x80\x9cquotes\xe2\x80\x9d survive.\n") * 20
OTHER = b"We went to the zoo. The zebra sat on the mat?\n"


class BytesTests(unittest.TestCase):
    def test_round_trip_and_lengths(self):
        e = LE.Bytes()
        t, lens = e.encode(TEXT)
        self.assertEqual(e.decode(t), TEXT)
        self.assertEqual(int(lens.sum()), len(TEXT))
        self.assertEqual(e.vocab, 256)


class V0DictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d, _ = cv.fit(lambda: [TEXT], 400, phrase_entries=16, max_phrase_units=3, case_flags=False)
        self.path = os.path.join(self.tmp.name, "tiny.cv0d")
        d.save(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_lengths_and_description(self):
        e = LE.V0Dict(self.path)
        for data in (TEXT, OTHER, bytes(range(256))):
            t, lens = e.encode(data)
            self.assertEqual(e.decode(t), data)
            self.assertEqual(int(lens.sum()), len(data))   # every token accounts for its own bytes
            self.assertTrue(((t >= 0) & (t < e.vocab)).all())
        info = e.describe()
        self.assertEqual(info["dictionary_hash"], f"{cv.Dictionary.load(self.path).hash:016x}")
        self.assertEqual(info["phrases"], 16)

    def test_encode_file_without_native_matches_encode(self):
        e = LE.V0Dict(self.path)
        p = os.path.join(self.tmp.name, "a.txt")
        Path(p).write_bytes(TEXT)
        t1, _ = e.encode_file(p, Path(self.tmp.name))
        t2, _ = e.encode(TEXT)
        self.assertTrue(np.array_equal(t1, t2))


class HashCodesTests(unittest.TestCase):
    def test_codes_are_a_pure_function_of_the_unit(self):
        a, b = LE.HashCodes(64, 128), LE.HashCodes(64, 128)
        for unit in (b" the", b"the", b"\n", b".", b" park"):
            self.assertEqual(a.code(unit), b.code(unit))
            self.assertTrue(0 <= a.code(unit) < 64 * 128)
        self.assertNotEqual(a.code(b" the"), a.code(b"the"))

    def test_fit_decode_fidelity_and_save_load(self):
        h = LE.HashCodes(4096, 4096)
        h.fit([TEXT])
        t, lens = h.encode(TEXT)
        self.assertEqual(h.decode(t), TEXT)             # no collisions among these few units
        self.assertEqual(int(lens.sum()), len(TEXT))
        f = h.fidelity(OTHER)
        self.assertEqual(f["bytes"], len(OTHER))
        self.assertGreater(f["unseen_share"], 0)          # " zoo", " zebra" were never seen
        self.assertAlmostEqual(f["exact_share"] + f["unseen_share"] + f["collided_share"], 1.0)
        self.assertIn(LE.REPLACEMENT, h.decode(h.encode(OTHER)[0]))
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "h.json")
            h.save(p)
            g = LE.HashCodes.load(p)
        self.assertEqual(g.reverse, h.reverse)
        self.assertEqual(g.describe()["distinct_units"], len(set(cv.scan(TEXT))))

    def test_collisions_are_resolved_by_frequency(self):
        h = LE.HashCodes(1, 1)      # every unit shares code 0
        h.fit([b"a a a b"])         # units: "a", " a", " a", " b" -> " a" is the most frequent
        self.assertEqual(h.reverse[0], b" a")
        self.assertEqual(h.collisions["units_lost"], 2)   # three distinct units, one kept
        self.assertAlmostEqual(h.collisions["unit_mass_lost"], 0.5)


class LadderConfigTests(unittest.TestCase):
    def test_configuration_names(self):
        import ladder1
        self.assertEqual(ladder1.parse_config("v0_8k_plain"), ("v0_8k_plain", 0, (2,)))
        self.assertEqual(ladder1.parse_config("v0_8k_table"), ("v0_8k_plain", 1 << 20, (2,)))
        self.assertEqual(ladder1.parse_config("v0_8k_table18"), ("v0_8k_plain", 1 << 18, (2,)))
        self.assertEqual(ladder1.parse_config("v0_8k_table20_tri"), ("v0_8k_plain", 1 << 20, (2, 3)))
        self.assertEqual(ladder1.parse_config("v0_8k_phr3"), ("v0_8k_phr3", 0, (2,)))
        self.assertEqual(ladder1.parse_config("bytes"), ("bytes", 0, (2,)))
        self.assertEqual(ladder1.parse_config("bytes_table20_tri"), ("bytes", 1 << 20, (2, 3)))
        self.assertEqual(ladder1.parse_config("hash4096x4096"), ("hash4096x4096", 0, (2,)))
        self.assertEqual(ladder1.parse_config("hash4096x4096_table20_tri"), ("hash4096x4096", 1 << 20, (2, 3)))
        self.assertEqual(ladder1.parse_config("v0_8k_table20_q4"), ("v0_8k_plain", 1 << 20, (2, 3, 4)))
        self.assertEqual(ladder1.parse_config_full("v0_8k_ng"), ("v0_8k_plain", 0, (2,), (2, 3)))
        self.assertEqual(ladder1.parse_config_full("v0_8k_table20_tri_ng"), ("v0_8k_plain", 1 << 20, (2, 3), (2, 3)))
        with self.assertRaises(KeyError):
            ladder1.parse_config("v0_99k_table")


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch not installed")
        import train_lm
        cls.tl = train_lm

    def test_total_bits_counts_every_token_but_the_first(self):
        import torch
        torch.manual_seed(0)
        e = LE.Bytes()
        model = self.tl.LM(e, width=16, layers=1, heads=2, ctx=8)
        for n in (3, 8, 9, 17, 50):
            bits, predicted = self.tl.total_bits(model, np.arange(n) % 256, batch=2)
            self.assertEqual(predicted, n - 1)
            self.assertGreater(bits, 0)

    def test_bigram_hash_stays_in_range(self):
        import torch
        rows = 1 << 10
        prev = torch.randint(0, 20_000_000, (1000,)); cur = torch.randint(0, 20_000_000, (1000,))
        k = self.tl.mix_bigram(prev, cur, rows)
        self.assertTrue(bool(((k >= 0) & (k < rows)).all()))

    def test_ngram_table_counts_and_lookup(self):
        import torch
        tokens = np.array([1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 3, 5], dtype=np.int64)
        t = self.tl.NgramTable(tokens, order=2, topk=2)
        row = int(np.searchsorted(t.keys, 2))              # context "2" is followed by 3,4,3,3
        self.assertEqual(list(t.ids[row]), [3, 4])
        self.assertEqual(list(t.counts[row]), [3.0, 1.0])
        self.assertEqual(float(t.totals[row]), 4.0)
        ids, counts, totals = t.features(np.array([[1, 2, 9]], dtype=np.int64))
        self.assertEqual(ids[0, 1].tolist(), [3, 4])        # after "2"
        self.assertEqual(ids[0, 2].tolist(), [-1, -1])      # "9" never seen
        self.assertEqual(float(totals[0, 1]), 4.0)
        self.assertEqual(float(totals[0, 2]), 0.0)
        t3 = self.tl.NgramTable(tokens, order=3, topk=4)
        ids3, counts3, _ = t3.features(np.array([[1, 2, 3]], dtype=np.int64))
        self.assertEqual(ids3[0, 2].tolist()[0], 1)          # "2 3" is followed by 1 twice, 5 once
        self.assertEqual(float(counts3[0, 2][0]), 2.0)
        self.assertEqual(ids3[0, 1].tolist(), [3, 4, -1, -1])  # "1 2" is followed by 3 three times, 4 once
        self.assertEqual(ids3[0, 0].tolist(), [-1] * 4)      # incomplete context at the window start

    def test_leave_one_out_removes_the_training_occurrence(self):
        import torch
        torch.manual_seed(0)
        tokens = np.array([7, 8, 9, 7, 8, 9, 7, 8, 1], dtype=np.int64)
        e = LE.Bytes()
        model = self.tl.LM(e, width=16, layers=1, heads=2, ctx=8, ngrams=[self.tl.NgramTable(tokens, 2, topk=4)])
        with torch.no_grad():
            model.mix_head.weight.zero_(); model.mix_head.bias.zero_(); model.mix_head.bias[1] = 20.0  # trust the table
        x = torch.tensor([[7, 8, 9, 7, 8, 9, 7, 8]]); y = torch.tensor([[8, 9, 7, 8, 9, 7, 8, 1]])
        model.train()
        loo = model.nll(x, y)[0]
        model.eval()
        full = model.nll(x, y)[0]
        # "8" is followed by 9 twice and 1 once: with the table trusted, in eval the last target (1 after 8)
        # gets 1/3; in training its own occurrence is removed and it gets 0 -> loss near the clamp
        self.assertAlmostEqual(float(full[-1]), -np.log(1 / 3), places=3)
        self.assertGreater(float(loo[-1]), 10.0)
        # "7" is always followed by 8 (3 times): eval gives probability 1, training 2/2 = 1 as well
        self.assertAlmostEqual(float(full[0]), 0.0, places=3)
        self.assertAlmostEqual(float(loo[0]), 0.0, places=3)

    def test_short_runs_for_every_model_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            held = work / "held.txt"; held.write_bytes(OTHER * 30)
            sets = {"heldout": [str(held)]}
            h = LE.HashCodes(256, 256); h.fit([TEXT])
            for enc, table, orders, ngram in ((LE.Bytes(), 0, (2,), ()), (h, 0, (2,), ()), (LE.Bytes(), 1 << 8, (2,), ()),
                                              (LE.Bytes(), 1 << 8, (2, 3), ()), (LE.Bytes(), 0, (2,), (2, 3))):
                tokens, lens = enc.encode(TEXT)
                r = self.tl.run(enc, tokens, lens, sets, work, budget_s=1.0, width=16, layers=1, heads=2, ctx=16,
                                batch=4, table_rows=table, eval_every_s=0.5, quick_bytes=64, gen_tokens=5,
                                log=lambda *_: None, table_orders=orders, ngram_orders=ngram)
                if ngram:
                    self.assertEqual([t["order"] for t in r["model"]["ngram_tables"]], [2, 3])
                if table:
                    self.assertEqual(r["model"]["params"]["table"], table * 16 * len(orders))
                self.assertGreater(r["training"]["steps"], 0)
                self.assertGreater(r["eval"]["heldout_overall"]["bits_per_byte"], 0)
                self.assertEqual(r["generation"]["tokens"], 5)
                json.dumps(r)   # result must be serialisable


if __name__ == "__main__":
    unittest.main()
