"""Regression checks for the semantic compiler and reasoning model.

Run: .venv\\Scripts\\python.exe -m unittest discover -s scripts -p test_semantic_v1.py -v
Fixtures are generated from development protocols and do not score the reserved final set.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import dataclasses
import io
from pathlib import Path
import random
import re
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


# Baseline task generation and independent label checks
class SemanticTaskTests(unittest.TestCase):
    """Validate the visible task independently of generator oracle rows."""

    @classmethod
    def setUpClass(cls):
        import semantic_tasks

        cls.tasks = semantic_tasks
        cls.training = semantic_tasks.make_split(120, 70191)
        seen = {example.text for example in cls.training}
        cls.splits = {}
        for index, shift in enumerate(semantic_tasks.SHIFTS):
            examples = semantic_tasks.make_split(120, 80191 + index, shift, seen)
            cls.splits[shift] = examples
            seen.update(example.text for example in examples)

    @staticmethod
    def visible_answer(text):
        """Test-only prose interpreter; never reads rows, task, or answer."""
        balances, edges = {}, {}
        def relation_pair(sentence, names):
            first_name = re.search(r"\b[A-Z][a-z]+\b", sentence).start()
            marker = re.search(r"\b(before|after)\b", sentence)
            if marker:
                reverse = ((marker.group() == "before") == (marker.start() < first_name))
            else:
                reverse = bool(re.search(r"\b(follow|follows|succeed|succeeds|postdate|postdates)\b", sentence))
            return tuple(reversed(names)) if reverse else tuple(names)

        def predicate_roles(sentence, names):
            giving = {"give", "gives", "hand", "hands", "pass", "passes", "transfer", "transfers", "remit", "remits"}
            receiving = {"receive", "receives", "obtain", "obtains", "acquire", "acquires"}
            roles = {"source": set(), "destination": set()}
            for clause in sentence.split(" and "):
                predicates = list(re.finditer(r"\b(?:" + "|".join(sorted(giving | receiving)) + r")\b", clause))
                for index, predicate in enumerate(predicates):
                    following = clause[predicate.end():predicates[index + 1].start() if index + 1 < len(predicates) else None]
                    postposed = re.search(r"^[^A-Z]*?(?<!who )\bis ([A-Z][a-z]+)\b", following)
                    preceding = re.findall(r"\b[A-Z][a-z]+\b", clause[:predicate.start()])
                    subject = postposed.group(1) if postposed else preceding[-1] if preceding else None
                    if subject:
                        roles["source" if predicate.group() in giving else "destination"].add(subject)
            if any(len(subjects) > 1 for subjects in roles.values()):
                raise AssertionError("Contradictory giver/receiver predicates")
            source = next(iter(roles["source"]), None)
            destination = next(iter(roles["destination"]), None)
            if source is not None and destination is not None:
                if source == destination:
                    raise AssertionError("Both transfer predicates bind the same subject")
                return source, destination
            if source is not None:
                return source, next(name for name in names if name != source)
            if destination is not None:
                return next(name for name in names if name != destination), destination
            return tuple(names)

        sentences = re.findall(r"[^.?]+[.?]", text)
        for raw_sentence in sentences:
            sentence = raw_sentence.strip()
            names = re.findall(r"\b[A-Z][a-z]+\b", sentence)
            if sentence.endswith("?"):
                if "coin" in sentence or "balance" in sentence:
                    return str(balances[names[0]]), 0
                start, target = relation_pair(sentence, names)
                frontier, seen, distance = {start}, set(), 0
                while frontier and target not in frontier:
                    seen.update(frontier)
                    frontier = {nxt for node in frontier for nxt in edges.get(node, ())
                                if nxt not in seen}
                    distance += 1
                reachable = target in frontier
                if " not " in sentence or "is it false that" in sentence:
                    reachable = not reachable
                return "yes" if reachable else "no", distance
            if any(phrase in sentence for phrase in ("starts with", "initially", "starting balance",
                                                    "starting number", "at the start")):
                balances[names[0]] = int(re.search(r"-?\d+", sentence).group())
            elif "coins" in sentence:
                explicit_source = re.search(r"\b(?:from|by) (?:the one who is )?([A-Z][a-z]+)\b", sentence)
                explicit_destination = re.search(r"\bto (?:the one who is )?([A-Z][a-z]+)\b", sentence)
                if explicit_source:
                    source = explicit_source.group(1)
                    destination = next(name for name in names if name != source)
                elif explicit_destination:
                    destination = explicit_destination.group(1)
                    source = next(name for name in names if name != destination)
                else:
                    source, destination = predicate_roles(sentence, names)
                if not re.search(r"\b(?:not|never|no movement|no transfer)\b", sentence):
                    amount = int(re.search(r"-?\d+", sentence).group())
                    balances[source] -= amount
                    balances[destination] += amount
            else:
                source, destination = relation_pair(sentence, names)
                edges.setdefault(source, set()).add(destination)
        raise AssertionError("Missing visible question")

    def test_semantic_labels_match_independent_visible_interpretation(self):
        for split, examples in {"train": self.training, **self.splits}.items():
            max_distance = 0
            for example in examples:
                with self.subTest(split=split, text=example.text):
                    answer, distance = self.visible_answer(example.text)
                    self.assertEqual(answer, example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), answer)
                    self.assertEqual(self.tasks.split_sentences(example.text), example.sentences)
                    self.assertEqual(len(example.rows), len(example.sentences))
                    self.assertEqual(len(example.rows[-1]), 5)
                    if example.task == "relations" and answer == "yes":
                        max_distance = max(max_distance, distance)
            self.assertEqual(max_distance, 3 if split == "composition" else 2)

    def test_semantic_splits_are_disjoint_reproducible_and_balanced(self):
        self.assertEqual(self.training, self.tasks.make_split(120, 70191))
        seen, normalized_seen = set(), set()
        for split, examples in {"train": self.training, **self.splits}.items():
            with self.subTest(split=split):
                texts = {example.text for example in examples}
                self.assertEqual(len(texts), 120)
                self.assertTrue(seen.isdisjoint(texts))
                seen.update(texts)
                normalized = {self.tasks.surface_sentences(text) for text in texts}
                self.assertEqual(len(normalized), len(examples))
                self.assertTrue(normalized_seen.isdisjoint(normalized))
                normalized_seen.update(normalized)
                self.assertEqual(sum(example.task == "accounting" for example in examples), 60)
                relations = [example.answer for example in examples if example.task == "relations"]
                self.assertEqual(relations.count("yes"), 30)
                self.assertEqual(relations.count("no"), 30)

    def test_semantic_shift_axes_follow_declared_surface_constraints(self):
        for split, examples in {"train": self.training, **self.splits}.items():
            for example in examples:
                with self.subTest(split=split, text=example.text):
                    names = set(self.tasks.surface_names(example.text))
                    allowed = self.tasks.HELDOUT_NAMES if split == "names" else self.tasks.TRAIN_NAMES
                    self.assertTrue(names.issubset(allowed))
                    if example.task == "accounting":
                        initial_numbers = [int(re.search(r"\d+", sentence).group())
                                           for sentence in example.sentences[:4]]
                        transfer_numbers = [int(re.search(r"\d+", sentence).group())
                                            for sentence in example.sentences[4:-1]]
                        initial_bounds = (10, 16) if split == "numbers" else (2, 9)
                        transfer_bounds = (5, 8) if split == "numbers" else (1, 4)
                        self.assertTrue(all(initial_bounds[0] <= number <= initial_bounds[1]
                                            for number in initial_numbers))
                        self.assertTrue(all(transfer_bounds[0] <= number <= transfer_bounds[1]
                                            for number in transfer_numbers))
                    if split == "length":
                        self.assertGreaterEqual(len(example.sentences), 9)
                    self.assertEqual(len(names), 4)

    def test_surface_name_canonicalization_preserves_visible_task(self):
        for example in self.training[:30]:
            names = self.tasks.surface_names(example.text)
            replacements = dict(zip(names, self.tasks.HELDOUT_NAMES))
            renamed = re.sub(r"\b[A-Z][a-z]+\b", lambda match: replacements[match.group()], example.text)
            with self.subTest(text=example.text):
                self.assertEqual(self.tasks.surface_sentences(renamed), self.tasks.surface_sentences(example.text))
                self.assertEqual(self.visible_answer(renamed)[0], example.answer)

    def test_exact_executor_distinguishes_roles_negation_and_values(self):
        prefix = [(self.tasks.INITIAL, 0, -1, 10, 1),
                  (self.tasks.INITIAL, 1, -1, 4, 1)]
        query = [(self.tasks.QUERY_BALANCE, 0, -1, 0, 1)]
        variants = [((self.tasks.TRANSFER, 0, 1, 3, 1), "7"),
                    ((self.tasks.TRANSFER, 1, 0, 3, 1), "13"),
                    ((self.tasks.TRANSFER, 0, 1, 3, 0), "10"),
                    ((self.tasks.TRANSFER, 0, 1, 4, 1), "6")]
        for row, expected in variants:
            with self.subTest(row=row):
                self.assertEqual(self.tasks.execute(prefix + [row] + query), expected)
        chain = [(self.tasks.BEFORE, index, index + 1, 0, 1) for index in range(3)]
        self.assertEqual(self.tasks.execute(chain + [(self.tasks.QUERY_BEFORE, 0, 3, 0, 1)]), "yes")
        self.assertEqual(self.tasks.execute(chain + [(self.tasks.QUERY_BEFORE, 3, 0, 0, 1)]), "no")


class SemanticStabilityTaskTests(unittest.TestCase):
    """Check the harder syntax, length, and combined-shift task families."""

    @classmethod
    def setUpClass(cls):
        import semantic_tasks

        cls.tasks = semantic_tasks
        cls.training = semantic_tasks.make_stability_split(80, 81021)
        excluded = {example.text for example in cls.training}
        cls.splits = {}
        for index, shift in enumerate(semantic_tasks.STABILITY_SHIFTS):
            examples = semantic_tasks.make_stability_split(80, 81031 + index, shift, excluded)
            cls.splits[shift] = examples
            excluded.update(example.text for example in examples)

    def test_stability_labels_match_independent_visible_prose(self):
        for split, examples in {"train": self.training, **self.splits}.items():
            maximum_distance = 0
            for example in examples:
                with self.subTest(split=split, text=example.text):
                    answer, distance = SemanticTaskTests.visible_answer(example.text)
                    self.assertEqual(answer, example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), answer)
                    if example.task == "relations" and answer == "yes":
                        maximum_distance = max(maximum_distance, distance)
            self.assertEqual(maximum_distance, 3 if split in ("composition", "combined") else 2)

    def test_stability_shift_axes_are_disjoint_and_exercise_the_claimed_changes(self):
        self.assertEqual(self.training, self.tasks.make_stability_split(80, 81021))
        seen = set()
        for split, examples in {"train": self.training, **self.splits}.items():
            with self.subTest(split=split):
                normalized = {self.tasks.surface_sentences(example.text) for example in examples}
                self.assertEqual(len(normalized), len(examples))
                self.assertTrue(seen.isdisjoint(normalized))
                seen.update(normalized)
                self.assertEqual(sum(example.task == "accounting" for example in examples), 40)
                labels = [example.answer for example in examples if example.task == "relations"]
                self.assertEqual(labels.count("yes"), labels.count("no"))
                initial_values, transfer_values = [], []
                for example in examples:
                    names = set(self.tasks.surface_names(example.text))
                    expected_names = (self.tasks.STABILITY_HELDOUT_NAMES
                                      if split in ("names", "combined") else self.tasks.TRAIN_NAMES)
                    self.assertEqual(len(names), 4)
                    self.assertTrue(names.issubset(expected_names))
                    self.assertLessEqual(len(example.sentences), 21)
                    if split in ("length", "combined"):
                        self.assertGreaterEqual(len(example.sentences), 9)
                    if example.task == "accounting":
                        initial_values.extend(int(re.search(r"-?\d+", s).group()) for s in example.sentences[:4])
                        transfer_values.extend(int(re.search(r"-?\d+", s).group()) for s in example.sentences[4:-1])
                if split in ("numbers", "combined"):
                    self.assertTrue(all(value not in range(2, 10) for value in initial_values))
                    self.assertTrue(all(value not in range(1, 5) for value in transfer_values))
                    self.assertTrue(any(value < 0 for value in initial_values))
                    self.assertIn(0, transfer_values)
                else:
                    self.assertTrue(all(value in range(2, 10) for value in initial_values))
                    self.assertTrue(all(value in range(1, 5) for value in transfer_values))


# Baseline model, checkpoint, and inference contracts
class SemanticModelTests(unittest.TestCase):
    """Protect padding, inference isolation, and checkpoint round trips."""

    @classmethod
    def setUpClass(cls):
        import semantic_model
        import semantic_tasks

        cls.model, cls.tasks = semantic_model, semantic_tasks
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        candidates = semantic_tasks.make_split(12, 92177, "length")
        cls.long_example = max(candidates, key=lambda example: len(example.text))
        cls.short_example = semantic_tasks.make_split(4, 92178)[0]
        cls.config = {"width": 16, "local_layers": 1, "layers": 1}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_models(self):
        torch.manual_seed(59107)
        models = {"parser": self.model.SemanticParser(16, 1),
                  "raw": self.model.RawReasoner(16, 1, 1),
                  "raw_aux": self.model.RawReasoner(16, 1, 1),
                  "oracle": self.model.Reasoner(16, 1),
                  "hybrid": self.model.Reasoner(16, 1, source=True)}
        for model in models.values():
            model.eval()
        return models

    def test_text_preparation_rejects_oracle_objects_and_preserves_literals(self):
        text = "Alice starts with 12 coins. Bob starts with 9 coins. Alice gives Bob 3 coins. how many coins does Alice have now?"
        with patch.object(self.model, "oracle_tensor", side_effect=AssertionError("oracle read")), \
             patch.object(self.model, "answer_targets", side_effect=AssertionError("answer read")):
            surface = self.model.prepare_texts([text])
        self.assertEqual(surface.values.tolist(), [[12.0, 9.0, 3.0, 0.0]])
        self.assertTrue(surface.sentence_mask.all().item())
        encoded = bytes(surface.ids[0, 0, 1:][surface.token_mask[0, 0, 1:]].tolist())
        self.assertEqual(encoded, b"e0 starts with 12 coins.")
        neg = self.model.prepare_texts(["Alice starts with -12 coins. what is Alice's balance now?"])
        self.assertEqual(neg.values.tolist(), [[-12.0, 0.0]])
        for invalid in ([], [self.short_example], [dataclasses.replace(self.short_example, rows=(), answer="999")]):
            with self.subTest(invalid=type(invalid[0]).__name__ if invalid else "empty"):
                with self.assertRaises(ValueError):
                    self.model.prepare_texts(invalid)
        with self.assertRaises(ValueError):
            self.model.prepare_texts(["Alice gives Bob 3 of 5 coins."])

    def test_parser_and_answer_models_ignore_extra_batch_padding(self):
        models = self.make_models()
        examples = [self.short_example, self.long_example]
        single = self.model.prepare_texts([examples[0].text])
        batch = self.model.prepare_texts([example.text for example in examples])
        single_rows, batch_rows = self.model.oracle_tensor(examples[:1]), self.model.oracle_tensor(examples)
        sentence_count = len(examples[0].sentences)
        self.assertGreater(batch.ids.shape[1], single.ids.shape[1])
        with torch.no_grad():
            one_output, batch_output = models["parser"](single), models["parser"](batch)
            for field in (*self.model.FIELD_SIZES, "features"):
                torch.testing.assert_close(one_output[field][0], batch_output[field][0, :sentence_count],
                                           atol=2e-6, rtol=2e-5)
            self.assertEqual(int(torch.count_nonzero(batch_output["features"][~batch.sentence_mask])), 0)
            torch.testing.assert_close(models["raw"](single)[0], models["raw"](batch)[0], atol=2e-6, rtol=2e-5)
            for mode in ("oracle", "hybrid"):
                one = models[mode](single_rows, single.sentence_mask, one_output["features"])
                many = models[mode](batch_rows, batch.sentence_mask, batch_output["features"])
                torch.testing.assert_close(one[0], many[0], atol=2e-6, rtol=2e-5)
            trimmed = batch.take(torch.tensor([0]))
            torch.testing.assert_close(models["raw"](trimmed), models["raw"](single), atol=2e-6, rtol=2e-5)
            rows = self.model.parsed_rows(batch_output, batch)
            self.assertTrue(torch.equal(rows[:, :, 3], batch.values))
            self.assertTrue(torch.equal(rows[:, :, 0][~batch.sentence_mask], torch.zeros_like(rows[:, :, 0][~batch.sentence_mask])))

    def test_zero_source_residual_starts_at_copied_structured_model(self):
        models = self.make_models()
        copied = models["hybrid"].load_state_dict(models["oracle"].state_dict(), strict=False)
        self.assertEqual(copied.missing_keys, ["source.weight"])
        self.assertEqual(copied.unexpected_keys, [])
        self.assertEqual(int(torch.count_nonzero(models["hybrid"].source.weight)), 0)
        surface = self.model.prepare_texts([self.short_example.text])
        rows = self.model.oracle_tensor([self.short_example])
        arbitrary_source = torch.randn((*surface.sentence_mask.shape, 16))
        with torch.no_grad():
            torch.testing.assert_close(models["oracle"](rows, surface.sentence_mask),
                                       models["hybrid"](rows, surface.sentence_mask, arbitrary_source),
                                       atol=0, rtol=0)
        with self.assertRaises(ValueError):
            models["hybrid"](rows, surface.sentence_mask)

    def test_neural_inference_never_calls_exact_executor(self):
        models = self.make_models()
        with patch.object(self.model, "execute", side_effect=RuntimeError("executor invoked")) as executor, \
             patch.object(self.tasks, "execute", side_effect=RuntimeError("task executor invoked")) as task_executor, \
             patch.object(self.model, "oracle_tensor", side_effect=AssertionError("oracle accessed")), \
             patch.object(self.model, "answer_targets", side_effect=AssertionError("labels accessed")):
            for mode in ("raw", "raw_aux", "parsed", "hybrid"):
                with self.subTest(mode=mode):
                    self.assertIsInstance(self.model.predict(models, self.short_example.text, mode), str)
            executor.assert_not_called()
            task_executor.assert_not_called()
            with self.assertRaises(ValueError):
                self.model.predict(models, self.short_example.text, "oracle")
            with self.assertRaisesRegex(RuntimeError, "executor invoked"):
                self.model.predict(models, self.short_example.text, "executor")

    def test_semantic_checkpoint_restores_all_components_and_predictions(self):
        models = self.make_models()
        metadata = {"config": self.config, "selected_mode": "hybrid", "fixture": "regression"}
        with tempfile.TemporaryDirectory(prefix="text-vector-semantic-test-") as directory:
            path = Path(directory) / "model.pt"
            self.model.save_checkpoint(path, models["parser"], models["raw"], models["oracle"],
                                       models["hybrid"], metadata, raw_aux=models["raw_aux"])
            restored, restored_metadata = self.model.load_checkpoint(path)
        self.assertEqual(metadata, restored_metadata)
        self.assertEqual(models.keys(), restored.keys())
        for name, model in models.items():
            self.assertFalse(restored[name].training)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, restored[name].state_dict()[key], atol=0, rtol=0)
        for mode in ("raw", "raw_aux", "parsed", "hybrid"):
            with self.subTest(mode=mode):
                self.assertEqual(self.model.predict(models, self.short_example.text, mode),
                                 self.model.predict(restored, self.short_example.text, mode))


class SemanticBoundModelTests(SemanticModelTests):
    """Exercise the entity-bound reasoner under permutations and masking."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.config = {**cls.config, "architecture": "bound"}

    def make_models(self):
        models = super().make_models()
        torch.manual_seed(59109)
        models["oracle"] = self.model.BoundReasoner(16, 1).eval()
        models["hybrid"] = self.model.BoundReasoner(16, 1, source=True).eval()
        return models

    def test_bound_branches_receive_finite_nonzero_gradients(self):
        model = self.make_models()["oracle"]
        examples = self.tasks.make_split(12, 192177)
        rows = self.model.oracle_tensor(examples)
        surface = self.model.prepare_texts([example.text for example in examples])
        with patch.object(self.model, "execute", side_effect=AssertionError("executor invoked")):
            predictions = model(rows, surface.sentence_mask)
            self.assertTrue(torch.isfinite(predictions).all().item())
            self.model.answer_loss(predictions, self.model.answer_targets(examples)).backward()
        branches = {"quantity_messages": model.quantity_messages.weight,
                    "quantity_readout": model.quantity_readout.weight,
                    "graph_seed": model.graph_seed,
                    "graph_self": model.graph_self.weight,
                    "graph_message": model.graph_message.weight,
                    "graph_readout": model.graph_readout.weight,
                    "residual_gates": model.residual_gates}
        for name, parameter in branches.items():
            with self.subTest(branch=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_bound_masked_rows_and_invalid_entity_ids_use_dummy_slot(self):
        model = self.make_models()["oracle"]
        rows = self.model.oracle_tensor([self.short_example])
        mask = torch.ones(rows.shape[:2], dtype=torch.bool)
        padded_rows = torch.cat((rows, torch.tensor([[[0, 99, -20, 500, 1],
                                                     [5, 88, -21, -500, 1]]], dtype=rows.dtype)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros((1, 2), dtype=torch.bool)), dim=1)
        with torch.no_grad():
            torch.testing.assert_close(model(rows, mask), model(padded_rows, padded_mask),
                                       atol=2e-6, rtol=2e-5)
            invalid, dummy = rows.clone(), rows.clone()
            invalid[0, 0, 1:3] = torch.tensor([99, -20])
            dummy[0, 0, 1:3] = -1
            torch.testing.assert_close(model(invalid, mask), model(dummy, mask), atol=0, rtol=0)

    def test_bound_entity_permutation_and_duplicate_edges_preserve_bound_outputs(self):
        model = self.make_models()["oracle"]
        self.assertEqual(int(torch.count_nonzero(model.residual_gates)), 0)
        examples = self.tasks.make_split(8, 292177)
        rows = self.model.oracle_tensor(examples)
        mask = self.model.prepare_texts([example.text for example in examples]).sentence_mask
        permutation = torch.tensor([3, 2, 0, 1])
        permuted = rows.clone()
        for field in (1, 2):
            real = rows[:, :, field] >= 0
            permuted[:, :, field][real] = permutation[rows[:, :, field][real].long()].float()
        chain = torch.tensor([[[3, 0, 1, 0, 1], [3, 1, 2, 0, 1], [3, 2, 3, 0, 1],
                               [5, 0, 3, 0, 1]]], dtype=torch.float32)
        repeated = torch.cat((chain[:, :-1], chain[:, 1:2], chain[:, -1:]), dim=1)
        with torch.no_grad():
            # Global type logits use ordinary learned ID embeddings; only the
            # explicitly bound branches promise this permutation property.
            torch.testing.assert_close(model(rows, mask)[:, 2:], model(permuted, mask)[:, 2:],
                                       atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(model(chain, torch.ones((1, 4), dtype=torch.bool))[:, 3],
                                       model(repeated, torch.ones((1, 5), dtype=torch.bool))[:, 3],
                                       atol=0, rtol=0)

    def test_bound_checkpoint_configuration_restores_bound_classes(self):
        models = self.make_models()
        metadata = {"config": self.config, "selected_mode": "parsed"}
        with tempfile.TemporaryDirectory(prefix="text-vector-bound-test-") as directory:
            path = Path(directory) / "model.pt"
            self.model.save_checkpoint(path, models["parser"], models["raw"], models["oracle"],
                                       models["hybrid"], metadata)
            restored, _ = self.model.load_checkpoint(path)
        self.assertIsInstance(restored["oracle"], self.model.BoundReasoner)
        self.assertIsInstance(restored["hybrid"], self.model.BoundReasoner)


# Stable reasoner and exact-query behavior
class SemanticStableModelTests(unittest.TestCase):
    """Protect the fixed-size reasoner, quantity fit, and query validation."""

    @classmethod
    def setUpClass(cls):
        import semantic_model
        import semantic_tasks

        cls.model, cls.tasks = semantic_model, semantic_tasks

    def make_models(self):
        torch.manual_seed(81061)
        models = {"parser": self.model.SemanticParser(16, 1, representation="typed"),
                  "raw": self.model.RawReasoner(16, 1, 1, representation="typed"),
                  "oracle": self.model.StableReasoner(16, 1)}
        for model in models.values():
            model.eval()
        return models

    @staticmethod
    def visible_targets(examples):
        answers = [SemanticTaskTests.visible_answer(example.text)[0] for example in examples]
        return torch.tensor([[int(answer in ("yes", "no")),
                              0 if answer in ("yes", "no") else int(answer),
                              int(answer == "yes")] for answer in answers], dtype=torch.float32)

    def test_typed_parser_is_invariant_to_quantity_magnitude_and_retains_exact_values(self):
        values = [(0, -12, 10), (99999, 0, -3), (-50, 111, 0)]
        texts = [f"Alice starts with {a} coins. Bob starts with {b} coins. "
                 f"Alice gives Bob {c} coins. how many coins does Bob have now?"
                 for a, b, c in values]
        surface = self.model.prepare_texts(texts, representation="typed")
        self.assertEqual(surface.representation, "typed")
        self.assertEqual(surface.values.tolist(), [[float(a), float(b), float(c), 0.0] for a, b, c in values])
        for index in range(1, len(texts)):
            torch.testing.assert_close(surface.ids[index], surface.ids[0], atol=0, rtol=0)
            torch.testing.assert_close(surface.token_mask[index], surface.token_mask[0], atol=0, rtol=0)
        content = surface.ids[surface.token_mask]
        self.assertIn(self.model.ENTITY_START, content.tolist())
        self.assertIn(self.model.ENTITY_START + 1, content.tolist())
        self.assertIn(self.model.QUANTITY, content.tolist())
        self.assertNotIn(self.model.QUANTITY, range(self.model.ENTITY_START, self.model.ENTITY_START + 4))
        parser = self.make_models()["parser"]
        with torch.no_grad():
            output = parser(surface)
        for field, tensor in output.items():
            for index in range(1, len(texts)):
                with self.subTest(field=field, quantity_set=index):
                    torch.testing.assert_close(tensor[index], tensor[0], atol=2e-6, rtol=2e-5)
        rows = self.model.parsed_rows(output, surface)
        torch.testing.assert_close(rows[:, :, 3], surface.values, atol=0, rtol=0)
        selected = surface.take(torch.tensor([2, 0]))
        self.assertEqual(selected.representation, "typed")
        torch.testing.assert_close(selected.values, surface.values[[2, 0]], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            parser(self.model.prepare_texts(texts))

    def test_typed_literal_storage_accepts_exact_boundaries_and_rejects_rounding(self):
        for value in (-(2 ** 24), 2 ** 24):
            with self.subTest(value=value):
                surface = self.model.prepare_texts(
                    [f"Alice starts with {value} coins. what is Alice's balance now?"],
                    representation="typed",
                )
                self.assertEqual(surface.values[0, 0].item(), value)
        for value in (-(2 ** 24 + 1), 2 ** 24 + 1):
            with self.subTest(value=value):
                text = f"Alice starts with {value} coins. what is Alice's balance now?"
                with self.assertRaises(ValueError):
                    self.model.prepare_texts([text], representation="typed")
                # The older byte checkpoint's accepted-input contract remains unchanged.
                self.assertEqual(self.model.prepare_texts([text]).representation, "bytes")

    def test_stable_complete_outputs_preserve_entity_permutation_fact_order_and_duplicate_edges(self):
        model = self.make_models()["oracle"]
        examples = self.tasks.make_stability_split(12, 81062)
        rows = self.model.oracle_tensor(examples)
        mask = self.model.prepare_texts([e.text for e in examples], representation="typed").sentence_mask
        permutation = torch.tensor([3, 0, 2, 1])
        permuted = rows.clone()
        for field in (1, 2):
            real = rows[:, :, field] >= 0
            permuted[:, :, field][real] = permutation[rows[:, :, field][real].long()].float()
        chain = torch.tensor([[[3, 0, 1, 0, 1], [3, 1, 2, 0, 1], [3, 2, 3, 0, 1],
                               [5, 0, 3, 0, 1]]], dtype=torch.float32)
        reordered = chain[:, [2, 0, 1, 3]]
        repeated = torch.cat((chain[:, :-1].repeat(1, 5, 1), chain[:, -1:]), dim=1)
        with torch.no_grad():
            torch.testing.assert_close(model(rows, mask), model(permuted, mask), atol=2e-6, rtol=2e-5)
            expected = model(chain, torch.ones(chain.shape[:2], dtype=torch.bool))
            for variant in (reordered, repeated):
                torch.testing.assert_close(
                    expected, model(variant, torch.ones(variant.shape[:2], dtype=torch.bool)),
                    atol=0, rtol=0,
                )
        original = "Alice is before Bob. Bob is before Clara. Clara is before David. is Alice before David?"
        renamed = "Yara is before Vera. Vera is before Sana. Sana is before Quinn. is Yara before Quinn?"
        a = self.model.prepare_texts([original], representation="typed")
        b = self.model.prepare_texts([renamed], representation="typed")
        torch.testing.assert_close(a.ids, b.ids, atol=0, rtol=0)
        with torch.no_grad():
            one, two = self.make_models()["parser"](a), self.make_models()["parser"](b)
        for field in one:
            torch.testing.assert_close(one[field], two[field], atol=0, rtol=0)

    def test_stable_query_routing_is_independent_of_length_and_invalid_queries_are_marked(self):
        model = self.make_models()["oracle"]
        facts = torch.tensor([[[3, 0, 1, 0, 1], [3, 1, 2, 0, 1], [3, 2, 3, 0, 1]]], dtype=torch.float32)
        for query, expected_kind in (([4, 0, -1, 0, 1], 0), ([5, 0, 3, 0, 1], 1)):
            for repeats in (1, 6):
                rows = torch.cat((facts.repeat(1, repeats, 1), torch.tensor([[query]], dtype=torch.float32)), dim=1)
                mask = torch.ones(rows.shape[:2], dtype=torch.bool)
                self.assertTrue(model.valid_queries(rows, mask).item())
                self.assertEqual(int(model(rows, mask)[0, :2].argmax()), expected_kind)
        invalid_programs = [
            facts,
            torch.cat((facts, torch.tensor([[[5, 0, 3, 0, 1], [5, 0, 3, 0, 1]]], dtype=torch.float32)), dim=1),
            torch.cat((facts, torch.tensor([[[5, 0, 99, 0, 1]]], dtype=torch.float32)), dim=1),
            torch.cat((facts, torch.tensor([[[5, 0, 0, 0, 1]]], dtype=torch.float32)), dim=1),
            torch.cat((facts, torch.tensor([[[5, 0, 3, 0, 0]]], dtype=torch.float32)), dim=1),
        ]
        for rows in invalid_programs:
            mask = torch.ones(rows.shape[:2], dtype=torch.bool)
            valid = model.valid_queries(rows, mask)
            self.assertFalse(valid.item())
            output = model(rows, mask)
            self.assertTrue(torch.isfinite(output).all().item())
            self.assertEqual(self.model.decode_answers(output, valid), ["<invalid>"])

    def test_stable_branches_have_finite_gradients_without_global_or_executor_paths(self):
        model = self.make_models()["oracle"]
        # Ensure ReLU paths are active so this fixture tests graph connectivity.
        with torch.no_grad():
            model.graph_seed.copy_(model.graph_seed.abs() + .1)
            model.graph_self.weight.copy_(torch.eye(8) * .5 + .05)
            model.graph_message.weight.copy_(torch.eye(8) * .5 + .05)
            model.graph_readout.weight.fill_(.2)
        examples = self.tasks.make_stability_split(12, 81063)
        rows = self.model.oracle_tensor(examples)
        surface = self.model.prepare_texts([e.text for e in examples], representation="typed")
        with patch.object(self.model, "execute", side_effect=AssertionError("executor invoked")):
            predictions = model(rows, surface.sentence_mask)
            self.model.answer_loss(predictions, self.model.answer_targets(examples)).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertFalse(hasattr(model, "core"))
        self.assertFalse(hasattr(model, "residual_gates"))
        with self.assertRaises(ValueError):
            self.model.StableReasoner(source=True)

    def test_stable_checkpoint_restores_typed_components_without_hybrid(self):
        models = self.make_models()
        config = {"width": 16, "local_layers": 1, "layers": 1,
                  "representation": "typed", "architecture": "stable"}
        metadata = {"config": config, "selected_mode": "parsed"}
        with tempfile.TemporaryDirectory(prefix="text-vector-stable-test-") as directory:
            path = Path(directory) / "model.pt"
            self.model.save_checkpoint(path, models["parser"], models["raw"], models["oracle"], None, metadata)
            restored, restored_metadata = self.model.load_checkpoint(path)
        self.assertEqual(restored_metadata, metadata)
        self.assertEqual(set(restored), set(models))
        self.assertIsInstance(restored["oracle"], self.model.StableReasoner)
        self.assertEqual(restored["parser"].representation, "typed")
        self.assertEqual(restored["raw"].representation, "typed")
        for name, model in models.items():
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored[name].state_dict()[key], atol=0, rtol=0)
        text = "Alice starts with -12 coins. what is Alice's balance now?"
        self.assertEqual(self.model.predict(models, text, "raw"), self.model.predict(restored, text, "raw"))

    def test_stable_text_inference_uses_declared_representation_and_rejects_missing_question(self):
        models = self.make_models()
        text = "Alice starts with 12 coins. what is Alice's balance now?"
        with patch.object(self.model, "prepare_texts", wraps=self.model.prepare_texts) as prepare:
            self.assertIsInstance(self.model.predict(models, text, "raw"), str)
            self.assertEqual(prepare.call_args.args[2], "typed")
        for invalid in ("Alice starts with 12 coins.",
                        "what is Alice's balance now? Alice starts with 12 coins.",
                        "what is Alice's balance now? how many coins does Alice have now?"):
            with self.subTest(text=invalid):
                for mode in ("raw", "parsed", "executor"):
                    with self.assertRaises(ValueError):
                        self.model.predict(models, invalid, mode)

    def test_stable_executor_evaluation_rejects_the_same_invalid_queries_as_inference(self):
        import semantic_experiment

        sentences = ("Alice starts with 12 coins.", "Bob starts with 3 coins.",
                     "what is Alice's balance now?")
        example = self.tasks.Example(
            " ".join(sentences), sentences,
            ((1, 0, -1, 12, 1), (1, 1, -1, 3, 1), (4, 0, -1, 0, 1)),
            "12", "accounting",
        )
        data = semantic_experiment.prepare([example], "cpu", "typed")
        data.predicted = data.rows.clone()
        validator = self.model.StableReasoner.valid_queries
        self.assertEqual(semantic_experiment.evaluate_executor(data, validator)["exact_accuracy"], 1.0)
        inactive, not_last, unsupported = data.rows.clone(), data.rows.clone(), data.rows.clone()
        inactive[0, -1, 4] = 0
        not_last[0] = not_last[0, [0, 2, 1]]
        unsupported[0, -1, 0] = 6
        models = self.make_models()
        for label, predicted in (("inactive", inactive), ("not_last", not_last), ("unsupported", unsupported)):
            with self.subTest(query=label):
                data.predicted = predicted
                # Legacy execution alone permits inactive/nonfinal queries, exposing the old mismatch.
                legacy = semantic_experiment.evaluate_executor(data)
                if label != "unsupported":
                    self.assertEqual(legacy["exact_accuracy"], 1.0)
                validated = semantic_experiment.evaluate_executor(data, validator)
                self.assertEqual(validated["exact_accuracy"], 0.0)
                self.assertEqual(validated["invalid_programs"], 1)
                with patch.object(self.model, "parsed_rows", return_value=predicted):
                    with self.assertRaises(ValueError):
                        self.model.predict(models, example.text, "executor")

    def test_stability_gates_require_each_task_and_each_seed_to_pass(self):
        import semantic_experiment

        trials = [{"seed": seed, "test": {
            shift: {"parsed": {"macro_exact_accuracy": 1.0, "per_task": {
                task: {"exact_accuracy": 0.0 if shift == "lexical_unseen" else 1.0}
                for task in ("accounting", "relations")}}}
            for shift in self.tasks.STABILITY_SHIFTS}}
            for seed in (81071, 81072, 81073)]
        self.assertTrue(semantic_experiment.stability_gates(trials)["modes"]["parsed"]["passed"])
        hidden_task_failure = copy.deepcopy(trials)
        for trial in hidden_task_failure:
            row = trial["test"]["length"]["parsed"]
            row["macro_exact_accuracy"] = .95
            row["per_task"]["relations"]["exact_accuracy"] = .90
        gate = semantic_experiment.stability_gates(hidden_task_failure)["modes"]["parsed"]
        self.assertFalse(gate["passed"])
        self.assertEqual({failure["task"] for failure in gate["failures"]}, {"relations"})
        one_bad_seed = copy.deepcopy(trials)
        one_bad_seed[0]["test"]["numbers"]["parsed"]["per_task"]["accounting"]["exact_accuracy"] = .94
        gate = semantic_experiment.stability_gates(one_bad_seed)["modes"]["parsed"]
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 1)
        self.assertEqual(gate["failures"][0]["seed"], 81071)

    def test_quantity_fit_learns_training_targets_and_preserves_graph_parameters(self):
        model = self.make_models()["oracle"]
        training = self.tasks.make_stability_split(120, 81081)
        heldout = self.tasks.make_stability_split(
            120, 81082, "numbers", exclude=(example.text for example in training),
        )
        rows = self.model.oracle_tensor(training)
        mask = self.model.prepare_texts([example.text for example in training], representation="typed").sentence_mask
        targets = self.visible_targets(training)
        graph_before = {name: value.clone() for name, value in model.state_dict().items()
                        if name.startswith("graph_")}
        with patch.object(self.model, "execute", side_effect=AssertionError("executor invoked")):
            metadata = model.fit_quantities(rows, mask, targets)
        self.assertEqual(metadata["examples"], int((targets[:, 0] == 0).sum()))
        test_rows = self.model.oracle_tensor(heldout)
        test_mask = self.model.prepare_texts([example.text for example in heldout], representation="typed").sentence_mask
        expected = self.visible_targets(heldout)
        accounting = expected[:, 0] == 0
        with torch.no_grad():
            first = model(test_rows, test_mask)
        self.assertEqual(torch.round(first[accounting, 2] * 16).tolist(), expected[accounting, 1].tolist())
        doubled = targets.clone()
        doubled[:, 1] *= 2
        model.fit_quantities(rows, mask, doubled)
        with torch.no_grad():
            second = model(test_rows, test_mask)
        self.assertEqual(torch.round(second[accounting, 2] * 16).tolist(), (expected[accounting, 1] * 2).tolist())
        self.assertTrue(torch.any(expected[accounting, 1] != 0).item())
        for name, value in graph_before.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        torch.testing.assert_close(first[:, 3], second[:, 3], atol=0, rtol=0)

    def test_quantity_fit_rejects_missing_accounting_and_nonfinite_inputs_without_mutation(self):
        model = self.make_models()["oracle"]
        examples = self.tasks.make_stability_split(12, 81083)
        rows = self.model.oracle_tensor(examples)
        mask = self.model.prepare_texts([example.text for example in examples], representation="typed").sentence_mask
        targets = self.visible_targets(examples)
        account_index = int(torch.nonzero(targets[:, 0] == 0)[0, 0])
        missing = targets.clone()
        missing[:, 0] = 1
        invalid_rows = rows.clone()
        invalid_rows[account_index, 0, 3] = float("nan")
        invalid_targets = targets.clone()
        invalid_targets[account_index, 1] = float("inf")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        for label, candidate_rows, candidate_targets in (
            ("no_accounting", rows, missing),
            ("nonfinite_rows", invalid_rows, targets),
            ("nonfinite_targets", rows, invalid_targets),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    model.fit_quantities(candidate_rows, mask, candidate_targets)
                for name, value in before.items():
                    torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)


# Interpretation datasets: syntax, counterfactuals, and split isolation
class SemanticInterpretationTaskTests(unittest.TestCase):
    """Verify that learned-parser datasets preserve visible meaning and isolation."""

    @classmethod
    def setUpClass(cls):
        import semantic_tasks

        cls.tasks = semantic_tasks
        cls.training = semantic_tasks.make_interpretation_split(80, 84021)
        seen = [example.text for example in cls.training]
        cls.splits = {}
        for index, shift in enumerate(semantic_tasks.INTERPRETATION_SHIFTS):
            examples = semantic_tasks.make_interpretation_split(
                80, 84031 + index, shift, exclude=seen, stage="development",
            )
            cls.splits[shift] = examples
            seen.extend(example.text for example in examples)

    def test_interpretation_labels_and_grouped_queries_match_independent_prose(self):
        for shift, examples in {"train": self.training, **self.splits}.items():
            max_distance = 0
            for example in examples:
                with self.subTest(shift=shift, text=example.text):
                    answer, distance = SemanticTaskTests.visible_answer(example.text)
                    self.assertEqual(answer, example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), answer)
                    if answer == "yes":
                        max_distance = max(max_distance, distance)
            self.assertEqual(max_distance, 3 if shift in ("composition", "combined") else 2)
            groups = self.tasks.all_query_examples(
                examples[:8], seed=84051, stage="development", shift="iid" if shift == "train" else shift,
            )
            for group in groups:
                self.assertEqual(len(group), 4 if group[0].task == "accounting" else 12)
                self.assertEqual(len({example.rows[-1][1:3] for example in group}), len(group))
                self.assertEqual(len({self.tasks.interpretation_world_key(example.text) for example in group}), 1)
                for example in group:
                    self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), example.answer)

    def test_interpretation_splits_exclude_presented_worlds_across_queries(self):
        self.assertEqual(self.training, self.tasks.make_interpretation_split(80, 84021))
        seen = set()
        for examples in [self.training, *self.splits.values()]:
            worlds = {self.tasks.interpretation_world_key(example.text) for example in examples}
            self.assertEqual(len(worlds), len(examples))
            self.assertTrue(worlds.isdisjoint(seen))
            seen.update(worlds)
            self.assertEqual(sum(example.task == "accounting" for example in examples), 40)
            self.assertEqual(sum(example.answer == "yes" for example in examples), 20)
            self.assertEqual(sum(example.answer == "no" for example in examples), 20)
        alternatives = [group[-1].text for group in self.tasks.all_query_examples(self.training, seed=84052)]
        fresh = self.tasks.make_interpretation_split(80, 84021, exclude=alternatives)
        self.assertTrue({self.tasks.interpretation_world_key(text) for text in alternatives}.isdisjoint(
            self.tasks.interpretation_world_key(example.text) for example in fresh))

    def test_all_syntax_families_have_correct_roles_polarity_and_query_meaning(self):
        # Exhaust syntax fixtures, including final arrangements, without generating
        # or evaluating any final research worlds or trained model predictions.
        setup = "Alice starts with 9 coins. Bob starts with 2 coins. "
        for stage, events in self.tasks.INTERPRETATION_FAMILIES.items():
            for event, families in events.items():
                for index in range(len(families)):
                    sentence = self.tasks.interpretation_sentence(
                        random.Random(84060 + index), event, "Alice", "Bob", 3,
                        stage=stage, family_index=index,
                    )
                    if event == "initial":
                        text, expected = sentence + " what is Alice's balance now?", "3"
                    elif event in ("transfer", "inactive_transfer"):
                        text = setup + sentence + " what is Alice's balance now?"
                        expected = "6" if event == "transfer" else "9"
                    elif event == "balance_query":
                        text, expected = setup + sentence, "9"
                    elif event == "before":
                        text, expected = sentence + " is Alice before Bob?", "yes"
                    else:
                        text, expected = "Alice is before Bob. " + sentence, "yes"
                    with self.subTest(stage=stage, event=event, sentence=sentence):
                        self.assertEqual(SemanticTaskTests.visible_answer(text)[0], expected)
            for other in self.tasks.INTERPRETATION_FAMILIES:
                if other != stage:
                    self.assertTrue(set(template for rows in events.values() for template in rows).isdisjoint(
                        template for rows in self.tasks.INTERPRETATION_FAMILIES[other].values() for template in rows))

    def test_counterfactual_pairs_change_visible_answers_and_preserve_unrelated_facts(self):
        for shift in ("iid", "wording", "combined"):
            pairs = self.tasks.interpretation_contrast_pairs(
                self.splits[shift][:20], seed=84071, stage="development", shift=shift,
            )
            self.assertEqual({contrast for _, _, contrast in pairs}, {"role", "polarity"})
            for base, changed, contrast in pairs:
                with self.subTest(shift=shift, contrast=contrast, text=changed.text):
                    self.assertNotEqual(base.answer, changed.answer)
                    self.assertEqual(sum(a != b for a, b in zip(base.sentences, changed.sentences)), 1)
                    for example in (base, changed):
                        self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                        self.assertEqual(self.tasks.execute(example.rows), example.answer)

    def test_retention_split_preserves_old_meaning_and_excludes_worlds_across_queries(self):
        import semantic_experiment

        original = semantic_experiment.retention_split(40, 84211)
        self.assertEqual(original, semantic_experiment.retention_split(40, 84211))
        alternatives = [group[-1].text for group in self.tasks.all_query_examples(original, seed=84212)]
        fresh = semantic_experiment.retention_split(40, 84211, exclude=alternatives)
        old_worlds = {self.tasks.interpretation_world_key(example.text) for example in original}
        new_worlds = {self.tasks.interpretation_world_key(example.text) for example in fresh}
        self.assertEqual(old_worlds, {self.tasks.interpretation_world_key(text) for text in alternatives})
        self.assertTrue(old_worlds.isdisjoint(new_worlds))
        for examples in (original, fresh):
            self.assertEqual(len({self.tasks.interpretation_world_key(example.text) for example in examples}), 40)
            self.assertEqual(sum(example.task == "accounting" for example in examples), 20)
            self.assertEqual(sum(example.answer == "yes" for example in examples), 10)
            self.assertEqual(sum(example.answer == "no" for example in examples), 10)
            for example in examples:
                self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                self.assertEqual(self.tasks.execute(example.rows), example.answer)
        new_training = self.tasks.make_interpretation_split(40, 84213, exclude=alternatives)
        new_training_worlds = {self.tasks.interpretation_world_key(example.text) for example in new_training}
        self.assertTrue(old_worlds.isdisjoint(new_training_worlds))
        mixed_exclusion = [example.text for example in original + new_training]
        development = semantic_experiment.retention_split(40, 84214, exclude=mixed_exclusion)
        self.assertTrue((old_worlds | new_training_worlds).isdisjoint(
            self.tasks.interpretation_world_key(example.text) for example in development))

    def test_retention_candidate_budget_stays_fixed_when_requested_count_changes(self):
        import semantic_experiment

        for count in (20, 40):
            with patch.object(semantic_experiment, "make_stability_split",
                              wraps=self.tasks.make_stability_split) as generate:
                examples = semantic_experiment.retention_split(count, 84221, candidate_count=320)
            self.assertEqual(generate.call_args.args[:2], (320, 84221))
            self.assertEqual(len(examples), count)
            self.assertEqual(len({self.tasks.interpretation_world_key(example.text) for example in examples}), count)
            self.assertEqual(sum(example.task == "accounting" for example in examples), count // 2)
            self.assertEqual(sum(example.answer == "yes" for example in examples), count // 4)
            self.assertEqual(sum(example.answer == "no" for example in examples), count // 4)
            for example in examples:
                self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
        # Exhausting a declared candidate pool must fail instead of recycling worlds.
        with self.assertRaises(ValueError):
            semantic_experiment.retention_split(8, 84221, candidate_count=4)

    def test_oversized_wording_holdout_fails_before_training_or_evaluation(self):
        import semantic_experiment

        for requested, total in ((325, 600), (600, 600)):
            arguments = ["semantic_experiment.py", "--protocol", "interpretation", "--phase", "confirm",
                         "--wording-test-size", str(requested), "--test-size", str(total)]
            with patch("sys.argv", arguments), patch("sys.stderr", new=io.StringIO()), \
                    patch.object(semantic_experiment, "run_interpretation") as run:
                with self.assertRaises(SystemExit) as error:
                    semantic_experiment.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
        for requested, total in ((324, 600), (600, 300)):
            arguments = ["semantic_experiment.py", "--protocol", "interpretation",
                         "--wording-test-size", str(requested), "--test-size", str(total)]
            with patch("sys.argv", arguments), patch.object(semantic_experiment, "run_interpretation") as run:
                semantic_experiment.main()
                run.assert_called_once()


# Scaling datasets and experiment-selection controls
class SemanticScalingTaskTests(unittest.TestCase):
    """Validate nested scaling corpora, held-out syntax, and capacity limits."""

    @classmethod
    def setUpClass(cls):
        import semantic_tasks

        cls.tasks = semantic_tasks
        cls.training = semantic_tasks.make_scaling_split(120, 94001)
        seen = [example.text for example in cls.training]
        cls.splits = {}
        for index, shift in enumerate(semantic_tasks.INTERPRETATION_SHIFTS):
            examples = semantic_tasks.make_scaling_split(
                40, 94011 + index, shift, exclude=seen, stage="development",
            )
            cls.splits[shift] = examples
            seen.extend(example.text for example in examples)

    @staticmethod
    def visible_program(example):
        """Compare latent events after reversing first-mention identity numbering."""
        import semantic_tasks

        names = semantic_tasks.surface_names(example.text)
        return (tuple((kind, names[a] if a >= 0 else None, names[b] if b >= 0 else None, value, active)
                      for kind, a, b, value, active in example.rows), example.answer, example.task)

    def test_legacy_protocol_fingerprints_remain_unchanged(self):
        fixtures = (
            (self.tasks.make_split(120, 70191),
             "82735a8fe111e95b806e4c592c4bb80add4448130a91792f14c8421c47bf09aa"),
            (self.tasks.make_stability_split(120, 85001),
             "1c430d52662a8cdc05a169412ff550b505b6bb5fe416f784cdb239782e574fb0"),
            (self.tasks.make_interpretation_split(80, 84021),
             "a5ac8544699733adec41f17a410eee9026c102a468542451f05da2021bfd1f10"),
            (self.tasks.make_scaling_split(120, 94001, diversity="narrow"),
             "fb45dbe406bc4c4280d55868e8e064ba9dac844800fd8bb3178b7680a7a4430c"),
            (self.tasks.make_scaling_split(120, 94001, diversity="broad"),
             "5a3bb36eb536712159f42ad083e54f21e389c288e470784d928975c4f746b0f6"),
        )
        for examples, fingerprint in fixtures:
            self.assertEqual(self.tasks.split_fingerprint(examples), fingerprint)

    def test_scaling_counts_are_nested_and_diversity_preserves_latent_examples(self):
        for diversity in ("narrow", "broad"):
            with self.subTest(diversity=diversity):
                small = self.tasks.make_scaling_split(80, 94031, diversity=diversity)
                large = self.tasks.make_scaling_split(240, 94031, diversity=diversity)
                self.assertEqual(small, large[:80])
                for examples in (small, large):
                    self.assertEqual(sum(example.task == "accounting" for example in examples), len(examples) // 2)
                    self.assertEqual(sum(example.answer == "yes" for example in examples), len(examples) // 4)
                    self.assertEqual(sum(example.answer == "no" for example in examples), len(examples) // 4)
        narrow = self.tasks.make_scaling_split(240, 94032, diversity="narrow")
        broad = self.tasks.make_scaling_split(240, 94032, diversity="broad")
        self.assertEqual([self.visible_program(example) for example in narrow],
                         [self.visible_program(example) for example in broad])
        self.assertTrue(any(left.text != right.text for left, right in zip(narrow, broad)))

    def test_scaling_worlds_exclude_other_queries_and_remain_disjoint(self):
        seen = set()
        for examples in (self.training, *self.splits.values()):
            worlds = {self.tasks.interpretation_world_key(example.text) for example in examples}
            self.assertEqual(len(worlds), len(examples))
            self.assertTrue(worlds.isdisjoint(seen))
            seen.update(worlds)
        alternatives = [group[-1].text for group in self.tasks.all_query_examples(
            self.training, seed=94033, protocol="scaling",
        )]
        for diversity in ("narrow", "broad"):
            small = self.tasks.make_scaling_split(40, 94001, exclude=alternatives, diversity=diversity)
            large = self.tasks.make_scaling_split(80, 94001, exclude=alternatives, diversity=diversity)
            self.assertEqual(small, large[:40])
            self.assertTrue({self.tasks.interpretation_world_key(text) for text in alternatives}.isdisjoint(
                self.tasks.interpretation_world_key(example.text) for example in large))

    def test_scaling_labels_queries_and_contrasts_match_independent_prose(self):
        for shift, examples in {"train": self.training, **self.splits}.items():
            effective_shift = "iid" if shift == "train" else shift
            max_distance = 0
            for example in examples:
                answer, distance = SemanticTaskTests.visible_answer(example.text)
                self.assertEqual(answer, example.answer, example.text)
                self.assertEqual(self.tasks.execute(example.rows), answer)
                if answer == "yes":
                    max_distance = max(max_distance, distance)
            self.assertEqual(max_distance, 3 if shift in ("composition", "combined") else 2)
            groups = self.tasks.all_query_examples(
                examples[:8], seed=94041, stage="development", shift=effective_shift, protocol="scaling",
            )
            for group in groups:
                self.assertEqual(len(group), 4 if group[0].task == "accounting" else 12)
                self.assertEqual(len({example.rows[-1][1:3] for example in group}), len(group))
                self.assertEqual(len({self.tasks.interpretation_world_key(example.text) for example in group}), 1)
                for example in group:
                    self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), example.answer)
            pairs = self.tasks.interpretation_contrast_pairs(
                examples[:8], seed=94042, stage="development", shift=effective_shift, protocol="scaling",
            )
            self.assertEqual({contrast for _, _, contrast in pairs}, {"role", "polarity"})
            for base, changed, contrast in pairs:
                self.assertNotEqual(base.answer, changed.answer, contrast)
                self.assertEqual(sum(a != b for a, b in zip(base.sentences, changed.sentences)), 1)
                for example in (base, changed):
                    self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), example.answer)

    def test_scaling_family_meaning_and_familiar_vocabulary_are_independently_checked(self):
        import semantic_model

        vocabularies = {diversity: semantic_model.fit_lexicon(example.text for example in
            self.tasks.make_scaling_split(800, 94061, diversity=diversity))
            for diversity in ("narrow", "broad")}
        setup = "Alice starts with 9 coins. Bob starts with 2 coins. "
        for stage, events in self.tasks.SCALING_FAMILIES.items():
            for event, families in events.items():
                for index in range(len(families)):
                    for lexical_seed in range(4):
                        sentence = self.tasks.scaling_sentence(
                            random.Random(94071 + lexical_seed), event, "Alice", "Bob", 3,
                            stage=stage, family_index=index,
                        )
                        if event == "initial":
                            text, expected = sentence + " what is Alice's balance now?", "3"
                        elif event in ("transfer", "inactive_transfer"):
                            text = setup + sentence + " what is Alice's balance now?"
                            expected = "6" if event == "transfer" else "9"
                        elif event == "balance_query":
                            text, expected = setup + sentence, "9"
                        elif event == "before":
                            text, expected = sentence + " is Alice before Bob?", "yes"
                        else:
                            text, expected = "Alice is before Bob. " + sentence, "yes"
                        with self.subTest(stage=stage, event=event, sentence=sentence):
                            self.assertEqual(SemanticTaskTests.visible_answer(text)[0], expected)
                            tokens = semantic_model.lexical_tokens(self.tasks.surface_sentences(sentence)[0])
                            for diversity, vocabulary in vocabularies.items():
                                self.assertTrue(set(tokens).issubset(vocabulary),
                                                (diversity, set(tokens) - vocabulary.keys()))
        observed = {template for stage in self.tasks.INTERPRETATION_FAMILIES.values()
                    for families in stage.values() for template in families}
        for stage in ("development", "final"):
            reserved = {template for families in self.tasks.SCALING_FAMILIES[stage].values() for template in families}
            self.assertTrue(reserved.isdisjoint(observed))
            observed.update(reserved)
        broad = {template for families in self.tasks.SCALING_FAMILIES["broad"].values() for template in families}
        for stage in ("development", "final"):
            self.assertTrue(broad.isdisjoint(template for families in self.tasks.SCALING_FAMILIES[stage].values()
                                            for template in families))

    def test_scaling_capacity_and_stage_errors_fail_without_recycling_worlds(self):
        for kwargs in ({"count": -1}, {"shift": "unknown"}, {"stage": "unknown"},
                       {"diversity": "unknown"}, {"shift": "wording"}, {"shift": "combined"},
                       {"count": 6001, "diversity": "narrow"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.tasks.make_scaling_split(seed=94081, **{"count": 4, **kwargs})
        for stage in ("development", "final"):
            capacity = self.tasks.scaling_relation_capacity(stage)
            self.assertGreaterEqual(capacity, 300)
            with self.assertRaises(ValueError):
                self.tasks.make_scaling_split(2 * capacity + 2, 94082, "wording", stage=stage)


class SemanticScalingRunnerTests(unittest.TestCase):
    """Guard sampling, ranking, and gates used by scaling experiments."""

    def test_data_amount_sampling_preserves_retained_examples_and_canonical_draws(self):
        import semantic_experiment

        global_state = torch.random.get_rng_state().clone()
        sequences = {}
        for count in (9000, 21000):
            generator = torch.Generator().manual_seed(94111)
            amount_generator = torch.Generator().manual_seed(94112)
            sequences[count] = torch.cat([semantic_experiment.scaling_indices(
                count, 128, generator, amount_generator,
            ) for _ in range(4)])
            self.assertTrue(((sequences[count] >= 0) & (sequences[count] < count)).all().item())
        small, large = sequences[9000], sequences[21000]
        expected_generator = torch.Generator().manual_seed(94111)
        expected = torch.cat([torch.randint(9000, (128,), generator=expected_generator) for _ in range(4)])
        torch.testing.assert_close(small, expected, atol=0, rtol=0)
        retained = small < 3000
        torch.testing.assert_close(retained, large < 3000, atol=0, rtol=0)
        torch.testing.assert_close(small[retained], large[retained], atol=0, rtol=0)
        torch.testing.assert_close((large[~retained] - 3000) % 6000 + 3000,
                                   small[~retained], atol=0, rtol=0)
        self.assertEqual(set(((large[~retained] - 3000) // 6000).tolist()), {0, 1, 2})
        repeated_generator = torch.Generator().manual_seed(94111)
        repeated_amount_generator = torch.Generator().manual_seed(94112)
        repeated = torch.cat([semantic_experiment.scaling_indices(
            21000, 128, repeated_generator, repeated_amount_generator,
        ) for _ in range(4)])
        torch.testing.assert_close(large, repeated, atol=0, rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state, atol=0, rtol=0)
        for count in (0, 3000, 6000, 9001, 18000, 21001):
            with self.subTest(count=count), self.assertRaises(ValueError):
                semantic_experiment.scaling_indices(count, 32, repeated_generator, repeated_amount_generator)

    def test_role_distance_gate_requires_every_seed_and_both_answer_paths(self):
        import semantic_experiment

        self.assertFalse(semantic_experiment.scaling_diagnostic_gate([])["passed"])
        good = {"parser": {"program_accuracy": 1.},
                "parsed": {"macro_exact_accuracy": 1.}, "executor": {"macro_exact_accuracy": 1.}}
        trials = [{"seed": seed, "role_distance_diagnostic": copy.deepcopy(good)}
                  for seed in (94121, 94122, 94123)]
        self.assertTrue(semantic_experiment.scaling_diagnostic_gate(trials)["passed"])
        for mode, field, metric in (("parser", "program_accuracy", "program"),
                                   ("parsed", "macro_exact_accuracy", "parsed"),
                                   ("executor", "macro_exact_accuracy", "executor")):
            changed = copy.deepcopy(trials)
            changed[1]["role_distance_diagnostic"][mode][field] = .5
            gate = semantic_experiment.scaling_diagnostic_gate(changed)
            self.assertFalse(gate["passed"])
            self.assertEqual([(row["seed"], row["metric"]) for row in gate["failures"]], [(94122, metric)])
            for value in (float("nan"), float("inf"), -.01, 1.01):
                changed[1]["role_distance_diagnostic"][mode][field] = value
                with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                    semantic_experiment.scaling_diagnostic_gate(changed)

    @staticmethod
    def ranked_result(score=1.0, parameters=76094):
        shifts = ("iid", "retention", "wording", "combined", "length", "numbers", "composition")
        return {"trials": [
            {"seed": seed, "inference_parameters": {"parsed": parameters},
             "development": {shift: {"parser": {"per_task": {
                 task: {"program_accuracy": score} for task in ("accounting", "relations")}}}
                 for shift in shifts}}
            for seed in (94101, 94102, 94103)]}

    def test_recipe_ranking_uses_complete_development_programs_and_ignores_test_scores(self):
        import semantic_experiment

        base = self.ranked_result(.95)
        rank = semantic_experiment.development_rank(base)
        changed_test = copy.deepcopy(base)
        for trial in changed_test["trials"]:
            trial["test"] = {"wording": {"program_accuracy": float("nan")},
                             "combined": {"parsed": {"exact_accuracy": -1000}}}
        self.assertEqual(rank, semantic_experiment.development_rank(changed_test))
        # A high aggregate cannot hide one failed seed, task, or shift.
        high_mean = self.ranked_result(.999)
        high_mean["trials"][2]["development"]["combined"]["parser"]["per_task"]["accounting"]["program_accuracy"] = .94
        self.assertGreater(rank, semantic_experiment.development_rank(high_mean))
        better_mean = self.ranked_result(.96)
        better_mean["trials"][0]["development"]["wording"]["parser"]["per_task"]["relations"]["program_accuracy"] = .95
        self.assertGreater(semantic_experiment.development_rank(better_mean), rank)
        self.assertGreater(semantic_experiment.development_rank(self.ranked_result(.95, 60000)), rank)
        for field in ("trial", "shift", "task", "parameters"):
            invalid = copy.deepcopy(base)
            if field == "trial":
                invalid["trials"][0]["development"] = {}
            elif field == "shift":
                del invalid["trials"][0]["development"]["combined"]
            elif field == "task":
                del invalid["trials"][0]["development"]["wording"]["parser"]["per_task"]["accounting"]
            else:
                invalid["trials"][1]["inference_parameters"]["parsed"] += 1
            with self.subTest(missing=field), self.assertRaises(ValueError):
                semantic_experiment.development_rank(invalid)
        for score in (float("nan"), float("inf"), -.01, 1.01):
            with self.subTest(score=score), self.assertRaises(ValueError):
                semantic_experiment.development_rank(self.ranked_result(score))
        with self.assertRaises(ValueError):
            semantic_experiment.development_rank({"trials": []})

    def test_every_capacity_data_cell_uses_the_same_complete_exclusion_universe(self):
        import semantic_experiment
        import semantic_tasks

        def fixtures(count, label):
            return [semantic_tasks.Example(f"{label} {index}.", (), (), "0", "accounting")
                    for index in range(count)]

        retained = fixtures(3000, "retained")
        narrow = fixtures(6000, "narrow")
        broad = fixtures(18000, "broad")
        training_universe = {example.text for example in retained + narrow + broad}
        snapshots = []
        def generate(count, seed, shift="iid", exclude=(), stage="train", diversity="broad"):
            if seed == 92001:
                self.assertEqual(set(exclude), {example.text for example in retained})
                expected = narrow if diversity == "narrow" else broad
                self.assertEqual(count, len(expected))
                return expected
            self.assertTrue(training_universe.issubset(exclude))
            snapshots.append((seed, shift, frozenset(exclude)))
            return fixtures(count, f"development {seed} {shift}")

        def retention(count, seed, exclude=(), candidate_count=None):
            self.assertTrue(training_universe.issubset(exclude))
            snapshots.append((seed, "retention", frozenset(exclude)))
            return fixtures(count, "retention development")

        results, exclusions = [], []
        with patch.object(semantic_experiment, "make_stability_split", return_value=retained), \
                patch.object(semantic_tasks, "make_scaling_split", side_effect=generate), \
                patch.object(semantic_experiment, "retention_split", side_effect=retention):
            for count, diversity in ((6000, "narrow"), (6000, "broad"), (18000, "broad")):
                snapshots.clear()
                results.append(semantic_experiment.scaling_data(count, diversity, 8, 8))
                exclusions.append(list(snapshots))
        self.assertEqual([len(result[0]) for result in results], [9000, 9000, 21000])
        self.assertEqual(results[1][0], results[2][0][:9000])
        for result in results:
            self.assertEqual(result[1], retained)
            self.assertEqual(result[2], results[0][2])
            self.assertEqual(result[3], results[0][3])
            self.assertEqual(result[4], results[0][4])
        self.assertEqual(exclusions[0], exclusions[1])
        self.assertEqual(exclusions[1], exclusions[2])


# Role-binding and polarity datasets
class SemanticBindingTaskTests(unittest.TestCase):
    """Validate distant-role language, balanced views, and paraphrase labels."""

    @classmethod
    def setUpClass(cls):
        import semantic_tasks

        cls.tasks = semantic_tasks
        cls.training = semantic_tasks.make_binding_split(120, 94221)
        cls.splits, excluded = {}, [example.text for example in cls.training]
        for index, shift in enumerate(semantic_tasks.BINDING_SHIFTS):
            examples = semantic_tasks.make_binding_split(40, 94231 + index, shift, exclude=excluded,
                                                         stage="development")
            cls.splits[shift] = examples
            excluded.extend(example.text for example in examples)

    def test_independent_predicate_interpreter_preserves_names_and_dual_role_meaning(self):
        prefix = "Alice starts with 9 coins. Bruno starts with 2 coins. "
        query = " what is Bruno's balance now?"
        for sentence, answer in (
            ("Bruno transfers Alice 1 coins.", "1"),
            ("Bruno does not transfer Alice 1 coins.", "2"),
            ("Bruno receives 1 coins and Alice gives the coins.", "3"),
            ("the one who receives the coins is Alice and Bruno is the one who gives 1 coins.", "1"),
            ("the one who gives the coins is Alice and Bruno is the one who receives 1 coins.", "3"),
            ("Bruno does not receive 1 coins and Alice does not give the coins.", "2"),
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(SemanticTaskTests.visible_answer(prefix + sentence + query)[0], answer)

    def test_quantity_balancing_changes_only_word_order_in_paired_training_examples(self):
        import semantic_model

        baseline = self.tasks.make_binding_split(240, 94301)
        self.assertEqual(self.tasks.split_fingerprint(baseline),
                         "ebc86363022169d1fe8fe6a6367d177e7799f5f32adc2ea1fc48716bd3cd3991")
        self.assertEqual(baseline, self.tasks.make_binding_split(240, 94301, balance_quantities=False))
        balanced = self.tasks.make_binding_split(240, 94301, balance_quantities=True)
        self.assertEqual(balanced, self.tasks.make_binding_split(480, 94301, balance_quantities=True)[:240])
        self.assertEqual([SemanticScalingTaskTests.visible_program(example) for example in baseline],
                         [SemanticScalingTaskTests.visible_program(example) for example in balanced])
        changed = 0
        for old, new in zip(baseline, balanced):
            self.assertEqual(len(old.text.encode("utf-8")), len(new.text.encode("utf-8")))
            self.assertEqual(len(old.sentences), len(new.sentences))
            self.assertEqual(old.rows, new.rows)
            for left, right in zip(self.tasks.surface_sentences(old.text), self.tasks.surface_sentences(new.text)):
                self.assertEqual(sorted(semantic_model.lexical_tokens(left)), sorted(semantic_model.lexical_tokens(right)))
                changed += left != right
            self.assertEqual(SemanticTaskTests.visible_answer(new.text)[0], new.answer)
        self.assertGreater(changed, 0)

    def test_quantity_balancing_covers_both_conjuncts_without_reserved_family_leakage(self):
        setup = "Alice starts with 9 coins. Bob starts with 2 coins. "
        self.assertEqual(len(self.tasks.BINDING_QUANTITY_VARIANTS), 14)
        held = {template for stage in ("development", "final")
                for families in self.tasks.BINDING_FAMILIES[stage].values() for template in families}
        self.assertTrue(set(self.tasks.BINDING_QUANTITY_VARIANTS.values()).isdisjoint(held))
        changed_families, placements = {"transfer": set(), "inactive_transfer": set()}, set()
        for event in changed_families:
            for index, template in enumerate(self.tasks.BINDING_FAMILIES["train"][event]):
                for seed in range(94311, 94327):
                    old = self.tasks.binding_sentence(random.Random(seed), event, "Alice", "Bob", 3,
                                                      family_index=index)
                    new = self.tasks.binding_sentence(random.Random(seed), event, "Alice", "Bob", 3,
                                                      family_index=index, balance_quantities=True)
                    if old != new:
                        changed_families[event].add(template)
                        self.assertIn(template, self.tasks.BINDING_QUANTITY_VARIANTS)
                        self.assertEqual(sorted(re.findall(r"\w+|[^\w\s]", old)), sorted(re.findall(r"\w+|[^\w\s]", new)))
                        for sentence in (old, new):
                            self.assertEqual(sentence.count(" and "), 1)
                            placements.add(sentence.index("3") < sentence.index(" and "))
                            expected = "6" if event == "transfer" else "9"
                            self.assertEqual(SemanticTaskTests.visible_answer(
                                setup + sentence + " what is Alice's balance now?")[0], expected)
            self.assertEqual(len(changed_families[event]), 7)
        self.assertEqual(placements, {False, True})
        for stage in ("development", "final"):
            for event, families in self.tasks.BINDING_FAMILIES[stage].items():
                for index in range(len(families)):
                    options = {"stage": stage, "family_index": index}
                    old = self.tasks.binding_sentence(random.Random(94331), event, "Alice", "Bob", 3, **options)
                    new = self.tasks.binding_sentence(random.Random(94331), event, "Alice", "Bob", 3,
                                                      balance_quantities=True, **options)
                    self.assertEqual(old, new)

    def test_binding_worlds_are_nested_balanced_and_exclude_alternate_queries(self):
        self.assertEqual(self.training, self.tasks.make_binding_split(240, 94221)[:120])
        seen = set()
        for examples in (self.training, *self.splits.values()):
            worlds = {self.tasks.interpretation_world_key(example.text) for example in examples}
            self.assertEqual(len(worlds), len(examples))
            self.assertTrue(worlds.isdisjoint(seen))
            seen.update(worlds)
            self.assertEqual(sum(example.task == "accounting" for example in examples), len(examples) // 2)
            self.assertEqual(sum(example.answer == "yes" for example in examples), len(examples) // 4)
            self.assertEqual(sum(example.answer == "no" for example in examples), len(examples) // 4)
        alternatives = [group[-1].text for group in self.tasks.all_query_examples(
            self.training, seed=94241, protocol="binding")]
        fresh = self.tasks.make_binding_split(80, 94221, exclude=alternatives)
        self.assertTrue({self.tasks.interpretation_world_key(text) for text in alternatives}.isdisjoint(
            self.tasks.interpretation_world_key(example.text) for example in fresh))
        for kwargs in ({"count": -1}, {"stage": "unknown"}, {"shift": "unknown"},
                       {"shift": "wording"}, {"shift": "combined"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.tasks.make_binding_split(seed=94242, **{"count": 4, **kwargs})

    def test_binding_labels_expansions_and_paraphrases_follow_visible_predicates(self):
        for shift, examples in {"train": self.training, **self.splits}.items():
            effective_shift = "iid" if shift == "train" else shift
            max_distance = 0
            for example in examples:
                answer, distance = SemanticTaskTests.visible_answer(example.text)
                self.assertEqual(answer, example.answer, example.text)
                self.assertEqual(self.tasks.execute(example.rows), answer)
                if answer == "yes":
                    max_distance = max(max_distance, distance)
            self.assertEqual(max_distance, 3 if shift in ("combined", "composition") else 2)
            groups = self.tasks.all_query_examples(examples[:8], seed=94243, stage="development",
                                                    shift=effective_shift, protocol="binding")
            for group in groups:
                self.assertEqual(len(group), 4 if group[0].task == "accounting" else 12)
                self.assertEqual(len({example.rows[-1][1:3] for example in group}), len(group))
                self.assertEqual(len({self.tasks.interpretation_world_key(example.text) for example in group}), 1)
                for example in group:
                    self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), example.answer)
            pairs = self.tasks.interpretation_contrast_pairs(examples[:8], seed=94244,
                stage="development", shift=effective_shift, protocol="binding")
            self.assertEqual({contrast for _, _, contrast in pairs}, {"role", "polarity"})
            for base, changed, contrast in pairs:
                self.assertNotEqual(base.answer, changed.answer, contrast)
                self.assertEqual(sum(left != right for left, right in zip(base.sentences, changed.sentences)), 1)
                for example in (base, changed):
                    self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)
                    self.assertEqual(self.tasks.execute(example.rows), example.answer)
        paraphrases = self.tasks.binding_paraphrase_examples(self.training, seed=94245)
        self.assertEqual([SemanticScalingTaskTests.visible_program(example) for example in self.training],
                         [SemanticScalingTaskTests.visible_program(example) for example in paraphrases])
        self.assertTrue(any(left.text != right.text for left, right in zip(self.training, paraphrases)))
        for example in paraphrases:
            self.assertEqual(SemanticTaskTests.visible_answer(example.text)[0], example.answer)

    def test_binding_family_fixtures_keep_meaning_and_familiar_words_without_old_holdout_overlap(self):
        import semantic_model

        training = self.tasks.make_binding_split(1200, 94251)
        vocabulary = semantic_model.fit_lexicon(example.text for example in training)
        self.assertIn("and", vocabulary)
        setup = "Alice starts with 9 coins. Bob starts with 2 coins. "
        for stage, events in self.tasks.BINDING_FAMILIES.items():
            self.assertEqual(len(events["transfer"]), len(events["inactive_transfer"]))
            for event, families in events.items():
                for index in range(len(families)):
                    for lexical_seed in range(4):
                        sentence = self.tasks.binding_sentence(random.Random(94261 + lexical_seed),
                            event, "Alice", "Bob", 3, stage=stage, family_index=index)
                        if event == "initial":
                            text, expected = sentence + " what is Alice's balance now?", "3"
                        elif event in ("transfer", "inactive_transfer"):
                            text, expected = setup + sentence + " what is Alice's balance now?", "6" if event == "transfer" else "9"
                        elif event == "balance_query":
                            text, expected = setup + sentence, "9"
                        elif event == "before":
                            text, expected = sentence + " is Alice before Bob?", "yes"
                        else:
                            text, expected = "Alice is before Bob. " + sentence, "yes"
                        with self.subTest(stage=stage, event=event, sentence=sentence):
                            self.assertEqual(SemanticTaskTests.visible_answer(text)[0], expected)
                            tokens = semantic_model.lexical_tokens(self.tasks.surface_sentences(sentence)[0])
                            self.assertTrue(set(tokens).issubset(vocabulary), set(tokens) - vocabulary.keys())
                            self.assertLessEqual(len(re.findall(r"-?\d+", sentence)), 1)
        observed = {template for stage in self.tasks.SCALING_FAMILIES.values()
                    for families in stage.values() for template in families}
        observed.update(template for families in self.tasks.BINDING_FAMILIES["train"].values() for template in families)
        for stage in ("development", "final"):
            held = {template for families in self.tasks.BINDING_FAMILIES[stage].values() for template in families}
            self.assertTrue(held.isdisjoint(observed))
            observed.update(held)
            capacity = self.tasks.binding_relation_capacity(stage)
            self.assertGreaterEqual(capacity, 300)
            with self.assertRaises(ValueError):
                self.tasks.make_binding_split(2 * capacity + 2, 94262, "wording", stage=stage)


class SemanticPolarityTaskTests(unittest.TestCase):
    """Check active/inactive transfer reconstruction and source provenance."""

    @staticmethod
    def fixture(sentence="3 coins are given by Alice to Bob.", active=1, amount=3):
        import semantic_tasks as tasks

        sentences = [f"{name} starts with {value} coins." for name, value in
                     (("Alice", 9), ("Bob", 2), ("Clara", 5), ("David", 4))]
        rows = [(tasks.INITIAL, name, None, value, 1) for name, value in
                (("Alice", 9), ("Bob", 2), ("Clara", 5), ("David", 4))]
        return tasks._finish(sentences + [sentence, "what is Clara's balance now?"],
            rows + [(tasks.TRANSFER, "Alice", "Bob", amount, active),
                    (tasks.QUERY_BALANCE, "Clara", None, 0, 1)], "5", "accounting")

    def test_exact_polarity_reconstruction_preserves_source_facts_queries_and_lexical_choices(self):
        import semantic_tasks as tasks

        fixtures = []
        for stage, families in tasks.BINDING_FAMILIES.items():
            for active, event in ((0, "inactive_transfer"), (1, "transfer")):
                for index in range(len(families[event])):
                    for balance in (False, True) if stage == "train" else (False,):
                        sentence = tasks.binding_sentence(random.Random(94511), event, "Alice", "Bob", 3,
                            stage=stage, family_index=index, balance_quantities=balance)
                        fixtures.append((stage, self.fixture(sentence, active)))
        for example in tasks.make_stability_split(40, 94512):
            for index, row in enumerate(example.rows):
                if row[0] == tasks.TRANSFER:
                    fixtures.append(("train", self.fixture(example.sentences[index]
                        .replace(tasks.surface_names(example.text)[row[1]], "SENDER")
                        .replace(tasks.surface_names(example.text)[row[2]], "RECIPIENT")
                        .replace("SENDER", "Alice").replace("RECIPIENT", "Bob"), row[4], row[3])))
        for stage, example in fixtures:
            with self.subTest(stage=stage, sentence=example.sentences[4]):
                provenance = tasks.transfer_provenance(example, 4, stage)
                inactive, active = tasks.polarity_pair(example, 4, stage)
                self.assertEqual(provenance.render(provenance.original_active), example.sentences[4])
                self.assertEqual(provenance.origin_key, tasks.interpretation_world_key(example.text))
                self.assertEqual(provenance.original_sentence, example.sentences[4])
                self.assertEqual(provenance.clause_index, 4)
                self.assertEqual(provenance.sender, "Alice")
                self.assertEqual(provenance.recipient, "Bob")
                self.assertEqual((inactive, active)[example.rows[4][4]], example)
                self.assertEqual(tasks.polarity_world_key(example, stage), tasks.polarity_world_key(inactive, stage))
                self.assertEqual(tasks.polarity_world_key(inactive, stage), tasks.polarity_world_key(active, stage))
                for name, start, end in provenance.entity_spans:
                    self.assertEqual(example.sentences[4][start:end], name)
                for polarity, variant in enumerate((inactive, active)):
                    self.assertEqual(variant.sentences[:4] + variant.sentences[5:], example.sentences[:4] + example.sentences[5:])
                    self.assertEqual(variant.rows[:4] + variant.rows[5:], example.rows[:4] + example.rows[5:])
                    self.assertEqual(variant.rows[4], (*example.rows[4][:-1], polarity))
                    self.assertEqual(variant.sentences[-1], example.sentences[-1])
                    self.assertEqual(SemanticTaskTests.visible_answer(variant.text)[0], variant.answer)
                self.assertEqual(tuple(field.name for field in dataclasses.fields(active)),
                                 ("text", "sentences", "rows", "answer", "task"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            provenance.amount = 999
        with self.assertRaises(ValueError):
            provenance.render(True)

    def test_direct_polarity_groups_keep_origin_partitions_and_probe_both_participants(self):
        import semantic_tasks as tasks

        parents = tasks.make_binding_split(80, 94521, balance_quantities=True)
        groups, coverage = tasks.direct_polarity_groups(parents)
        parent_by_origin = {tasks.interpretation_world_key(example.text): example for example in parents}
        expected_transfers = sum(row[0] == tasks.TRANSFER for example in parents for row in example.rows)
        self.assertEqual(coverage["parents"], len(parents))
        self.assertEqual(coverage["transfer_clauses"], expected_transfers)
        self.assertEqual(coverage["supported_clauses"], expected_transfers)
        self.assertEqual(coverage["groups"], len(groups))
        self.assertEqual(coverage["views"], len(groups) * 4)
        self.assertEqual(coverage["unsupported_clauses"], 0)
        self.assertEqual(coverage["ambiguous_clauses"], 0)
        for group in groups:
            parent = parent_by_origin[group.origin_key]
            index, row = group.clause_index, parent.rows[group.clause_index]
            equivalent_world = tasks.polarity_world_key(parent)
            self.assertEqual(group.provenance.origin_key, group.origin_key)
            self.assertEqual(len(group.examples), 4)
            for participant_index, participant in enumerate(row[1:3]):
                inactive, active = group.examples[2 * participant_index:2 * participant_index + 2]
                self.assertEqual(int(active.answer) - int(inactive.answer), -row[3] if participant_index == 0 else row[3])
                for polarity, variant in enumerate((inactive, active)):
                    self.assertEqual(tasks.polarity_world_key(variant), equivalent_world)
                    self.assertEqual(variant.rows[-1][1], participant)
                    self.assertEqual(variant.rows[index][-1], polarity)
                    self.assertEqual(re.sub(r"\b[A-Z][a-z]+\b", "NAME", variant.sentences[-1]),
                                     re.sub(r"\b[A-Z][a-z]+\b", "NAME", parent.sentences[-1]))
                    for clause in range(len(parent.sentences) - 1):
                        if clause != index:
                            self.assertEqual(variant.sentences[clause], parent.sentences[clause])
                    self.assertEqual(SemanticTaskTests.visible_answer(variant.text)[0], variant.answer)
        zero = self.fixture("0 coins are given by Alice to Bob.", amount=0)
        inactive, active = tasks.polarity_pair(zero, 4)
        self.assertEqual(inactive.answer, active.answer)
        zero_groups, zero_coverage = tasks.direct_polarity_groups([zero])
        self.assertEqual(zero_groups, [])
        self.assertEqual(zero_coverage["zero_amount_skipped"], 1)
        self.assertEqual(zero_coverage["supported_clauses"], 1)
        included, included_coverage = tasks.direct_polarity_groups([zero], include_zero=True)
        self.assertEqual(len(included), 1)
        self.assertEqual(included_coverage["zero_amount_skipped"], 0)
        self.assertEqual(included_coverage["zero_amount_groups"], 1)
        self.assertEqual([example.answer for example in included[0].examples], ["9", "9", "2", "2"])
        self.assertEqual([example.rows[4][4] for example in included[0].examples], [0, 1, 0, 1])
        for example in parents:
            if example.task == "relations":
                self.assertEqual(tasks.polarity_world_key(example), tasks.interpretation_world_key(example.text))

    def test_polarity_reconstruction_rejects_unknown_corrupt_and_ambiguous_source(self):
        import semantic_tasks as tasks

        example = self.fixture()
        for invalid, index in ((self.fixture("Alice dances Bob 3 coins."), 4),
                               (dataclasses.replace(example, text=example.text + " "), 4),
                               (dataclasses.replace(example, rows=example.rows[:4] +
                                ((tasks.TRANSFER, 0, 1, 4, 1),) + example.rows[5:]), 4),
                               (example, 0), (example, -1), (example, 99)):
            with self.subTest(index=index, text=invalid.text), self.assertRaises(ValueError):
                tasks.transfer_provenance(invalid, index)
        with self.assertRaises(ValueError):
            tasks.direct_polarity_groups([self.fixture("Alice dances Bob 3 coins.")])
        canonical = "QUANTITY coins are given by SENDER to RECIPIENT."
        records = tasks._polarity_lookup("train")[(1, canonical)]
        conflict = (*records[0][:-1], ("a different polarity partner.", records[0][-1][1]))
        with patch.object(tasks, "_polarity_lookup", return_value={(1, canonical): [records[0], conflict]}):
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                tasks.transfer_provenance(example, 4)

    def test_source_frames_label_both_semantic_arguments_at_each_coreferring_predicate(self):
        import semantic_model
        import semantic_tasks as tasks

        fixtures = (
            ("Alice gives Bob 3 coins.", 1, (1,), 0, 2),
            ("Alice does not give Bob 3 coins.", 0, (3,), 0, 4),
            ("Alice gives 3 coins and Bob receives the coins.", 1, (1, 6), 0, 5),
            ("Alice does not give 3 coins and Bob does not receive the coins.", 0, (3, 10), 0, 7),
            ("Bob receives from Alice the 3 coins that are given.", 1, (1, 9), 3, 0),
            ("Bob does not receive from Alice the 3 coins that are not given.", 0, (3, 12), 5, 0),
            ("3 coins are given by Alice and are given to Bob.", 1, (3, 8), 5, 10),
            ("3 coins are not given by Alice and are not given to Bob.", 0, (4, 10), 6, 12),
        )
        for sentence, active, predicates, sender, recipient in fixtures:
            example = self.fixture(sentence, active)
            with self.subTest(sentence=sentence):
                frame = tasks.transfer_source_frame(example, 4)
                self.assertEqual((frame.predicate_positions, frame.sender_position, frame.recipient_position),
                                 (predicates, sender, recipient))
                self.assertEqual(frame.origin_key, tasks.interpretation_world_key(example.text))
                self.assertEqual((frame.clause_index, frame.sentence), (4, sentence))
                normalized = tasks.surface_sentences(sentence)[0]
                tokens = semantic_model.lexical_tokens(normalized)
                self.assertTrue(tokens[sender].startswith("<entity:"))
                self.assertTrue(tokens[recipient].startswith("<entity:"))
                self.assertEqual({tokens[position] for position in predicates},
                                 set(re.findall(r"\b(?:gives|give|receives|receive|given)\b", sentence)))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    frame.sender_position = 99
        self.assertEqual(tuple(field.name for field in dataclasses.fields(example)),
                         ("text", "sentences", "rows", "answer", "task"))

    def test_source_frames_validate_exact_input_and_keep_heldout_annotations_explicit(self):
        import semantic_tasks as tasks

        for stage in ("train", "development", "final"):
            sentence = tasks.binding_sentence(random.Random(94601), "inactive_transfer", "Alice", "Bob", 0,
                                              stage=stage, family_index=0)
            example = self.fixture(sentence, active=0, amount=0)
            frame = tasks.transfer_source_frame(example, 4, stage=stage)
            self.assertEqual(frame.sentence, sentence)
            self.assertTrue(frame.predicate_positions)
            self.assertNotEqual(frame.sender_position, frame.recipient_position)
        for invalid, index, stage in ((self.fixture(), 0, "train"),
                                      (self.fixture("Alice dances Bob 3 coins."), 4, "train"),
                                      (self.fixture(), 4, "unknown"),
                                      (dataclasses.replace(self.fixture(), text="corrupt source"), 4, "train")):
            with self.subTest(index=index, stage=stage), self.assertRaises(ValueError):
                tasks.transfer_source_frame(invalid, index, stage=stage)
        provenance = tasks.transfer_provenance(self.fixture(), 4)
        for changed in (dataclasses.replace(provenance, predicate_positions=()),
                        dataclasses.replace(provenance, predicate_positions=(("given", 999),)),
                        dataclasses.replace(provenance, entity_positions=(("Bob", 8),))):
            with patch.object(tasks, "transfer_provenance", return_value=changed), self.assertRaises(ValueError):
                tasks.transfer_source_frame(self.fixture(), 4)


# Guarded training, preservation, and publication protocols
class SemanticBindingRunnerTests(unittest.TestCase):
    """Protect preservation, polarity, and control arms from data leakage."""

    def test_preservation_cli_routes_full_model_control_and_rejects_recipe_changes(self):
        import semantic_experiment

        arguments = ["semantic_experiment.py", "--protocol", "preservation", "--parser-type",
                     "lexical_attachment", "--width", "192", "--parser-steps", "1500",
                     "--balance-quantities", "--retain-stability", "--train-size", "6000",
                     "--valid-size", "400", "--wording-test-size", "600", "--pair-consistency", "0",
                     "--source-weight", "0", "--freeze-after", "300"]
        with patch("sys.argv", arguments), patch.object(semantic_experiment, "run_interpretation") as run:
            semantic_experiment.main()
            run.assert_called_once()
            settings = run.call_args.args[0]
        self.assertEqual((settings.freeze_after, settings.source_weight, settings.pair_consistency), (300, 0., 0.))
        self.assertEqual(semantic_experiment.interpretation_controls(settings),
                         [("full_model", "lexical_attachment", 192, "full_model")])
        for field, value in (("freeze_after", 0), ("freeze_after", 299), ("source_weight", .1),
                             ("pair_consistency", .1), ("parser_type", "lexical_constrained")):
            changed = copy.deepcopy(settings)
            setattr(changed, field, value)
            with self.subTest(field=field), patch.object(semantic_experiment, "load_checkpoint") as load:
                with self.assertRaises(ValueError):
                    semantic_experiment.run_interpretation(changed)
                load.assert_not_called()
        for extra in (("--freeze-after", "-1"), ("--protocol", "attachment"), ("--protocol", "polarity")):
            with patch("sys.argv", [*arguments, *extra]), patch("sys.stderr", new=io.StringIO()), \
                    patch.object(semantic_experiment, "run_interpretation") as run:
                with self.assertRaises(SystemExit) as error:
                    semantic_experiment.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_preservation_routine_records_reconstruct_only_exact_complete_frozen_development_cases(self):
        import semantic_experiment

        examples = [SemanticPolarityTaskTests.fixture(f"Alice is the one who transfers Bob {amount} coins.",
                                                      amount=amount) for amount in (1, 2, 3)]
        records = [{"world_index": index, "task": example.task, "text": example.text, "answer": example.answer,
                    "faults": [{"clause_index": 4, "sentence": example.sentences[4],
                                "expected": list(example.rows[4]), "predicted": [2, 1, 0, index + 1, 1]}]}
                   for index, example in enumerate(examples)]
        development = {"iid": examples}
        reconstructed = semantic_experiment.preservation_routine_examples(development, records)
        self.assertEqual(reconstructed, examples)
        self.assertTrue(all(actual is original for actual, original in zip(reconstructed, examples)))
        invalid = [records[:-1], records + [records[0]], [records[0], records[0], records[2]]]
        for field, value in (("world_index", -1), ("world_index", 3), ("world_index", True),
                             ("text", "different source"), ("answer", "999"), ("task", "relations"),
                             ("faults", [])):
            changed = copy.deepcopy(records)
            changed[0][field] = value
            invalid.append(changed)
        for field, value in (("clause_index", -1), ("clause_index", True), ("clause_index", 99),
                             ("sentence", "different clause"), ("expected", [2, 1, 0, 1, 1])):
            changed = copy.deepcopy(records)
            changed[0]["faults"][0][field] = value
            invalid.append(changed)
        changed = copy.deepcopy(records)
        changed[0]["faults"].append(copy.deepcopy(changed[0]["faults"][0]))
        invalid.append(changed)
        for changed in invalid:
            with self.subTest(record=changed[0] if changed else None), self.assertRaises(ValueError):
                semantic_experiment.preservation_routine_examples(development, changed)

    def test_preservation_routine_gate_requires_every_known_program_and_answer_for_each_seed(self):
        import semantic_experiment

        good = {"parser": {"program_accuracy": 1.},
                "parsed": {"macro_exact_accuracy": 1.}, "executor": {"macro_exact_accuracy": 1.}}
        trials = [{"seed": seed, "routine_regression": copy.deepcopy(good)} for seed in (94631, 94632, 94633)]
        self.assertTrue(semantic_experiment.preservation_gates(trials)["passed"])
        for mode, field, metric in (("parser", "program_accuracy", "program"),
                                    ("parsed", "macro_exact_accuracy", "parsed"),
                                    ("executor", "macro_exact_accuracy", "executor")):
            changed = copy.deepcopy(trials)
            changed[1]["routine_regression"][mode][field] = 2 / 3
            gate = semantic_experiment.preservation_gates(changed)
            self.assertFalse(gate["passed"])
            self.assertEqual([(row["seed"], row["metric"], row["required"]) for row in gate["failures"]],
                             [(94632, metric, 1.)])
            del changed[1]["routine_regression"][mode][field]
            with self.assertRaises((ValueError, KeyError)):
                semantic_experiment.preservation_gates(changed)
        for value in (float("nan"), float("inf"), -.01, 1.01):
            changed = copy.deepcopy(trials)
            changed[2]["routine_regression"]["parser"]["program_accuracy"] = value
            with self.assertRaises(ValueError):
                semantic_experiment.preservation_gates(changed)
        with self.assertRaises(ValueError):
            semantic_experiment.preservation_gates([])

    def test_role_preservation_freezes_exact_parameters_and_adamw_skips_stale_gradients(self):
        import semantic_experiment
        import semantic_model

        vocabulary = semantic_model.fit_lexicon(["Alice gives Bob 3 coins."])
        parser = semantic_experiment.interpretation_parser("lexical_attachment", 16, 1, vocabulary)
        parameters = dict(parser.named_parameters())
        optimizer = torch.optim.AdamW(parser.parameters(), lr=.01, weight_decay=.2)
        for parameter in parameters.values():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        states = {name: copy.deepcopy(optimizer.state[parameter]) for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = torch.ones_like(parameter)
        report = semantic_experiment.freeze_parser_for_roles(parser)
        expected_roles = {"link_distance_bias", *(f"{module}.{field}" for module in
            ("trigger_head", "link_queries", "link_keys", "none_roles") for field in ("weight", "bias"))}
        self.assertEqual(set(report["trainable_names"]), expected_roles)
        self.assertEqual(set(report["frozen_names"]), set(parameters) - expected_roles)
        self.assertEqual(report["trainable_parameters"], sum(parameters[name].numel() for name in expected_roles))
        self.assertEqual(report["frozen_parameters"], sum(parameters[name].numel() for name in report["frozen_names"]))
        for name, parameter in parameters.items():
            self.assertEqual(parameter.requires_grad, name in expected_roles)
            if name not in expected_roles:
                self.assertIsNone(parameter.grad)
        # The existing optimizer still holds momentum and weight decay for all
        # parameters; cleared frozen gradients must make it skip both updates.
        optimizer.step()
        for name, parameter in parameters.items():
            if name in expected_roles:
                self.assertFalse(torch.equal(before[name], parameter))
                self.assertEqual(optimizer.state[parameter]["step"].item(), 2.)
            else:
                torch.testing.assert_close(before[name], parameter, atol=0, rtol=0)
                for key, value in states[name].items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value, atol=0, rtol=0)
        self.assertEqual(semantic_experiment.parameter_subset_hash(parser, report["frozen_names"]),
                         report["boundary_frozen_sha256"])
        self.assertEqual(semantic_experiment.parameter_subset_hash(parser, list(reversed(report["frozen_names"]))),
                         report["boundary_frozen_sha256"])
        for names in ([], ["nonexistent.weight"]):
            with self.assertRaises(ValueError):
                semantic_experiment.parameter_subset_hash(parser, names)
        with self.assertRaises(ValueError):
            semantic_experiment.freeze_parser_for_roles(
                semantic_experiment.interpretation_parser("lexical_constrained", 16, 1, vocabulary))
        parser.register_parameter("unexpected_parameter", torch.nn.Parameter(torch.ones(1)))
        with self.assertRaises(ValueError):
            semantic_experiment.freeze_parser_for_roles(parser)

    def test_preservation_checkpoint_selection_prefers_primary_gate_eligibility_before_worst_score(self):
        import semantic_experiment

        early = {shift: {"per_task": {task: {"program_accuracy": 1.} for task in ("accounting", "relations")},
                         "loss": .01} for shift in ("iid", "retention", "wording", "combined", "length",
                                                  "numbers", "composition")}
        early["iid"]["per_task"]["accounting"]["program_accuracy"] = .985
        qualified = copy.deepcopy(early)
        qualified["iid"]["per_task"]["accounting"]["program_accuracy"] = .99
        qualified["wording"]["per_task"]["accounting"]["program_accuracy"] = .95
        self.assertGreater(semantic_experiment.preservation_selection_key(qualified),
                           semantic_experiment.preservation_selection_key(early))
        self.assertEqual(semantic_experiment.preservation_selection_key(early)[:2], (False, .985))
        self.assertEqual(semantic_experiment.preservation_selection_key(qualified)[:2], (True, .95))
        improved = copy.deepcopy(qualified)
        improved["wording"]["per_task"]["accounting"]["program_accuracy"] = .96
        self.assertGreater(semantic_experiment.preservation_selection_key(improved),
                           semantic_experiment.preservation_selection_key(qualified))
        lower_loss = copy.deepcopy(qualified)
        lower_loss["iid"]["loss"] = 0.
        self.assertGreater(semantic_experiment.preservation_selection_key(lower_loss),
                           semantic_experiment.preservation_selection_key(qualified))
        for value in (float("nan"), float("inf"), -.01, 1.01):
            invalid = copy.deepcopy(qualified)
            invalid["combined"]["per_task"]["relations"]["program_accuracy"] = value
            with self.assertRaises(ValueError):
                semantic_experiment.preservation_selection_key(invalid)
        with self.assertRaises(ValueError):
            semantic_experiment.preservation_selection_key({})

    def test_preservation_training_boundary_keeps_prefix_optimizer_schedule_and_paired_draws(self):
        import semantic_experiment
        import semantic_model
        from types import SimpleNamespace

        texts = [f"Alice gives Bob {index + 1} coins." for index in range(8)]
        vocabulary = semantic_model.fit_lexicon(texts)
        def prepared(values):
            return SimpleNamespace(examples=[SimpleNamespace(text=text) for text in values],
                surface=semantic_model.prepare_texts(values, representation="lexical", vocabulary=vocabulary),
                rows=torch.zeros((len(values), 1, 5)))
        train = prepared(texts)
        paired = semantic_experiment.PairedTraining(prepared([text for text in texts for _ in range(2)]),
            torch.full((len(texts), 1, 3), -1, dtype=torch.long), {"views_sha256": "unit-fixture"})
        settings = SimpleNamespace(parser_steps=303, steps=303, batch_size=4, lr=.01,
                                   eval_every=303, protocol="unit-fixture")
        adamw = torch.optim.AdamW
        runs = []
        for boundary in (0, 300):
            torch.manual_seed(94621)
            model = semantic_experiment.interpretation_parser("lexical_attachment", 16, 1, vocabulary)
            parameters = dict(model.named_parameters())
            frozen_names = [name for name in parameters if name.startswith(("encoder.", "heads."))]
            frozen_parameter, role_parameter = parameters["heads.kind.weight"], parameters["link_queries.weight"]
            milestones, rates, draws, optimizers = {}, [], [], []
            def make_optimizer(*args, **kwargs):
                optimizer = adamw(*args, **kwargs)
                optimizers.append(optimizer)
                original_step = optimizer.step
                def step(*step_args, **step_kwargs):
                    original_step(*step_args, **step_kwargs)
                    rates.append(optimizer.param_groups[0]["lr"])
                    number = len(rates)
                    if number in (299, 300, 301, 303):
                        milestones[number] = {
                            "frozen_hash": semantic_experiment.parameter_subset_hash(model, frozen_names),
                            "frozen_step": optimizer.state[frozen_parameter]["step"].item(),
                            "role_step": optimizer.state[role_parameter]["step"].item(),
                            "frozen_trainable": frozen_parameter.requires_grad,
                        }
                optimizer.step = step
                return optimizer
            def forward(surface, **options):
                draws.append(tuple(surface.values[:, 0].tolist()))
                self.assertTrue(options["return_token_features"])
                # A tiny parameter objective exercises the actual optimizer and
                # training loop without fitting or scoring a research model.
                loss = sum((parameter.flatten()[0] - 1).square()
                           for parameter in model.parameters() if parameter.requires_grad)
                return {"unit_loss": loss, "token_features": loss.expand(len(surface.ids), 1, 1, 1)}
            with patch.object(torch.optim, "AdamW", side_effect=make_optimizer), \
                    patch.object(model, "forward", side_effect=forward), \
                    patch.object(semantic_experiment, "parser_loss", side_effect=lambda output, *_: output["unit_loss"]), \
                    patch.object(semantic_experiment, "cache_parser", side_effect=[
                        {"event_accuracy": 0., "loss": 1.}, {"event_accuracy": 1., "loss": 0.}]), \
                    redirect_stdout(io.StringIO()):
                result = semantic_experiment.train_component(model, "parser", train, train, settings, 94622,
                    paired=paired, freeze_after=boundary, capture_step=300)
            self.assertEqual(len(optimizers), 1)
            self.assertEqual(result["selected_step"], 303)
            self.assertEqual((result["examples_seen"], result["parent_draws"]), (1212, 606))
            self.assertTrue(all(values[0] == values[1] and values[2] == values[3] for values in draws))
            runs.append((result, milestones, rates, draws))
        control, preserved = runs
        self.assertEqual(control[0]["prefix"], preserved[0]["prefix"])
        self.assertEqual(control[0]["original_bytes_seen"], preserved[0]["original_bytes_seen"])
        self.assertEqual(control[2:], preserved[2:])
        self.assertNotEqual(preserved[1][299]["frozen_hash"], preserved[1][300]["frozen_hash"])
        for step in (301, 303):
            self.assertEqual(preserved[1][300]["frozen_hash"], preserved[1][step]["frozen_hash"])
            self.assertEqual(preserved[1][step]["frozen_step"], 300.)
            self.assertEqual(preserved[1][step]["role_step"], float(step))
            self.assertFalse(preserved[1][step]["frozen_trainable"])
            self.assertNotEqual(control[1][300]["frozen_hash"], control[1][step]["frozen_hash"])
        self.assertTrue(preserved[1][300]["frozen_trainable"])
        report = preserved[0]["preservation"]
        self.assertEqual(report["boundary_frozen_sha256"], report["final_frozen_sha256"])
        self.assertEqual(report["boundary_frozen_sha256"], report["selected_frozen_sha256"])
        self.assertFalse(report["optimizer_reset"])
        self.assertFalse(report["learning_rate_schedule_changed"])

    def test_attachment_control_routing_keeps_architecture_and_excludes_cosine_objective(self):
        import semantic_experiment

        arguments = ["semantic_experiment.py", "--protocol", "attachment", "--parser-type",
                     "lexical_attachment", "--width", "192", "--parser-steps", "1500",
                     "--balance-quantities", "--retain-stability", "--train-size", "6000",
                     "--valid-size", "400", "--wording-test-size", "600", "--pair-consistency", "0"]
        with patch("sys.argv", arguments), patch.object(semantic_experiment, "run_interpretation") as run:
            semantic_experiment.main()
            run.assert_called_once()
            settings = run.call_args.args[0]
        self.assertEqual(settings.source_weight, .1)
        self.assertEqual(semantic_experiment.interpretation_controls(settings),
                         [("paired_ce", "lexical_attachment", 192, "paired_ce")])
        plain = copy.deepcopy(settings)
        plain.source_weight = 0.
        self.assertEqual(semantic_experiment.interpretation_controls(plain), [])
        for changed in ({"pair_consistency": .1}, {"parser_type": "lexical_constrained"}):
            invalid = copy.deepcopy(settings)
            for field, value in changed.items():
                setattr(invalid, field, value)
            with self.subTest(changed=changed), patch.object(semantic_experiment, "load_checkpoint") as load:
                with self.assertRaises(ValueError):
                    semantic_experiment.run_interpretation(invalid)
                load.assert_not_called()
        for value in ("nan", "inf", "-0.1"):
            with patch("sys.argv", [*arguments, "--source-weight", value]), \
                    patch("sys.stderr", new=io.StringIO()), \
                    patch.object(semantic_experiment, "run_interpretation") as run:
                with self.assertRaises(SystemExit) as error:
                    semantic_experiment.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_oracle_source_intervention_changes_only_annotated_roles_and_cannot_replace_release_metrics(self):
        import semantic_experiment
        import semantic_model

        example = SemanticPolarityTaskTests.fixture("Alice does not give Bob 3 coins.", active=0)
        vocabulary = semantic_model.fit_lexicon([example.text])
        surface = semantic_model.prepare_texts([example.text], representation="lexical", vocabulary=vocabulary)
        labels = semantic_experiment.prepare_source_labels([example], surface)
        parser = semantic_experiment.interpretation_parser("lexical_attachment", 16, 1, vocabulary).eval()
        with torch.no_grad():
            output = parser(surface, return_source_links=True)
        rows = semantic_model.oracle_tensor([example])
        for name, column, classes in (("kind", 0, 6), ("a", 1, 5), ("b", 2, 5), ("active", 4, 2)):
            output[name] = torch.full((*rows.shape[:2], classes), -20.)
            targets = rows[:, :, column].long().remainder(classes)
            output[name].scatter_(-1, targets.unsqueeze(-1), 20.)
        # Keep an activity error to prove that a gold source frame only
        # intervenes on arguments, rather than substituting a gold event.
        for name, target in (("a", 1), ("b", 0), ("active", 1)):
            output[name][0, 4] = -20.
            output[name][0, 4, target] = 20.
        sender, recipient = labels.mentions[0, 4].tolist()
        output["link_log_probs"][0, 4] = float("-inf")
        output["link_log_probs"][0, 4, 0, :, recipient] = 0.
        output["link_log_probs"][0, 4, 1, :, sender] = 0.
        snapshot = {name: value.clone() for name, value in output.items() if isinstance(value, torch.Tensor)}
        ordinary = semantic_model.parsed_rows(output, surface)
        predicates_only = semantic_experiment.oracle_source_rows(output, surface, labels)
        both = semantic_experiment.oracle_source_rows(output, surface, labels, replace_links=True)
        torch.testing.assert_close(predicates_only, ordinary, atol=0, rtol=0)
        expected = ordinary.clone()
        expected[0, 4, 1:3] = rows[0, 4, 1:3]
        torch.testing.assert_close(both, expected, atol=0, rtol=0)
        self.assertEqual(both[0, 4, 4].item(), 1.)
        self.assertFalse(torch.equal(both, rows))
        for name, value in snapshot.items():
            torch.testing.assert_close(output[name], value, atol=0, rtol=0)
        trials = self.good_trials()
        trials[0]["test"]["wording"]["parser"]["per_task"]["accounting"]["program_accuracy"] = 0.
        trials[0]["source_diagnostics"] = {"wording": {"oracle_interventions": {
            "gold_predicates_and_links": {"parser": {"program_accuracy": 1.}}}}}
        gates = semantic_experiment.interpretation_gates(trials, strict=True, require_retention=True)
        self.assertFalse(gates["passed"])
        self.assertEqual(gates["failures"][0]["metric"], "program")

    def test_source_label_packing_aligns_every_predicate_and_semantic_role_with_lexical_positions(self):
        import semantic_experiment
        import semantic_model
        import semantic_tasks

        examples = [
            SemanticPolarityTaskTests.fixture("Bob receives from Alice the 3 coins that are given."),
            SemanticPolarityTaskTests.fixture("Bob does not receive from Alice the 3 coins that are not given.", 0),
            SemanticPolarityTaskTests.fixture("0 coins are given by Alice to Bob.", amount=0),
            next(example for example in semantic_tasks.make_binding_split(4, 94611) if example.task == "relations"),
        ]
        vocabulary = semantic_model.fit_lexicon(example.text for example in examples)
        surface = semantic_model.prepare_texts([example.text for example in examples], representation="lexical",
                                                vocabulary=vocabulary)
        labels = semantic_experiment.prepare_source_labels(examples, surface)
        self.assertEqual(labels.triggers.shape, surface.ids.shape)
        self.assertEqual(labels.triggers.dtype, torch.bool)
        self.assertEqual(labels.mentions.shape, (*surface.ids.shape[:2], 2))
        self.assertEqual(labels.mentions.dtype, torch.long)
        expected = torch.zeros_like(labels.triggers)
        for world, positions in enumerate(((2, 10), (4, 13), (4,))):
            expected[world, 4, list(positions)] = True
        self.assertTrue(torch.equal(labels.triggers, expected))
        self.assertEqual(labels.mentions[:3, 4].tolist(), [[4, 1], [6, 1], [6, 8]])
        self.assertTrue((labels.mentions[~expected.any(-1)] == -1).all().item())
        self.assertFalse(labels.triggers[~surface.token_mask].any().item())
        self.assertEqual({key: labels.report[key] for key in
                          ("stage", "worlds", "transfer_clauses", "predicates", "semantic_argument_links",
                           "dual_predicate_clauses")},
                         {"stage": "train", "worlds": 4, "transfer_clauses": 3, "predicates": 5,
                          "semantic_argument_links": 10, "dual_predicate_clauses": 2})
        self.assertEqual(labels.report["examples_sha256"], semantic_tasks.split_fingerprint(examples))
        self.assertFalse(hasattr(surface, "source"))
        self.assertFalse(hasattr(surface, "triggers"))
        diagnostic = semantic_experiment.prepare_source_labels(examples, surface, stage="development")
        self.assertEqual(diagnostic.report["stage"], "development")
        self.assertTrue(torch.equal(diagnostic.triggers, labels.triggers))
        bad_entities = surface.entity_ids.clone()
        bad_entities[0, 4, [1, 4]] = bad_entities[0, 4, [4, 1]]
        invalid = (dataclasses.replace(surface, entity_ids=None),
                   dataclasses.replace(surface, entity_ids=bad_entities),
                   dataclasses.replace(surface, sentence_mask=torch.zeros_like(surface.sentence_mask)))
        for changed in invalid:
            with self.assertRaises(ValueError):
                semantic_experiment.prepare_source_labels(examples, changed)
        with self.assertRaises(ValueError):
            semantic_experiment.prepare_source_labels(examples[:-1], surface)

    def test_observed_polarity_probes_require_complete_unique_known_keys_including_zero_amounts(self):
        import semantic_experiment
        import semantic_tasks

        nonzero = SemanticPolarityTaskTests.fixture()
        zero = SemanticPolarityTaskTests.fixture("0 coins are given by Alice to Bob.", amount=0)
        extra = SemanticPolarityTaskTests.fixture("5 coins are given by Alice to Bob.", amount=5)
        development = {"wording": [nonzero], "combined": [zero, extra]}
        known = [{"shift": shift, "origin_key": semantic_tasks.interpretation_world_key(example.text),
                  "clause_index": 4} for shift, example in (("wording", nonzero), ("combined", zero))]
        probes, observed = semantic_experiment.polarity_probe_sets(development, known)
        self.assertEqual(len(observed), 2)
        self.assertEqual({(group.origin_key, group.clause_index) for group in observed},
                         {(tuple(case["origin_key"]), case["clause_index"]) for case in known})
        self.assertEqual(sorted(group.provenance.amount for group in observed), [0, 3])
        self.assertEqual([group.provenance.amount for group in probes["combined"][0]], [5])
        absent = {**known[0], "origin_key": semantic_tasks.interpretation_world_key(
            SemanticPolarityTaskTests.fixture("4 coins are given by Alice to Bob.", amount=4).text)}
        for cases in ([], known + [known[0]], known + [absent],
                      [known[0], {**known[1], "shift": "numbers"}],
                      [known[0], {**known[1], "clause_index": 3}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                semantic_experiment.polarity_probe_sets(development, cases)
        repeated = {**development, "wording": [nonzero, nonzero]}
        with self.assertRaises(ValueError):
            semantic_experiment.polarity_probe_sets(repeated, known)

    def test_polarity_training_rejects_development_final_origin_overlap_before_preparing_tensors(self):
        import semantic_experiment
        import semantic_model
        import semantic_tasks

        parent = SemanticPolarityTaskTests.fixture()
        inactive_parent, _ = semantic_tasks.polarity_pair(parent, 4)
        vocabulary = semantic_model.fit_lexicon([parent.text, inactive_parent.text])
        development = SemanticPolarityTaskTests.fixture("4 coins are given by Alice to Bob.", amount=4)
        negative, _ = semantic_tasks.polarity_pair(development, 4, "development")
        query = "what is Alice's balance now?"
        final = dataclasses.replace(negative, sentences=negative.sentences[:-1] + (query,),
            text=" ".join(negative.sentences[:-1] + (query,)), answer="9",
            rows=negative.rows[:-1] + ((semantic_tasks.QUERY_BALANCE, 0, -1, 0, 1),))
        self.assertNotEqual(semantic_tasks.interpretation_world_key(development.text),
                            semantic_tasks.interpretation_world_key(final.text))
        self.assertEqual(semantic_tasks.polarity_world_key(development, "development"),
                         semantic_tasks.polarity_world_key(final, "final"))
        with patch.object(semantic_experiment, "prepare") as prepare:
            with self.assertRaisesRegex(ValueError, "(?i)development.*final.*overlap"):
                semantic_experiment.prepare_polarity_training([parent], {"wording": [development]},
                                                              {"combined": [final]}, "cpu", vocabulary)
            prepare.assert_not_called()

    def test_paired_training_preserves_parent_partitions_and_rejects_polarity_equivalent_holdouts(self):
        import semantic_experiment
        import semantic_model
        import semantic_tasks

        positive = SemanticPolarityTaskTests.fixture()
        negative, _ = semantic_tasks.polarity_pair(positive, 4)
        relation = next(example for example in semantic_tasks.make_binding_split(4, 94531) if example.task == "relations")
        parents = [negative, positive, relation]
        vocabulary = semantic_model.fit_lexicon(example.text for example in parents)
        development = {"fixture": [SemanticPolarityTaskTests.fixture("4 coins are given by Alice to Bob.", amount=4)]}
        tests = {"fixture": [SemanticPolarityTaskTests.fixture("5 coins are given by Alice to Bob.", amount=5)]}
        paired = semantic_experiment.prepare_polarity_training(parents, development, tests, "cpu", vocabulary)
        self.assertEqual(paired.data.examples, [negative, positive, negative, positive, relation, relation])
        self.assertEqual(paired.report["parents"], 3)
        self.assertEqual(paired.report["views"], 6)
        self.assertEqual(paired.report["paired_parents"], 2)
        self.assertEqual(paired.report["identity_parents"], 1)
        self.assertEqual(paired.report["original_parent_sha256"], semantic_tasks.split_fingerprint(parents))
        self.assertEqual(paired.report["views_sha256"], semantic_tasks.split_fingerprint(paired.data.examples))
        self.assertEqual(paired.report["origin_overlap"], 0)
        self.assertTrue((paired.alignment[2] == -1).all().item())
        # A holdout with another polarity and another query still belongs to
        # the same originating world and must not become a training view.
        query = "what is Alice's balance now?"
        conflicting = dataclasses.replace(positive, sentences=positive.sentences[:-1] + (query,),
            text=" ".join(positive.sentences[:-1] + (query,)), answer="6",
            rows=positive.rows[:-1] + ((semantic_tasks.QUERY_BALANCE, 0, -1, 0, 1),))
        with self.assertRaisesRegex(ValueError, "origin overlap"):
            semantic_experiment.prepare_polarity_training([negative, relation], {"fixture": [conflicting]}, {},
                                                          "cpu", vocabulary)
        incomplete_vocabulary = semantic_model.fit_lexicon([positive.text])
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            semantic_experiment.prepare_polarity_training([positive], {}, {}, "cpu", incomplete_vocabulary)

    def test_polarity_gates_reject_consistent_wrong_roles_missing_results_and_failed_seeds(self):
        import semantic_experiment

        fields = ("all_correct", "all_programs_correct", "target_row_accuracy", "role_consistency_accuracy")
        good = {"per_task": {"accounting": {field: 1. for field in fields}}}
        trials = [{"seed": seed, "development": {shift: {"direct_polarity": copy.deepcopy(good)}
                   for shift in ("wording", "combined")}, "observed_polarity": copy.deepcopy(good)}
                  for seed in (94541, 94542, 94543)]
        self.assertTrue(semantic_experiment.polarity_gates(trials)["passed"])
        with self.assertRaises(ValueError):
            semantic_experiment.polarity_gates([])
        for metric in fields:
            changed = copy.deepcopy(trials)
            changed[1]["development"]["combined"]["direct_polarity"]["per_task"]["accounting"][metric] = .94
            result = semantic_experiment.polarity_gates(changed)
            self.assertFalse(result["passed"])
            self.assertEqual([(row["seed"], row["metric"]) for row in result["failures"]], [(94542, metric)])
        changed = copy.deepcopy(trials)
        changed[0]["observed_polarity"]["per_task"]["accounting"]["target_row_accuracy"] = .99
        self.assertFalse(semantic_experiment.polarity_gates(changed)["passed"])
        for metric in fields:
            changed = copy.deepcopy(trials)
            del changed[0]["development"]["wording"]["direct_polarity"]["per_task"]["accounting"][metric]
            with self.subTest(missing=metric), self.assertRaises((KeyError, ValueError)):
                semantic_experiment.polarity_gates(changed)
        for value in (float("nan"), float("inf"), -.01, 1.01):
            changed = copy.deepcopy(trials)
            changed[0]["observed_polarity"]["per_task"]["accounting"]["role_consistency_accuracy"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                semantic_experiment.polarity_gates(changed)

    def test_polarity_alignment_tracks_identity_and_occurrence_across_inserted_negation(self):
        import semantic_experiment
        import semantic_model

        negative = "Alice does not give Bob 3 coins and Bob does not receive the coins from Alice."
        positive = "Alice gives Bob 3 coins and Bob receives the coins from Alice."
        texts = [negative, positive, "Alice starts with 3 coins.", "Alice starts with 3 coins."]
        vocabulary = semantic_model.fit_lexicon(texts)
        surface = semantic_model.prepare_texts(texts, representation="lexical", vocabulary=vocabulary)
        alignment = semantic_experiment.polarity_alignment(surface, [0, -1])
        self.assertEqual(alignment.dtype, torch.long)
        self.assertEqual(alignment[0].tolist(), [[0, 1, 1], [0, 16, 12], [0, 5, 3], [0, 9, 7]])
        self.assertTrue((alignment[1] == -1).all().item())
        for invalid in ([], [0], [0, -2], [1, -1], [[0, -1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                semantic_experiment.polarity_alignment(surface, invalid)
        missing_entity = semantic_model.prepare_texts(
            [negative, positive.replace("from Alice", "from")], representation="lexical", vocabulary=vocabulary)
        with self.assertRaises(ValueError):
            semantic_experiment.polarity_alignment(missing_entity, [0])
        with self.assertRaises(ValueError):
            semantic_experiment.polarity_alignment(dataclasses.replace(surface, entity_ids=None), [0, -1])

    def test_final_polarity_gates_require_each_direct_metric_without_reusing_observed_scores(self):
        import semantic_experiment

        fields = ("all_correct", "all_programs_correct", "target_row_accuracy", "role_consistency_accuracy")
        # Final gates operate on fresh direct probes, with the exact 95% boundary.
        good = {"per_task": {"accounting": {field: .95 for field in fields}}}
        trials = [{"seed": seed, "test": {shift: {"direct_polarity": copy.deepcopy(good)}
                   for shift in ("wording", "combined")}} for seed in (94551, 94552, 94553)]
        self.assertTrue(semantic_experiment.polarity_gates(trials, "test")["passed"])
        for shift in ("wording", "combined"):
            for metric in fields:
                changed = copy.deepcopy(trials)
                changed[2]["test"][shift]["direct_polarity"]["per_task"]["accounting"][metric] = .949
                result = semantic_experiment.polarity_gates(changed, "test")
                self.assertFalse(result["passed"])
                self.assertEqual([(row["seed"], row["shift"], row["metric"], row["required"])
                                  for row in result["failures"]],
                                 [(94553, f"{shift}_direct_polarity", metric, .95)])
            missing = copy.deepcopy(trials)
            del missing[0]["test"][shift]["direct_polarity"]
            with self.assertRaises((KeyError, ValueError)):
                semantic_experiment.polarity_gates(missing, "test")

    def test_polarity_feature_loss_uses_only_matched_mentions_and_has_differentiable_empty_behavior(self):
        import semantic_experiment

        links = torch.tensor([[[0, 1, 2], [0, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]])
        values = torch.full((4, 1, 5, 2), 1000.)
        for negative, positive in ((1, 2), (3, 4)):
            values[0, 0, negative], values[1, 0, positive] = torch.tensor([1., 0.]), torch.tensor([1., 0.])
        features = values.clone().requires_grad_()
        self.assertEqual(semantic_experiment.polarity_feature_loss({"token_features": features}, links).item(), 0.)
        values[1, 0, 2] = torch.tensor([0., 1.])
        features = values.clone().requires_grad_()
        loss = semantic_experiment.polarity_feature_loss({"token_features": features}, links)
        self.assertEqual(loss.item(), .5)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        allowed = torch.zeros(features.shape[:-1], dtype=torch.bool)
        allowed[0, 0, [1, 3]], allowed[1, 0, [2, 4]] = True, True
        self.assertTrue((features.grad[~allowed] == 0).all().item())
        features = values.clone().requires_grad_()
        empty = semantic_experiment.polarity_feature_loss({"token_features": features}, torch.full_like(links, -1))
        self.assertEqual(empty.item(), 0.)
        self.assertTrue(empty.requires_grad)
        empty.backward()
        self.assertTrue((features.grad == 0).all().item())
        invalid_links = [links.float(), links[:, :, :2], links[:1]]
        for replacement in ((1, 1, 2), (0, 5, 2), (-1, 1, -1), (-2, -2, -2)):
            invalid = links.clone()
            invalid[0, 0] = torch.tensor(replacement)
            invalid_links.append(invalid)
        for invalid in invalid_links:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                semantic_experiment.polarity_feature_loss({"token_features": features}, invalid)
        with self.assertRaises(ValueError):
            semantic_experiment.polarity_feature_loss({}, links)

    def test_unbalanced_control_keeps_candidate_architecture_and_routes_only_its_training_view(self):
        import semantic_experiment
        from types import SimpleNamespace

        settings = {"protocol": "binding", "parser_type": "lexical_constrained", "width": 48,
                    "comparison_parsers": ["lexical_joint"], "comparison_widths": [96],
                    "balance_quantities": True, "comparison_unbalanced": True}
        controls = semantic_experiment.interpretation_controls(SimpleNamespace(**settings))
        self.assertEqual(controls, [
            ("lexical_joint", "lexical_joint", 48, "candidate"),
            ("lexical_constrained_width96", "lexical_constrained", 96, "candidate"),
            ("lexical_constrained_unbalanced", "lexical_constrained", 48, "unbalanced"),
        ])
        legacy = {key: value for key, value in settings.items() if key != "comparison_unbalanced"}
        self.assertEqual(semantic_experiment.interpretation_controls(SimpleNamespace(**legacy)), controls[:2])
        for changed in ({"protocol": "interpretation"}, {"protocol": "scaling"}, {"balance_quantities": False}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                semantic_experiment.interpretation_controls(SimpleNamespace(**{**settings, **changed}))
        for arguments in (("--protocol", "binding", "--comparison-unbalanced"),
                          ("--protocol", "scaling", "--balance-quantities", "--comparison-unbalanced")):
            with patch("sys.argv", ["semantic_experiment.py", *arguments]), \
                    patch("sys.stderr", new=io.StringIO()), \
                    patch.object(semantic_experiment, "run_interpretation") as run:
                with self.assertRaises(SystemExit) as error:
                    semantic_experiment.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
        arguments = ["semantic_experiment.py", "--protocol", "binding", "--balance-quantities",
                     "--comparison-unbalanced", "--retain-stability", "--train-size", "6000"]
        with patch("sys.argv", arguments), patch.object(semantic_experiment, "run_interpretation") as run:
            semantic_experiment.main()
            run.assert_called_once()
            self.assertTrue(run.call_args.args[0].comparison_unbalanced)

    def test_polarity_cli_routes_diagnosis_and_matched_paired_control_without_external_actions(self):
        import semantic_experiment

        arguments = ["semantic_experiment.py", "--protocol", "polarity", "--parser-type",
                     "lexical_constrained", "--width", "192", "--parser-steps", "1500",
                     "--balance-quantities", "--retain-stability", "--train-size", "6000",
                     "--valid-size", "400", "--wording-test-size", "600"]
        with patch("sys.argv", arguments), \
                patch.object(semantic_experiment, "run_interpretation") as run, \
                patch.object(semantic_experiment, "polarity_diagnosis") as diagnose:
            semantic_experiment.main()
            run.assert_called_once()
            diagnose.assert_not_called()
            settings = run.call_args.args[0]
        self.assertEqual(semantic_experiment.interpretation_controls(settings),
                         [("paired_ce", "lexical_constrained", 192, "paired_ce")])
        plain = copy.deepcopy(settings)
        plain.pair_consistency = 0.
        self.assertEqual(semantic_experiment.interpretation_controls(plain), [])
        with patch("sys.argv", [*arguments, "--diagnose-only"]), \
                patch.object(semantic_experiment, "run_interpretation") as run, \
                patch.object(semantic_experiment, "polarity_diagnosis") as diagnose:
            semantic_experiment.main()
            diagnose.assert_called_once()
            run.assert_not_called()
        for extra in (("--pair-consistency", "nan"), ("--pair-consistency", "inf"),
                      ("--pair-consistency", "-0.1"), ("--comparison-unbalanced",),
                      ("--diagnose-only", "--benchmark-only"),
                      ("--diagnose-only", "--regression-only"),
                      ("--diagnose-only", "--protocol", "binding")):
            with self.subTest(extra=extra), patch("sys.argv", [*arguments, *extra]), \
                    patch("sys.stderr", new=io.StringIO()), \
                    patch.object(semantic_experiment, "run_interpretation") as run, \
                    patch.object(semantic_experiment, "polarity_diagnosis") as diagnose:
                with self.assertRaises(SystemExit) as error:
                    semantic_experiment.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
                diagnose.assert_not_called()

    def test_polarity_runner_rejects_changed_architecture_data_and_exposure_before_checkpoint_access(self):
        import semantic_experiment
        from types import SimpleNamespace

        fixed = dict(protocol="polarity", parser_type="lexical_constrained", width=192,
                     parser_steps=1500, batch_size=32, balance_quantities=True, retain_stability=True,
                     train_size=6000, valid_size=400, development_size=200, test_size=600,
                     wording_test_size=600, retention_test_size=160, comparison_parsers=[],
                     comparison_widths=[], comparison_unbalanced=False)
        changes = dict(parser_type="lexical_joint", width=96, parser_steps=3000, batch_size=64,
                       balance_quantities=False, retain_stability=False, train_size=6004, valid_size=404,
                       development_size=204, test_size=604, wording_test_size=604, retention_test_size=164,
                       comparison_parsers=["lexical_joint"], comparison_widths=[96], comparison_unbalanced=True)
        for field, value in changes.items():
            with self.subTest(field=field), patch.object(semantic_experiment, "load_checkpoint") as load:
                with self.assertRaises(ValueError):
                    semantic_experiment.run_interpretation(SimpleNamespace(**{**fixed, field: value}))
                load.assert_not_called()

    def test_quantity_balance_arms_keep_exposure_and_development_exclusions_identical(self):
        import semantic_experiment
        import semantic_tasks

        def fixtures(count, label):
            return [semantic_tasks.Example(f"{label} {index}.", (), (), "0", "accounting")
                    for index in range(count)]

        retained, baseline, balanced = (fixtures(count, name) for count, name in
                                       ((3000, "retained"), (6000, "baseline"), (6000, "balanced")))
        universe = {example.text for example in retained + baseline + balanced}
        snapshots = []
        def generate(count, seed, shift="iid", exclude=(), stage="train", balance_quantities=False):
            if seed == 110001:
                self.assertEqual(count, 6000)
                self.assertEqual(set(exclude), {example.text for example in retained})
                return balanced if balance_quantities else baseline
            self.assertTrue(universe.issubset(exclude))
            self.assertFalse(balance_quantities)
            snapshots.append((seed, shift, frozenset(exclude)))
            return fixtures(count, f"development {seed} {shift}")

        def retention(count, seed, exclude=(), candidate_count=None):
            self.assertTrue(universe.issubset(exclude))
            snapshots.append((seed, "retention", frozenset(exclude)))
            return fixtures(count, "retention development")

        results, exclusions = [], []
        with patch.object(semantic_experiment, "make_stability_split", return_value=retained), \
                patch.object(semantic_tasks, "make_binding_split", side_effect=generate), \
                patch.object(semantic_experiment, "retention_split", side_effect=retention):
            for balance_quantities in (False, True):
                snapshots.clear()
                results.append(semantic_experiment.binding_data(6000, 8, 8, balance_quantities))
                exclusions.append(list(snapshots))
        self.assertEqual(results[0][0], retained + baseline)
        self.assertEqual(results[1][0], retained + balanced)
        self.assertEqual([len(result[0]) for result in results], [9000, 9000])
        self.assertEqual(results[0][1:], results[1][1:])
        self.assertEqual(exclusions[0], exclusions[1])

    @staticmethod
    def good_trials():
        tasks = ("accounting", "relations")
        shifts = ("iid", "names", "wording", "length", "numbers", "composition", "combined", "retention")
        good = {"parser": {"per_task": {task: {"program_accuracy": 1.} for task in tasks}},
                **{mode: {"per_task": {task: {"exact_accuracy": 1.} for task in tasks}}
                   for mode in ("parsed", "executor")},
                **{mode: {"per_task": {task: {"all_correct": 1., "all_programs_correct": 1.}
                                      for task in tasks}} for mode in ("all_queries", "contrasts")}}
        return [{"seed": seed, "test": {shift: copy.deepcopy(good) for shift in shifts}}
                for seed in (94211, 94212, 94213)]

    def test_binding_gates_require_faithful_expansions_and_stricter_combined_accuracy(self):
        import semantic_experiment

        trials = self.good_trials()
        gate = semantic_experiment.interpretation_gates(trials, strict=True, require_retention=True)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["thresholds"]["combined"], .95)
        self.assertEqual(semantic_experiment.interpretation_gates(trials)["thresholds"]["combined"], .90)
        for mode, field, metric in (("all_queries", "all_programs_correct", "all_queries_programs"),
                                   ("contrasts", "all_correct", "contrast_answers"),
                                   ("contrasts", "all_programs_correct", "contrast_programs")):
            changed = copy.deepcopy(trials)
            changed[1]["test"]["combined"][mode]["per_task"]["accounting"][field] = .94
            self.assertTrue(semantic_experiment.interpretation_gates(changed)["passed"])
            result = semantic_experiment.interpretation_gates(changed, strict=True)
            self.assertFalse(result["passed"])
            self.assertEqual([(failure["seed"], failure["task"], failure["metric"])
                              for failure in result["failures"]], [(94212, "accounting", metric)])
        changed = copy.deepcopy(trials)
        changed[2]["test"]["combined"]["parser"]["per_task"]["relations"]["program_accuracy"] = .94
        self.assertTrue(semantic_experiment.interpretation_gates(changed)["passed"])
        self.assertFalse(semantic_experiment.interpretation_gates(changed, strict=True)["passed"])
        for mode, field in (("contrasts", None), ("all_queries", "all_programs_correct"),
                            ("contrasts", "all_programs_correct")):
            changed = copy.deepcopy(trials)
            if field is None:
                del changed[0]["test"]["wording"][mode]
            else:
                del changed[0]["test"]["wording"][mode]["per_task"]["relations"][field]
            with self.subTest(mode=mode, field=field), self.assertRaises((ValueError, KeyError)):
                semantic_experiment.interpretation_gates(changed, strict=True)
        for value in (float("nan"), float("inf"), -.01, 1.01):
            changed = copy.deepcopy(trials)
            changed[0]["test"]["wording"]["contrasts"]["per_task"]["relations"]["all_programs_correct"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                semantic_experiment.interpretation_gates(changed, strict=True)

    def test_exact_program_gate_boundary_uses_integer_hits_without_float32_rounding(self):
        import semantic_experiment
        from types import SimpleNamespace

        for count, shift, errors in ((300, "wording", ((15, True), (16, False))),
                                      (200, "iid", ((2, True), (3, False)))):
            rows = torch.zeros((count, 1, 5))
            rows[:, :, 0] = 4
            data = SimpleNamespace(rows=rows, predicted=rows.clone(),
                surface=SimpleNamespace(sentence_mask=torch.ones((count, 1), dtype=torch.bool)),
                examples=[SimpleNamespace(task="accounting") for _ in range(count)])
            for incorrect, expected_pass in errors:
                data.predicted = rows.clone()
                data.predicted[:incorrect, 0, 1] = 1
                metrics = semantic_experiment.metric_parser(data)
                expected = (count - incorrect) / count
                self.assertEqual(metrics["program_accuracy"], expected)
                self.assertEqual(metrics["per_task"]["accounting"]["program_accuracy"], expected)
                trials = self.good_trials()
                trials[0]["test"][shift]["parser"]["per_task"]["accounting"] = metrics["per_task"]["accounting"]
                self.assertEqual(semantic_experiment.interpretation_gates(trials, strict=True)["passed"], expected_pass)


# Lexical parser architectures and complete-program metrics
class SemanticLexicalModelTests(unittest.TestCase):
    """Exercise lexical parser variants, learned links, and release gates."""

    @classmethod
    def setUpClass(cls):
        import semantic_model

        cls.model = semantic_model
        cls.texts = ["Alice starts with 9 coins. Bob starts with 2 coins. Alice gives Bob 3 coins. "
                     "what is Bob's balance now?",
                     "Alice is before Bob. Bob is before Clara. Clara is before David. is Alice before David?"]
        cls.vocabulary = semantic_model.fit_lexicon(cls.texts)

    def make_parser(self, pooling="pointer"):
        torch.manual_seed(84081)
        return self.model.LexicalParser(16, 1, self.vocabulary,
                                       pooling="pointer" if pooling in ("relative", "constrained", "local", "joint", "evidence", "attachment") else pooling,
                                       relative_positions=pooling in ("relative", "constrained", "local", "joint", "evidence", "attachment"),
                                       constrained_roles=pooling in ("constrained", "local", "joint", "evidence", "attachment"),
                                       local_roles=pooling == "local", joint_roles=pooling == "joint",
                                       evidence_roles=pooling == "evidence", attachment_roles=pooling == "attachment").eval()

    def surface(self, texts=None):
        return self.model.prepare_texts(self.texts if texts is None else texts,
                                       representation="lexical", vocabulary=self.vocabulary)

    def test_lexicon_is_training_only_frozen_and_validated(self):
        self.assertEqual(self.vocabulary, self.model.fit_lexicon(reversed(self.texts)))
        before = dict(self.vocabulary)
        heldout = self.surface([self.texts[0].replace("gives", "remits")])
        self.assertNotIn("remits", self.vocabulary)
        self.assertIn(2, heldout.ids[heldout.token_mask].tolist())
        self.assertEqual(self.vocabulary, before)
        self.assertEqual(self.model.lexical_tokens("e0 gives e1 -33 coins."),
                         ("<entity:0>", "gives", "<entity:1>", "<quantity>", "coins", "."))
        for invalid in ({}, {**before, "<entity:0>": 3}, {**before, "extra": len(before) + 1}):
            with self.assertRaises(ValueError):
                self.model.prepare_texts(self.texts, representation="lexical", vocabulary=invalid)
        with self.assertRaises(ValueError):
            self.model.fit_lexicon([])

    def test_lexical_parser_has_numeric_isolation_and_local_role_candidates(self):
        texts = [self.texts[0], re.sub(r"\d+", lambda match:
                 {"9": "-16777216", "2": "0", "3": "16777216"}[match.group()], self.texts[0])]
        surface = self.surface(texts)
        torch.testing.assert_close(surface.ids[0], surface.ids[1], atol=0, rtol=0)
        torch.testing.assert_close(surface.entity_ids[0], surface.entity_ids[1], atol=0, rtol=0)
        self.assertEqual(surface.values[1].tolist(), [-16777216., 0., 16777216., 0.])
        with self.assertRaises(ValueError):
            self.surface([texts[0].replace("9", "16777217")])
        for pooling in ("clause", "pointer", "relative", "constrained", "local", "joint", "evidence", "attachment"):
            output = self.make_parser(pooling)(surface)
            for name, value in output.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value[0], value[1], atol=2e-6, rtol=2e-5)
            for row in range(surface.ids.shape[1]):
                present = set(surface.entity_ids[0, row].tolist()) - {-1}
                for role in ("a", "b"):
                    for entity in set(range(4)) - present:
                        self.assertLess(float(output[role][0, row, entity].detach()), -1000)
                    self.assertGreater(float(output[role][0, row, 4].detach()), -1000)
        selected = surface.take(torch.tensor([1]))
        torch.testing.assert_close(selected.entity_ids, surface.entity_ids[1:], atol=0, rtol=0)

    def test_pointer_identity_permutation_and_padding_preserve_meaning(self):
        for pooling in ("pointer", "relative", "constrained", "local", "joint", "evidence", "attachment"):
            with self.subTest(pooling=pooling):
                self.check_pointer_invariants(pooling)

    def check_pointer_invariants(self, pooling):
        parser, surface = self.make_parser(pooling), self.surface()
        permutation = torch.tensor([3, 0, 2, 1])
        entities, ids = surface.entity_ids.clone(), surface.ids.clone()
        valid = entities >= 0
        entities[valid] = permutation[entities[valid]]
        ids[valid] = entities[valid] + 4
        permuted = dataclasses.replace(surface, ids=ids, entity_ids=entities)
        with torch.no_grad():
            original, changed = parser(surface), parser(permuted)
            for field in ("kind", "active", "features"):
                torch.testing.assert_close(original[field], changed[field], atol=0, rtol=0)
            for field in ("a", "b"):
                torch.testing.assert_close(original[field][..., :4], changed[field][..., permutation], atol=0, rtol=0)
                torch.testing.assert_close(original[field][..., 4], changed[field][..., 4], atol=0, rtol=0)
            for index, text in enumerate(self.texts):
                single = parser(self.surface([text]))
                for field, tensor in original.items():
                    if isinstance(tensor, torch.Tensor):
                        torch.testing.assert_close(tensor[index], single[field][0], atol=2e-6, rtol=2e-5)
            padded = dataclasses.replace(surface,
                ids=F.pad(surface.ids, (0, 3), value=self.vocabulary["gives"]),
                token_mask=F.pad(surface.token_mask, (0, 3), value=False),
                entity_ids=F.pad(surface.entity_ids, (0, 3), value=3))
            extra = parser(padded)
            for field, tensor in original.items():
                if isinstance(tensor, torch.Tensor):
                    torch.testing.assert_close(tensor, extra[field], atol=2e-6, rtol=2e-5)
        examples = __import__("semantic_tasks").make_interpretation_split(8, 84082)
        batch = self.surface([example.text for example in examples])
        loss = self.model.parser_loss(parser(batch), self.model.oracle_tensor(examples), batch.sentence_mask)
        loss.backward()
        for name, parameter in parser.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                if "relative_bias" in name:
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)
        if pooling in ("relative", "constrained", "local", "joint", "evidence", "attachment"):
            self.assertIsNone(parser.encoder.position)
            self.assertFalse(any("position.weight" in name for name, _ in parser.named_parameters()))
            with self.assertRaises(ValueError):
                self.model.LexicalParser(16, 1, self.vocabulary, pooling="clause", relative_positions=True)

    def test_partial_lexical_checkpoint_preserves_parser_vocabulary_and_frozen_core(self):
        import semantic_experiment

        for pooling in ("clause", "pointer", "relative", "constrained", "local", "joint", "evidence", "attachment"):
            parser = self.make_parser(pooling)
            core = self.model.StableReasoner(16, 1).eval().requires_grad_(False)
            before = semantic_experiment.tensor_state_hash(core)
            metadata = {"config": {"width": 16, "local_layers": 1, "layers": 1, "architecture": "stable"}}
            with tempfile.TemporaryDirectory(prefix="text-vector-lexical-test-") as directory:
                path = Path(directory) / "model.pt"
                self.model.save_checkpoint(path, parser, None, core, None, metadata)
                restored, restored_metadata = self.model.load_checkpoint(path)
            self.assertEqual(set(restored), {"parser", "oracle"})
            self.assertEqual(restored["parser"].vocabulary, self.vocabulary)
            self.assertEqual(restored["parser"].pooling, "pointer" if pooling in ("relative", "constrained", "local", "joint", "evidence", "attachment") else pooling)
            self.assertEqual(restored["parser"].relative_positions, pooling in ("relative", "constrained", "local", "joint", "evidence", "attachment"))
            self.assertEqual(restored["parser"].constrained_roles, pooling in ("constrained", "local", "joint", "evidence", "attachment"))
            self.assertEqual(restored["parser"].local_roles, pooling == "local")
            self.assertEqual(restored["parser"].joint_roles, pooling == "joint")
            self.assertEqual(restored["parser"].evidence_roles, pooling == "evidence")
            self.assertEqual(restored["parser"].attachment_roles, pooling == "attachment")
            self.assertEqual(restored_metadata["config"]["parser_type"], "lexical_" + pooling)
            self.assertEqual(restored_metadata["config"]["representation"], "lexical")
            self.assertEqual(semantic_experiment.tensor_state_hash(restored["oracle"]), before)
            with torch.no_grad():
                expected, actual = parser(self.surface()), restored["parser"](self.surface())
                for field in expected:
                    torch.testing.assert_close(expected[field], actual[field], atol=0, rtol=0)
                torch.testing.assert_close(self.model.parsed_rows(expected, self.surface()),
                                           self.model.parsed_rows(actual, self.surface()), atol=0, rtol=0)
            self.assertEqual(semantic_experiment.tensor_state_hash(core), before)
            self.assertTrue(all(parameter.grad is None for parameter in core.parameters()))

    def test_local_role_scores_do_not_depend_on_contextual_transformer_weights(self):
        parser, surface = self.make_parser("local"), self.surface()
        with torch.no_grad():
            before = parser(surface)
            for parameter in parser.encoder.blocks.parameters():
                parameter.zero_()
            after = parser(surface)
        for field in ("a", "b"):
            torch.testing.assert_close(before[field], after[field], atol=0, rtol=0)
        self.assertFalse(torch.allclose(before["features"], after["features"]))
        self.assertFalse(torch.allclose(before["kind"], after["kind"]))
        parser = self.make_parser("local")
        output = parser(surface)
        # Role-only supervision must train raw word bindings without any path
        # through the contextual Transformer used for event type and polarity.
        loss = (F.cross_entropy(output["a"][:, 0], torch.tensor([0, 0])) +
                F.cross_entropy(output["b"][:, 0], torch.tensor([4, 1])))
        loss.backward()
        for name, parameter in parser.named_parameters():
            if name.startswith(("mention_roles.", "none_roles.", "encoder.embedding.")):
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)
            elif name.startswith("encoder.blocks."):
                self.assertIsNone(parameter.grad)
        with self.assertRaises(ValueError):
            self.model.LexicalParser(16, 1, self.vocabulary, relative_positions=True, local_roles=True)

    def test_joint_role_supervision_reaches_local_words_and_ordered_context(self):
        parser, surface = self.make_parser("joint"), self.surface()
        captured = []
        def capture_input(module, args):
            args[0].retain_grad()
            captured.append(args[0])
        handle = parser.mention_roles[0].register_forward_pre_hook(capture_input)
        try:
            output = parser(surface)
            loss = (F.cross_entropy(output["a"][:, 0], torch.tensor([0, 0])) +
                    F.cross_entropy(output["b"][:, 0], torch.tensor([4, 1])))
            loss.backward()
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        width = parser.encoder.width
        self.assertEqual(captured[0].shape[-1], 7 * width)
        gradient = captured[0].grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(float(gradient[..., :6 * width].abs().sum()), 0)
        self.assertGreater(float(gradient[..., 6 * width:].abs().sum()), 0)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in parser.encoder.blocks.parameters()), 0)
        self.assertGreater(float(parser.encoder.embedding.weight.grad.abs().sum()), 0)
        for invalid in ({"joint_roles": True},
                        {"joint_roles": True, "relative_positions": True},
                        {"joint_roles": True, "relative_positions": True, "constrained_roles": True,
                         "local_roles": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.model.LexicalParser(16, 1, self.vocabulary, **invalid)

    def test_evidence_pooling_shares_word_values_and_masks_padding_with_a_finite_cls_fallback(self):
        parser = self.make_parser("evidence")
        surface = self.surface(["Alice gives Bob 3 coins.", self.texts[0]])
        self.assertFalse(surface.sentence_mask[0, 1:].any().item())
        with torch.no_grad():
            for module in (parser.role_queries, parser.role_keys):
                module.weight.zero_()
                module.bias.zero_()
            parser.role_distance_bias.zero_()
            values = parser.role_values(parser.encoder.raw_embeddings(surface))
            mask = surface.token_mask.unsqueeze(-1)
            means = (values * mask).sum(2) / mask.sum(2)
            scores = parser.evidence_role_scores(surface, parser.encoder(surface))
            expected = means.unsqueeze(2).expand_as(scores)
            torch.testing.assert_close(scores, expected, atol=2e-6, rtol=2e-5)
            self.assertTrue(torch.isfinite(scores).all().item())
            for sentence in range(1, surface.ids.shape[1]):
                self.assertEqual(surface.token_mask[0, sentence].sum().item(), 1)
                torch.testing.assert_close(scores[0, sentence, 0], values[0, sentence, 0], atol=0, rtol=0)
            # No token at a masked position may contribute a lexical value.
            changed_ids = surface.ids.clone()
            changed_ids[~surface.token_mask] = self.vocabulary["gives"]
            changed = dataclasses.replace(surface, ids=changed_ids)
            altered = parser.evidence_role_scores(changed, parser.encoder(changed))
            torch.testing.assert_close(scores, altered, atol=2e-6, rtol=2e-5)
            output = parser(changed)
            for field in ("kind", "active", "a", "b", "features"):
                self.assertTrue(torch.isfinite(output[field]).all().item())

    def test_evidence_head_retains_signed_distance_and_cold_locality_information(self):
        texts = ["by the one who is Alice 3 coins are given to the one who is Bob.",
                 "to the one who is Alice 3 coins are given by the one who is Bob."]
        vocabulary = self.model.fit_lexicon(texts)
        surface = self.model.prepare_texts(texts, representation="lexical", vocabulary=vocabulary)
        torch.manual_seed(94401)
        parser = self.model.LexicalParser(16, 1, vocabulary, relative_positions=True,
                                        constrained_roles=True, evidence_roles=True).eval()
        with torch.no_grad():
            output = parser(surface)
            self.assertGreater(max(float((output[field][0, 0, :2] - output[field][1, 0, :2]).abs().max())
                                   for field in ("a", "b")), 1e-5)
        # A legal controlled setting can separately select left and right words;
        # this checks directional information, not learned semantic accuracy.
        parser, surface = self.make_parser("evidence"), self.surface(["Alice gives Bob 3 coins."])
        position = (surface.entity_ids[0, 0] == 1).nonzero()[0].item()
        with torch.no_grad():
            for module in (parser.role_queries, parser.role_keys):
                module.weight.zero_()
                module.bias.zero_()
            parser.role_values.weight.zero_()
            parser.role_values.weight[:, 0] = 1
            parser.role_values.bias.zero_()
            parser.role_distance_bias.fill_(-80)
            parser.role_distance_bias[0, 31] = parser.role_distance_bias[1, 33] = 80
            values = parser.role_values(parser.encoder.raw_embeddings(surface))
            scores = parser.evidence_role_scores(surface, parser.encoder(surface))
            expected = torch.stack((values[0, 0, position - 1, 0], values[0, 0, position + 1, 1]))
            self.assertGreater(float((expected[0] - expected[1]).abs()), 1e-5)
            torch.testing.assert_close(scores[0, 0, position], expected, atol=2e-6, rtol=2e-5)
            parser.role_distance_bias.copy_(parser.role_distance_bias.flip(0))
            reversed_scores = parser.evidence_role_scores(surface, parser.encoder(surface))
            torch.testing.assert_close(reversed_scores[0, 0, position], expected.flip(0), atol=2e-6, rtol=2e-5)

    def test_evidence_role_training_reaches_queries_keys_values_and_distance_bias(self):
        parser, surface = self.make_parser("evidence"), self.surface()
        output = parser(surface)
        loss = (F.cross_entropy(output["a"][:, 0], torch.tensor([0, 0])) +
                F.cross_entropy(output["b"][:, 0], torch.tensor([4, 1])))
        loss.backward()
        for module in (parser.role_queries, parser.role_keys, parser.role_values, parser.encoder.blocks):
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in module.parameters()))
            self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()), 0)
        self.assertTrue(torch.isfinite(parser.role_distance_bias.grad).all().item())
        self.assertGreater(float(parser.role_distance_bias.grad.abs().sum()), 0)
        for invalid in ({"evidence_roles": True},
                        {"evidence_roles": True, "relative_positions": True},
                        {"evidence_roles": True, "relative_positions": True, "constrained_roles": True,
                         "local_roles": True},
                        {"evidence_roles": True, "relative_positions": True, "constrained_roles": True,
                         "joint_roles": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.model.LexicalParser(16, 1, self.vocabulary, **invalid)

    def test_attachment_event_loss_trains_trigger_and_argument_links_without_auxiliary_labels(self):
        parser, surface = self.make_parser("attachment"), self.surface()
        output = parser(surface)
        self.assertNotIn("trigger_log_probs", output)
        self.assertNotIn("link_log_probs", output)
        loss = (F.cross_entropy(output["a"][:, 0], torch.tensor([0, 1])) +
                F.cross_entropy(output["b"][:, 0], torch.tensor([4, 0])))
        loss.backward()
        for name in ("trigger_head", "link_queries", "link_keys"):
            parameters = list(getattr(parser, name).parameters())
            with self.subTest(module=name):
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in parameters))
                self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in parameters), 0.)
        self.assertTrue(torch.isfinite(parser.link_distance_bias.grad).all().item())
        self.assertGreater(float(parser.link_distance_bias.grad.abs().sum()), 0.)
        for invalid in ({"attachment_roles": True},
                        {"attachment_roles": True, "relative_positions": True},
                        *({"attachment_roles": True, "relative_positions": True, "constrained_roles": True,
                           flag: True} for flag in ("local_roles", "joint_roles", "evidence_roles"))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.model.LexicalParser(16, 1, self.vocabulary, **invalid)
        with self.assertRaises(ValueError):
            self.make_parser("constrained")(surface, return_source_links=True)

    def test_attachment_predicted_source_mixture_drives_roles_and_masks_padding_without_gold_inputs(self):
        import semantic_tasks

        parser = self.make_parser("attachment")
        surface = self.surface(["3 coins. what is Alice's balance now?", *self.texts])
        with patch.object(semantic_tasks, "transfer_source_frame", side_effect=AssertionError("gold frame accessed")), \
                patch.object(semantic_tasks, "transfer_provenance", side_effect=AssertionError("gold provenance accessed")), \
                torch.no_grad():
            default = parser(surface)
            output = parser(surface, return_source_links=True)
        self.assertEqual(set(output), set(default) | {"trigger_log_probs", "link_log_probs"})
        for field in default:
            torch.testing.assert_close(output[field], default[field], atol=0, rtol=0)
        triggers, links = output["trigger_log_probs"].exp(), output["link_log_probs"].exp()
        self.assertEqual(triggers.shape, surface.ids.shape)
        self.assertEqual(links.shape, (*surface.ids.shape[:2], 2, surface.ids.shape[-1], surface.ids.shape[-1]))
        torch.testing.assert_close(triggers.sum(-1), torch.ones_like(triggers[..., 0]))
        torch.testing.assert_close(links.sum(-1), torch.ones_like(links[..., 0]))
        self.assertTrue((triggers[~surface.token_mask] == 0).all().item())
        valid_keys = (surface.entity_ids >= 0) & surface.token_mask
        valid_keys[..., 0] |= ~valid_keys.any(-1)
        self.assertTrue((links.masked_select(~valid_keys[:, :, None, None, :]) == 0).all().item())
        mixture = (links * triggers[:, :, None, :, None]).sum(-2)
        for role, name in enumerate(("a", "b")):
            for entity in range(4):
                selected = (surface.entity_ids == entity) & surface.token_mask
                present = selected.any(-1)
                probability = (mixture[:, :, role] * selected).sum(-1)
                torch.testing.assert_close(output[name][..., entity][present], probability[present].log(),
                                           atol=2e-6, rtol=2e-5)
            self.assertTrue(torch.isfinite(output[name]).all().item())
        self.assertEqual(links[0, 0, 0, 0, 0].item(), 1.)
        permutation = torch.tensor([3, 0, 2, 1])
        entities, ids = surface.entity_ids.clone(), surface.ids.clone()
        visible = entities >= 0
        entities[visible] = permutation[entities[visible]]
        ids[visible] = entities[visible] + 4
        with torch.no_grad():
            changed = parser(dataclasses.replace(surface, ids=ids, entity_ids=entities), return_source_links=True)
        for field in ("trigger_log_probs", "link_log_probs"):
            torch.testing.assert_close(output[field], changed[field], atol=0, rtol=0)

    def test_attachment_link_scores_retain_learnable_signed_predicate_to_argument_distance(self):
        parser, surface = self.make_parser("attachment"), self.surface(["Alice gives Bob 3 coins."])
        with torch.no_grad():
            for module in (parser.link_queries, parser.link_keys):
                module.weight.zero_()
                module.bias.zero_()
            distances = torch.arange(-32, 33, dtype=torch.float32)
            parser.link_distance_bias.copy_(torch.stack((distances / 4, -distances / 4)))
            links = parser(surface, return_source_links=True)["link_log_probs"]
            self.assertGreater(float(links[0, 0, 0, 2, 3]), float(links[0, 0, 0, 2, 1]))
            self.assertGreater(float(links[0, 0, 1, 2, 1]), float(links[0, 0, 1, 2, 3]))
            parser.link_distance_bias.neg_()
            reversed_links = parser(surface, return_source_links=True)["link_log_probs"]
            torch.testing.assert_close(links[:, :, 0], reversed_links[:, :, 1], atol=0, rtol=0)

    def test_source_frame_loss_handles_exact_links_multiple_predicates_and_empty_annotations(self):
        import math

        triggers = torch.full((2, 1, 5), -1e4)
        triggers[0, 0, 1] = triggers[1, 0, 0] = 0.
        links = torch.full((2, 1, 2, 5, 5), float("-inf"))
        links[:, :, 0, :, 3] = links[:, :, 1, :, 4] = 0.
        mask = torch.zeros_like(triggers, dtype=torch.bool)
        mask[0, 0, 1] = True
        targets = torch.tensor([[[3, 4]], [[-1, -1]]])
        output = {"trigger_log_probs": triggers.clone().requires_grad_(),
                  "link_log_probs": links.clone().requires_grad_()}
        loss = self.model.source_frame_loss(output, mask, targets)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(all(torch.isfinite(value.grad).all().item() for value in output.values()))
        mask[0, 0, 2] = True
        triggers[0, 0, 1:3] = -math.log(2)
        links[0, 0, 1, 1:3, 3], links[0, 0, 1, 1:3, 4] = math.log(.75), math.log(.25)
        output = {"trigger_log_probs": triggers.clone().requires_grad_(),
                  "link_log_probs": links.clone().requires_grad_()}
        loss = self.model.source_frame_loss(output, mask, targets)
        self.assertAlmostEqual(loss.item(), 2 * math.log(2), places=6)
        loss.backward()
        self.assertTrue((output["trigger_log_probs"].grad[~mask] == 0).all().item())
        allowed = torch.zeros_like(links, dtype=torch.bool)
        allowed[0, 0, 0, 1:3, 3] = allowed[0, 0, 1, 1:3, 4] = True
        self.assertTrue((output["link_log_probs"].grad[~allowed] == 0).all().item())
        empty = {"trigger_log_probs": triggers.clone().requires_grad_(), "link_log_probs": links}
        loss = self.model.source_frame_loss(empty, torch.zeros_like(mask), torch.full_like(targets, -1))
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue((empty["trigger_log_probs"].grad == 0).all().item())

    def test_source_frame_loss_rejects_misaligned_partial_cls_and_masked_labels(self):
        parser, surface = self.make_parser("attachment"), self.surface(["Alice gives Bob 3 coins."])
        output = parser(surface, return_source_links=True)
        mask = torch.zeros_like(surface.ids, dtype=torch.bool)
        mask[0, 0, 2] = True
        targets = torch.tensor([[[1, 3]]])
        self.assertTrue(torch.isfinite(self.model.source_frame_loss(output, mask, targets)).item())
        for invalid_mask, invalid_targets in ((mask.float(), targets), (mask[:, :, :-1], targets),
                                              (mask, targets.float()), (mask, targets[:, :, :1]),
                                              (mask, torch.tensor([[[0, 3]]])),
                                              (mask, torch.tensor([[[-1, 3]]])),
                                              (mask, torch.tensor([[[1, 1]]])),
                                              (mask, torch.tensor([[[1, 999]]])),
                                              (mask, torch.tensor([[[1, 2]]])),
                                              (torch.zeros_like(mask), targets)):
            with self.subTest(mask_shape=invalid_mask.shape, targets=invalid_targets), self.assertRaises(ValueError):
                self.model.source_frame_loss(output, invalid_mask, invalid_targets)
        cls_mask = mask.clone()
        cls_mask[0, 0, 0] = True
        with self.assertRaises(ValueError):
            self.model.source_frame_loss(output, cls_mask, targets)
        masked = {**output, "trigger_log_probs": output["trigger_log_probs"].clone()}
        masked["trigger_log_probs"][0, 0, 2] = float("-inf")
        with self.assertRaises(ValueError):
            self.model.source_frame_loss(masked, mask, targets)
        with self.assertRaises(ValueError):
            self.model.source_frame_loss({}, mask, targets)

    def test_optional_token_features_preserve_defaults_and_reuse_the_live_encoder_output(self):
        surface = self.surface(["Alice gives Bob 3 coins.", self.texts[0]])
        for pooling in ("clause", "pointer", "relative", "constrained", "local", "joint", "evidence", "attachment"):
            with self.subTest(pooling=pooling):
                parser = self.make_parser(pooling)
                default = parser(surface)
                self.assertNotIn("token_features", default)
                captured = []
                handle = parser.encoder.register_forward_hook(lambda module, args, output: captured.append(output))
                try:
                    explicit = parser(surface, return_token_features=True)
                finally:
                    handle.remove()
                self.assertEqual(len(captured), 1)
                self.assertIs(explicit["token_features"], captured[0])
                self.assertTrue(explicit["token_features"].requires_grad)
                self.assertEqual(set(explicit), set(default) | {"token_features"})
                for field in default:
                    torch.testing.assert_close(default[field], explicit[field], atol=0, rtol=0)
                tokens = explicit["token_features"]
                self.assertEqual(tokens.shape, (*surface.ids.shape, parser.encoder.width))
                self.assertTrue(torch.isfinite(tokens).all().item())
                self.assertTrue((tokens[~surface.sentence_mask] == 0).all().item())
                tokens[0, 0, :, 0].sum().backward()
                self.assertGreater(float(parser.encoder.embedding.weight.grad.abs().sum()), 0)
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in parser.encoder.blocks.parameters()))
                self.assertIsNone(parser.heads["kind"].weight.grad)

    def test_optional_token_features_preserve_numeric_and_identity_invariance(self):
        texts = [self.texts[0], self.texts[0].replace("9", "-16777216").replace("3", "16777216")]
        surface = self.surface(texts)
        parser = self.make_parser("constrained")
        permutation = torch.tensor([3, 0, 2, 1])
        ids, entities = surface.ids.clone(), surface.entity_ids.clone()
        valid = entities >= 0
        entities[valid] = permutation[entities[valid]]
        ids[valid] = entities[valid] + 4
        with torch.no_grad():
            original = parser(surface, return_token_features=True)["token_features"]
            changed = parser(dataclasses.replace(surface, ids=ids, entity_ids=entities),
                             return_token_features=True)["token_features"]
        torch.testing.assert_close(original[0], original[1], atol=0, rtol=0)
        torch.testing.assert_close(original, changed, atol=0, rtol=0)

    def test_joint_context_retains_the_distant_role_distinction_lost_by_local_features(self):
        texts = ["by the one who is Alice 3 coins are given to the one who is Bob.",
                 "to the one who is Alice 3 coins are given by the one who is Bob."]
        vocabulary = self.model.fit_lexicon(texts)
        surface = self.model.prepare_texts(texts, representation="lexical", vocabulary=vocabulary)
        torch.manual_seed(94201)
        parser = self.model.LexicalParser(48, 2, vocabulary, relative_positions=True,
                                        constrained_roles=True, joint_roles=True).eval()
        with torch.no_grad():
            # Biases initialize at zero, so a cold encoder has not learned any
            # positional preference. A fixed legal nonzero setting witnesses
            # representational capacity without training on these sentences.
            generator = torch.Generator().manual_seed(94202)
            for block in parser.encoder.blocks:
                block.relative_bias.copy_(torch.randn(block.relative_bias.shape, generator=generator) * .1)
            local, _ = parser.local_role_features(surface)
            output = parser(surface)
        for entity in (0, 1):
            mentions = [(surface.entity_ids[index, 0] == entity).nonzero()[0].item() for index in (0, 1)]
            torch.testing.assert_close(local[0, 0, mentions[0]], local[1, 0, mentions[1]], atol=2e-6, rtol=2e-5)
        # Distinguishability is necessary capacity, not a claim these untrained
        # random weights have learned the two correct transfer directions.
        self.assertGreater(max(float((output[field][0, 0, :2] - output[field][1, 0, :2]).abs().max())
                               for field in ("a", "b")), 1e-5)

    def test_joint_text_features_never_consume_oracle_metadata(self):
        import semantic_tasks

        parser, vocabulary = self.make_parser("joint"), dict(self.vocabulary)
        with patch.object(self.model, "oracle_tensor", side_effect=AssertionError("oracle rows read")), \
                patch.object(self.model, "answer_targets", side_effect=AssertionError("answer labels read")), \
                patch.object(self.model, "execute", side_effect=AssertionError("executor used in parser")):
            output = parser(self.surface())
            self.assertEqual(output["a"].shape, (2, 4, 5))
            example = semantic_tasks.Example(self.texts[0], (), (), "999999", "secret task label")
            with self.assertRaises(ValueError):
                self.model.prepare_texts([example], representation="lexical", vocabulary=vocabulary)
            with self.assertRaises(ValueError):
                self.model.prepare_texts([dataclasses.asdict(example)], representation="lexical", vocabulary=vocabulary)
        self.assertEqual(parser.vocabulary, vocabulary)

    def test_existing_lexical_initializations_are_unchanged_by_new_role_heads(self):
        import semantic_experiment

        expected = {
            "clause": "e97261363a51bfd798609d7c97855b427efa8bdc43bbeccf6299c68a170f66e9",
            "pointer": "0c0be7e082bb6e634c4cf94a544d2670c9f74e59a96ed7966cd72a3c8db96103",
            "relative": "778abbc0fcc19ceb602203db063382e3ee3eb33c9fccb85793f3be73b0b4d8c9",
            "constrained": "778abbc0fcc19ceb602203db063382e3ee3eb33c9fccb85793f3be73b0b4d8c9",
            "local": "36f4a7de4a54d1146ac4f39edde16a5599df3949d77e2966d6986c7348b855e2",
            "joint": "b408bd5d760f0ce5990e588399605bc56f2ab852c640d58b64a89047af06a26b",
        }
        for pooling, state_hash in expected.items():
            with self.subTest(pooling=pooling):
                self.assertEqual(semantic_experiment.tensor_state_hash(self.make_parser(pooling)), state_hash)

    def test_scaled_parsers_restore_against_the_same_fixed_reasoning_core(self):
        import semantic_experiment

        core = self.model.StableReasoner(16, 1).eval().requires_grad_(False)
        core_hash = semantic_experiment.tensor_state_hash(core)
        self.assertEqual(sum(parameter.numel() for parameter in core.parameters()), 346)
        counts = []
        surface = self.surface()
        for width, layers in ((48, 2), (96, 2), (192, 2)):
            with self.subTest(width=width, layers=layers):
                parser = self.model.LexicalParser(
                    width, layers, self.vocabulary, relative_positions=True,
                    constrained_roles=True, local_roles=True,
                ).eval()
                counts.append(sum(parameter.numel() for parameter in parser.parameters()))
                metadata = {"config": {"width": width, "local_layers": layers,
                                       "layers": 1, "architecture": "stable"}}
                with tempfile.TemporaryDirectory(prefix="text-vector-scale-test-") as directory:
                    path = Path(directory) / "model.pt"
                    self.model.save_checkpoint(path, parser, None, core, None, metadata)
                    restored, restored_metadata = self.model.load_checkpoint(path)
                self.assertEqual(restored_metadata["config"]["width"], width)
                self.assertEqual(len(restored["parser"].encoder.blocks), layers)
                self.assertEqual(semantic_experiment.tensor_state_hash(restored["oracle"]), core_hash)
                self.assertEqual(sum(parameter.numel() for parameter in restored["oracle"].parameters()), 346)
                with torch.no_grad():
                    expected, actual = parser(surface), restored["parser"](surface)
                for field in ("kind", "active", "a", "b", "features"):
                    self.assertTrue(torch.isfinite(actual[field]).all().item())
                    torch.testing.assert_close(expected[field], actual[field], atol=0, rtol=0)
                torch.testing.assert_close(self.model.parsed_rows(expected, surface),
                                           self.model.parsed_rows(actual, surface), atol=0, rtol=0)
        self.assertTrue(all(left < right for left, right in zip(counts, counts[1:])))
        self.assertEqual(semantic_experiment.tensor_state_hash(core), core_hash)

    def test_local_baseline_loses_distant_role_order_even_when_scaled(self):
        texts = ["by the one who is Alice 3 coins are given to the one who is Bob.",
                 "to the one who is Alice 3 coins are given by the one who is Bob."]
        vocabulary = self.model.fit_lexicon(texts)
        surface = self.model.prepare_texts(texts, representation="lexical", vocabulary=vocabulary)
        self.assertNotIn(2, surface.ids[surface.token_mask].tolist())
        # This is an input-information collision, not evidence from a trained
        # candidate: each entity has the same five-token window and clause bag,
        # but the distant by/to order reverses the correct debit/credit roles.
        self.assertEqual(sorted(surface.ids[0, 0].tolist()), sorted(surface.ids[1, 0].tolist()))
        for entity in (0, 1):
            windows = []
            for index in (0, 1):
                position = int((surface.entity_ids[index, 0] == entity).nonzero()[0])
                windows.append(surface.ids[index, 0, position - 2:position + 3].tolist())
            self.assertEqual(*windows)
        gold_roles = torch.tensor([[0., 1.], [1., 0.]])
        for width in (48, 96, 192):
            with self.subTest(width=width):
                torch.manual_seed(94051)
                parser = self.model.LexicalParser(
                    width, 2, vocabulary, relative_positions=True,
                    constrained_roles=True, local_roles=True,
                ).eval()
                with torch.no_grad():
                    output = parser(surface)
                for field in ("a", "b"):
                    torch.testing.assert_close(output[field][0], output[field][1], atol=2e-6, rtol=2e-5)
                # Isolate role binding from the contextual event classifier.
                output["kind"] = F.one_hot(torch.full((2, 1), 2), 6).float()
                decoded = self.model.parsed_rows(output, surface)[:, 0, 1:3]
                torch.testing.assert_close(decoded[0], decoded[1], atol=0, rtol=0)
                self.assertFalse(torch.equal(decoded, gold_roles))

    def test_joint_role_decoder_uses_learned_direction_and_predicted_arity(self):
        surface = self.surface(["Alice gives Bob 3 coins."])
        output = {"kind": torch.zeros((1, 1, 6)), "active": torch.tensor([[[0., 1.]]]),
                  "a": torch.tensor([[[7., 6., -100., -100., 10.]]]),
                  "b": torch.tensor([[[6., 4., -100., -100., 10.]]])}
        for kind in (2, 3, 5):
            output["kind"].zero_()
            output["kind"][0, 0, kind] = 1
            legacy = self.model.parsed_rows(output, surface)
            self.assertEqual(legacy[0, 0, 1:3].tolist(), [-1., -1.])
            constrained = {**output, "constrained_roles": True}
            result = self.model.parsed_rows(constrained, surface)
            # (Alice,Bob)=7+4; (Bob,Alice)=6+6. Direction is the learned joint optimum.
            self.assertEqual(result[0, 0, 1:3].tolist(), [1., 0.])
            collision = {**output, "a": output["a"].clone(), "b": output["b"].clone()}
            collision["a"][..., 4] = collision["b"][..., 4] = -100
            self.assertEqual(self.model.parsed_rows(collision, surface)[0, 0, 1:3].tolist(), [0., 0.])
            self.assertEqual(self.model.parsed_rows({**collision, "constrained_roles": True}, surface)[0, 0, 1:3].tolist(), [1., 0.])
            reversed_scores = {**constrained, "b": output["b"].clone()}
            reversed_scores["b"][..., 1] = 9
            self.assertEqual(self.model.parsed_rows(reversed_scores, surface)[0, 0, 1:3].tolist(), [0., 1.])
        for kind in (1, 4):
            output["kind"].zero_()
            output["kind"][0, 0, kind] = 1
            result = self.model.parsed_rows({**output, "constrained_roles": True}, surface)
            self.assertEqual(result[0, 0, 1:3].tolist(), [0., -1.])
            self.assertEqual(result[0, 0, 3:].tolist(), [3., 1.])
        default, constrained = self.make_parser("relative"), self.make_parser("constrained")
        self.assertNotIn("constrained_roles", default(surface))
        self.assertTrue(constrained(surface)["constrained_roles"])
        self.assertEqual(sum(p.numel() for p in default.parameters()), sum(p.numel() for p in constrained.parameters()))
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, constrained.state_dict()[name], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.model.LexicalParser(16, 1, self.vocabulary, constrained_roles=True)

    def test_joint_role_decoder_preserves_identity_permutation_and_rejects_missing_roles(self):
        surface = self.surface(["Alice is before Bob. is Alice before Bob?"])
        output = {"kind": F.one_hot(torch.tensor([[3, 5]]), 6).float(),
                  "active": torch.tensor([[[0., 1.], [0., 1.]]]),
                  "a": torch.tensor([[[7., 6., -100., -100., 10.]]]).repeat(1, 2, 1),
                  "b": torch.tensor([[[6., 4., -100., -100., 10.]]]).repeat(1, 2, 1),
                  "constrained_roles": True}
        original = self.model.parsed_rows(output, surface)
        permutation = torch.tensor([3, 0, 2, 1])
        ids, entities = surface.ids.clone(), surface.entity_ids.clone()
        valid = entities >= 0
        entities[valid] = permutation[entities[valid]]
        ids[valid] = entities[valid] + 4
        changed = {**output}
        for field in ("a", "b"):
            changed[field] = output[field].clone()
            changed[field][..., permutation] = output[field][..., :4]
        result = self.model.parsed_rows(changed, dataclasses.replace(surface, ids=ids, entity_ids=entities))
        expected = original.clone()
        expected[..., 1:3] = permutation[original[..., 1:3].long()].float()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        for query in ("is Alice before?", "is it before?"):
            text = "Alice is before Bob. " + query
            missing = self.surface([text])
            rows = self.model.parsed_rows(output, missing)
            self.assertEqual(rows[0, -1, 1:3].tolist(), [-1., -1.])
            self.assertFalse(self.model.StableReasoner.valid_queries(rows, missing.sentence_mask).item())
            models = {"parser": self.make_parser("constrained"), "oracle": self.model.StableReasoner(16, 1)}
            with patch.object(models["parser"], "forward", return_value=output):
                for mode in ("parsed", "executor"):
                    with self.assertRaises(ValueError):
                        self.model.predict(models, text, mode)

    def test_interpretation_gates_reject_hidden_program_world_and_seed_failures(self):
        import semantic_experiment

        shifts = ("iid", "names", "wording", "length", "numbers", "composition", "combined")
        tasks = ("accounting", "relations")
        good = {"parser": {"per_task": {task: {"program_accuracy": 1.0} for task in tasks}},
                **{mode: {"per_task": {task: {"exact_accuracy": 1.0} for task in tasks}}
                   for mode in ("parsed", "executor")},
                "all_queries": {"per_task": {task: {"all_correct": 1.0} for task in tasks}}}
        trials = [{"seed": seed, "test": {shift: copy.deepcopy(good) for shift in shifts}}
                  for seed in (84091, 84092, 84093)]
        self.assertTrue(semantic_experiment.interpretation_gates(trials)["passed"])
        with self.assertRaises(ValueError):
            semantic_experiment.interpretation_gates(trials, require_retention=True)
        with self.assertRaises(ValueError):
            semantic_experiment.interpretation_gates([])
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            candidate = copy.deepcopy(trials)
            candidate[0]["test"]["wording"]["parser"]["per_task"]["relations"]["program_accuracy"] = value
            with self.assertRaises(ValueError):
                semantic_experiment.interpretation_gates(candidate)
        for mode, field, metric in (("parser", "program_accuracy", "program"),
                                   ("all_queries", "all_correct", "all_queries_world"),
                                   ("parsed", "exact_accuracy", "parsed")):
            candidate = copy.deepcopy(trials)
            candidate[0]["test"]["wording"][mode]["per_task"]["relations"][field] = .94
            gate = semantic_experiment.interpretation_gates(candidate)
            self.assertFalse(gate["passed"])
            self.assertEqual([(failure["seed"], failure["task"], failure["metric"]) for failure in gate["failures"]],
                             [(84091, "relations", metric)])
        for missing in ("shift", "worlds"):
            candidate = copy.deepcopy(trials)
            if missing == "shift":
                del candidate[0]["test"]["combined"]
            else:
                del candidate[0]["test"]["combined"]["all_queries"]
            with self.assertRaises(ValueError):
                semantic_experiment.interpretation_gates(candidate)
        retained_trials = copy.deepcopy(trials)
        for trial in retained_trials:
            trial["test"]["retention"] = copy.deepcopy(good)
        self.assertEqual(semantic_experiment.interpretation_gates(retained_trials)["thresholds"]["retention"], .99)
        for key in ("test", "development"):
            candidate = copy.deepcopy(retained_trials)
            if key == "development":
                for trial in candidate:
                    trial[key] = trial.pop("test")
                    del trial[key]["names"]
            self.assertTrue(semantic_experiment.interpretation_gates(candidate, key)["passed"])
            candidate[0][key]["retention"]["parser"]["per_task"]["accounting"]["program_accuracy"] = .98
            gate = semantic_experiment.interpretation_gates(candidate, key)
            self.assertFalse(gate["passed"])
            self.assertEqual([(row["seed"], row["shift"], row["task"]) for row in gate["failures"]],
                             [(84091, "retention", "accounting")])
            candidate[0][key]["retention"]["parser"]["per_task"]["accounting"]["program_accuracy"] = float("nan")
            with self.assertRaises(ValueError):
                semantic_experiment.interpretation_gates(candidate, key)
            del candidate[0][key]["retention"]
            with self.assertRaises(ValueError):
                semantic_experiment.interpretation_gates(candidate, key)

    def test_grouped_metrics_distinguish_program_fidelity_answers_and_worlds(self):
        import semantic_experiment
        import semantic_tasks

        groups = semantic_tasks.all_query_examples(
            semantic_tasks.make_interpretation_split(4, 84101), seed=84102,
        )
        gold = [example.answer for group in groups for example in group]
        predictions = list(gold)
        first_query, cursor = {}, 0
        kinds = []
        for group in groups:
            kinds.append("polarity" if group[0].task not in first_query else "role_swap")
            first_query.setdefault(group[0].task, cursor)
            cursor += len(group)
        for index in first_query.values():
            predictions[index] = "<invalid>"
        def controlled_parse(parser, data):
            data.predicted = data.rows.clone()
            data.predicted[first_query["accounting"], 0, 3] += 1

        core = self.model.StableReasoner(16, 1).requires_grad_(False)
        before = semantic_experiment.tensor_state_hash(core)
        # Inject known prediction errors to test aggregation independently of
        # whether a randomly initialized parser/core solves these examples.
        with patch.object(semantic_experiment, "cache_parser", side_effect=controlled_parse), \
                patch.object(semantic_experiment, "decode_answers", side_effect=[predictions, gold]):
            result = semantic_experiment.evaluate_query_worlds(
                {"parser": self.make_parser(), "oracle": core}, groups, "cpu", group_kinds=kinds,
            )
        self.assertEqual(result["world_count"], 4)
        self.assertEqual(result["query_count"], 32)
        for task, queries in (("accounting", 8), ("relations", 24)):
            row = result["per_task"][task]
            self.assertEqual(row["all_correct"], .5)
            self.assertEqual(row["query_accuracy"], (queries - 1) / queries)
            self.assertEqual(row["all_programs_correct"], .5 if task == "accounting" else 1.)
            for kind in ("polarity", "role_swap"):
                separate = result["by_contrast"][kind][task]
                self.assertEqual(separate["worlds"], 1)
                self.assertEqual(separate["all_correct_count"], int(kind == "role_swap"))
                self.assertEqual(separate["all_correct"], float(kind == "role_swap"))
                self.assertEqual(separate["all_programs_correct"],
                                 float(task == "relations" or kind == "role_swap"))
                self.assertEqual(separate["all_programs_correct_count"],
                                 int(task == "relations" or kind == "role_swap"))
        self.assertEqual(result["oracle_input_core"]["macro_exact_accuracy"], 1.)
        self.assertEqual(semantic_experiment.tensor_state_hash(core), before)
        with self.assertRaisesRegex(ValueError, "Contrast kinds"):
            semantic_experiment.evaluate_query_worlds(
                {"parser": self.make_parser(), "oracle": core}, groups, "cpu", group_kinds=kinds[:-1])

    def test_grouped_metrics_support_accounting_only_polarity_probes(self):
        import semantic_experiment
        import semantic_tasks

        groups, _ = semantic_tasks.direct_polarity_groups([SemanticPolarityTaskTests.fixture()])
        examples = [group.examples for group in groups]
        gold = [example.answer for group in examples for example in group]
        parser = self.make_parser("constrained")
        core = self.model.StableReasoner(16, 1).requires_grad_(False)
        for reverse_roles in (False, True):
            def controlled_parse(parser, data):
                data.predicted = data.rows.clone()
                if reverse_roles:
                    data.predicted[:, 4, 1:3] = data.predicted[:, 4, 1:3].flip(-1)
            with patch.object(semantic_experiment, "cache_parser", side_effect=controlled_parse), \
                    patch.object(semantic_experiment, "decode_answers", side_effect=[gold, gold]):
                result = semantic_experiment.evaluate_query_worlds(
                    {"parser": parser, "oracle": core}, examples, "cpu", changed_clauses=[4])
            self.assertEqual(result["world_count"], 1)
            self.assertEqual(result["query_count"], 4)
            self.assertNotIn("by_contrast", result)
            self.assertEqual(set(result["per_task"]), {"accounting"})
            metrics = result["per_task"]["accounting"]
            self.assertEqual(metrics["all_correct"], 1.)
            self.assertEqual(metrics["all_programs_correct"], float(not reverse_roles))
            self.assertEqual(metrics["target_row_accuracy"], float(not reverse_roles))
            self.assertEqual(metrics["role_consistency_accuracy"], 1.)
            self.assertEqual(result["oracle_input_core"]["macro_exact_accuracy"], 1.)


if __name__ == "__main__":
    # Small regression fixtures are faster and less noisy with one CPU worker.
    torch.set_num_threads(1)
    unittest.main()
