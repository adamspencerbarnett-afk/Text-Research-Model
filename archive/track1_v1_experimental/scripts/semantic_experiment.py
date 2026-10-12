"""Matched structural experiment; screen on validation, confirm on fresh holdouts.

Only this guarded runner trains models or updates the existing compact results.
No online learning is implemented. Run commands sequentially: the summary file
is one ordinary JSON document, not a concurrent experiment database.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import time

import torch
import torch.nn.functional as F

from semantic_model import (
    ROOT,
    BoundReasoner,
    RawReasoner,
    Reasoner,
    SemanticParser,
    StableReasoner,
    SurfaceBatch,
    answer_loss,
    answer_targets,
    decode_answers,
    load_checkpoint,
    oracle_tensor,
    parsed_rows,
    parser_loss,
    predict,
    prepare_texts,
    save_checkpoint,
)
from semantic_tasks import (
    QUERY_BALANCE,
    QUERY_BEFORE,
    SHIFTS,
    STABILITY_SHIFTS,
    execute,
    make_split,
    make_stability_split,
    split_fingerprint,
)

SUMMARY = ROOT / "results/semantic_v1.json"
CHECKPOINT = ROOT / "models/semantic_model.pt"
CANDIDATE = ROOT / "models/semantic_candidate.pt"
LEARNED_MODES = ("raw", "oracle", "parsed", "hybrid", "raw_aux")
DEPLOYABLE_MODES = ("raw", "parsed", "hybrid", "raw_aux", "executor")


# Shared preparation helpers. Gold rows stay in this runner for supervision and
# diagnostics; ordinary inference in semantic_model.py receives visible text.
def sync(device):
    """Finish queued CUDA work before recording a timing boundary."""

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def state_copy(model):
    """Clone a model state to CPU for deterministic checkpoint selection."""

    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parameter_count(model):
    """Return the number of learned scalar parameters in a component."""

    return sum(parameter.numel() for parameter in model.parameters())


def scaling_indices(count, batch_size, generator, amount_generator):
    """Keep old/new exposure fixed when the broad corpus grows from 6k to 18k."""
    if count not in (9000, 21000):
        raise ValueError("Scaling sampling requires 3000 retained plus 6000 or 18000 new examples")
    base = torch.randint(9000, (batch_size,), generator=generator)
    blocks = (count - 3000) // 6000
    offset = torch.randint(blocks, (batch_size,), generator=amount_generator) * 6000
    return torch.where(base < 3000, base, base + offset)


@dataclass
class Prepared:
    """One split with aligned examples, surface tensors, rows, and answers."""

    examples: list
    surface: SurfaceBatch
    rows: torch.Tensor
    targets: torch.Tensor
    predicted: torch.Tensor | None = None
    features: torch.Tensor | None = None


def prepare(examples, device, representation="bytes", vocabulary=None):
    """Encode a split once so every compared path sees the same tensors."""

    options = {"vocabulary": vocabulary} if vocabulary is not None else {}
    return Prepared(examples, prepare_texts([example.text for example in examples], device, representation, **options),
                    oracle_tensor(examples, device), answer_targets(examples, device))


@dataclass
class SourceLabels:
    """Training-only predicate and semantic-argument locations."""

    triggers: torch.Tensor
    mentions: torch.Tensor
    report: dict


@dataclass
class PairedTraining:
    """Aligned positive/negative views and optional source supervision."""

    data: Prepared
    alignment: torch.Tensor
    report: dict
    source: SourceLabels | None = None


def prepare_source_labels(examples, surface, stage="train"):
    """Labels for training or explicit oracle diagnostics, never model inputs."""
    from semantic_tasks import TRANSFER, transfer_source_frame
    if surface.representation != "lexical" or surface.entity_ids is None or len(examples) != len(surface.ids):
        raise ValueError("Source labels require matching lexical examples and surfaces")
    started = time.perf_counter()
    triggers = torch.zeros(surface.ids.shape, dtype=torch.bool)
    mentions = torch.full((*surface.ids.shape[:2], 2), -1, dtype=torch.long)
    entities, masks = surface.entity_ids.cpu(), surface.token_mask.cpu()
    sentence_mask = surface.sentence_mask.cpu()
    report = {"stage": stage, "worlds": len(examples), "transfer_clauses": 0,
              "predicates": 0, "semantic_argument_links": 0, "dual_predicate_clauses": 0,
              "annotation": "Every co-referring predicate links to both semantic event arguments; not grammatical dependency heads"}
    for world, example in enumerate(examples):
        if len(example.sentences) != int(sentence_mask[world].sum()):
            raise ValueError("Source labels and surface clause counts differ")
        for clause, row in enumerate(example.rows):
            if row[0] != TRANSFER:
                continue
            frame = transfer_source_frame(example, clause, stage)
            positions = [frame.sender_position + 1, frame.recipient_position + 1]
            if (not bool(masks[world, clause, positions].all()) or
                    entities[world, clause, positions].tolist() != list(row[1:3])):
                raise ValueError("Source argument positions do not match visible event identities")
            predicates = [position + 1 for position in frame.predicate_positions]
            if not bool(masks[world, clause, predicates].all()):
                raise ValueError("Source predicate lies outside visible text")
            triggers[world, clause, predicates] = True
            mentions[world, clause] = torch.tensor(positions)
            report["transfer_clauses"] += 1
            report["predicates"] += len(predicates)
            report["semantic_argument_links"] += 2 * len(predicates)
            report["dual_predicate_clauses"] += len(predicates) == 2
    report.update(preparation_seconds=time.perf_counter() - started,
                  examples_sha256=split_fingerprint(examples),
                  label_tensor_sha256=hashlib.sha256(triggers.numpy().tobytes() + mentions.numpy().tobytes()).hexdigest(),
                  label_tensor_bytes=triggers.numel() * triggers.element_size() + mentions.numel() * mentions.element_size(),
                  rendered_bytes=sum(len(example.text.encode("utf-8")) for example in examples))
    return SourceLabels(triggers.to(surface.ids.device), mentions.to(surface.ids.device), report)


def polarity_alignment(surface, changed_clauses):
    """Align actual mentions by visible identity and occurrence, not position."""
    clauses = torch.as_tensor(changed_clauses, dtype=torch.long, device="cpu")
    if (surface.entity_ids is None or clauses.ndim != 1 or
            surface.ids.shape[0] != 2 * len(clauses)):
        raise ValueError("Polarity alignment requires interleaved lexical pairs and one clause per pair")
    entities, masks = surface.entity_ids.cpu(), surface.token_mask.cpu()
    sentence_mask = surface.sentence_mask.cpu()
    links = []
    for pair, clause in enumerate(clauses.tolist()):
        current = []
        if clause < -1 or clause >= surface.ids.shape[1]:
            raise ValueError("Invalid changed clause")
        if clause >= 0:
            if not bool(sentence_mask[2 * pair:2 * pair + 2, clause].all()):
                raise ValueError("Changed clause must be present in both views")
            for entity in range(4):
                positions = [torch.where((entities[2 * pair + side, clause] == entity) &
                                         masks[2 * pair + side, clause])[0].tolist()
                             for side in (0, 1)]
                if len(positions[0]) != len(positions[1]):
                    raise ValueError("Polarity views changed visible mention multiplicity")
                current.extend((clause, left, right) for left, right in zip(*positions))
            if not current:
                raise ValueError("Changed transfer has no aligned entity mentions")
        links.append(current)
    result = torch.full((len(clauses), max(1, max(map(len, links), default=0)), 3), -1,
                        dtype=torch.long, device=surface.ids.device)
    for pair, current in enumerate(links):
        if current:
            result[pair, :len(current)] = torch.tensor(current, device=result.device)
    return result


def polarity_feature_loss(output, alignment):
    """A training-only representation constraint; activity keeps ordinary CE."""
    features = output.get("token_features")
    if (features is None or features.ndim != 4 or alignment.ndim != 3 or
            alignment.shape[-1] != 3 or alignment.dtype != torch.long or
            features.shape[0] != 2 * alignment.shape[0]):
        raise ValueError("Feature consistency requires encoder features and paired mention links")
    if bool(((alignment < -1).any(-1) | ((alignment == -1).any(-1) &
                                      ~(alignment == -1).all(-1))).any()):
        raise ValueError("Invalid padded mention alignment")
    valid = alignment[:, :, 0] >= 0
    if not bool(valid.any()):
        return features.sum() * 0
    pair = torch.arange(alignment.shape[0], device=features.device)[:, None].expand(valid.shape)[valid]
    clause, left, right = alignment[valid].unbind(-1)
    if bool(((clause >= features.shape[1]) | (left >= features.shape[2]) |
             (right >= features.shape[2])).any()):
        raise ValueError("Mention alignment exceeds captured features")
    return (1 - F.cosine_similarity(features[2 * pair, clause, left],
                                    features[2 * pair + 1, clause, right], dim=-1)).mean()


# Evaluation helpers keep program fidelity separate from final-answer accuracy.
def available_modes(models):
    """List learned inference paths actually present in a checkpoint."""

    return tuple(mode for mode in LEARNED_MODES
                 if ("oracle" if mode == "parsed" else mode) in models)


def metric_answers(examples, predictions):
    """Report exact answers overall and separately for each task family."""

    per_task = {}
    for task in ("accounting", "relations"):
        indices = [i for i, example in enumerate(examples) if example.task == task]
        correct = sum(predictions[i] == examples[i].answer for i in indices)
        per_task[task] = {"correct": correct, "count": len(indices),
                          "exact_accuracy": correct / len(indices) if indices else None}
    accuracies = [entry["exact_accuracy"] for entry in per_task.values() if entry["count"]]
    return {"macro_exact_accuracy": statistics.mean(accuracies),
            "exact_accuracy": sum(p == e.answer for p, e in zip(predictions, examples)) / len(examples),
            "per_task": per_task}


def metric_parser(data):
    """Score complete programs, individual events, queries, and row fields."""

    mask = data.surface.sentence_mask
    equal = data.predicted == data.rows
    events = equal.all(-1)
    programs = (events | ~mask).all(-1)
    query_mask = mask & ((data.rows[:, :, 0] == QUERY_BALANCE) |
                         (data.rows[:, :, 0] == QUERY_BEFORE))
    return {"event_accuracy": float(events[mask].float().mean()),
            "program_accuracy": int(programs.sum()) / len(data.examples),
            "per_task": {task: {"count": len(indices),
                                  "program_accuracy": int(programs[indices].sum()) / len(indices)}
                         for task in ("accounting", "relations")
                         if (indices := [i for i, example in enumerate(data.examples) if example.task == task])},
            "query_accuracy": float(events[query_mask].float().mean()),
            "field_accuracy": {field: float(equal[:, :, i][mask].float().mean())
                               for i, field in enumerate(("kind", "a", "b", "value", "active"))},
            "event_count": int(mask.sum()), "program_count": len(data.examples)}


@torch.no_grad()
def cache_parser(parser, data, batch_size=128):
    """Cache predicted rows/features once for fair downstream comparisons."""

    parser.eval()
    count, sentences = data.surface.sentence_mask.shape
    data.predicted = torch.zeros_like(data.rows)
    data.features = torch.zeros((count, sentences, parser.encoder.width), device=data.rows.device)
    loss_sum = 0.0
    for start in range(0, count, batch_size):
        indices = torch.arange(start, min(start + batch_size, count), device=data.rows.device)
        surface = data.surface.take(indices)
        width = surface.sentence_mask.shape[1]
        output = parser(surface)
        data.predicted[indices, :width] = parsed_rows(output, surface)
        data.features[indices, :width] = output["features"]
        loss_sum += float(parser_loss(output, data.rows[indices, :width], surface.sentence_mask)) * len(indices)
    result = metric_parser(data)
    result["loss"] = loss_sum / count
    return result


def forward_core(model, mode, data, indices):
    """Route one batch through the raw, oracle, parsed, or hybrid path."""

    surface = data.surface.take(indices)
    width = surface.sentence_mask.shape[1]
    if mode in ("raw", "raw_aux"):
        return model(surface)
    rows = data.rows if mode == "oracle" else data.predicted
    return model(rows[indices, :width], surface.sentence_mask,
                 data.features[indices, :width] if mode == "hybrid" else None)


@torch.no_grad()
def evaluate_core(model, mode, data, batch_size=128):
    """Evaluate a learned answer path without updating its parameters."""

    model.eval()
    predictions, loss_sum = [], 0.0
    for start in range(0, len(data.examples), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data.examples)), device=data.rows.device)
        output = forward_core(model, mode, data, indices)
        valid_queries = None
        if isinstance(model, StableReasoner):
            rows = data.rows if mode == "oracle" else data.predicted
            surface = data.surface.take(indices)
            valid_queries = model.valid_queries(rows[indices, :surface.sentence_mask.shape[1]],
                                                 surface.sentence_mask)
        predictions.extend(decode_answers(output, valid_queries))
        loss_sum += float(answer_loss(output, data.targets[indices])) * len(indices)
    result = metric_answers(data.examples, predictions)
    result["loss"] = loss_sum / len(data.examples)
    if mode == "parsed":
        correct_parse = (((data.predicted == data.rows).all(-1) |
                          ~data.surface.sentence_mask).all(-1)).cpu().tolist()
        for correct, name in ((True, "given_correct_parse"), (False, "given_incorrect_parse")):
            indices = [i for i, flag in enumerate(correct_parse) if flag == correct]
            hits = sum(predictions[i] == data.examples[i].answer for i in indices)
            result[name] = {"count": len(indices), "correct": hits,
                            "exact_accuracy": hits / len(indices) if indices else None}
    return result


def evaluate_executor(data, query_validator=None):
    """Execute predicted programs exactly while counting invalid programs."""

    rows = data.predicted.detach().cpu().tolist()
    valid_queries = (query_validator(data.predicted, data.surface.sentence_mask).cpu().tolist()
                     if query_validator is not None else [True] * len(rows))
    predictions, invalid = [], 0
    for example, predicted, valid_query in zip(data.examples, rows, valid_queries):
        if execute(example.rows) != example.answer:
            raise AssertionError("Oracle executor disagrees with independently generated answer")
        try:
            if not valid_query:
                raise ValueError("Missing or invalid final query")
            predictions.append(execute(tuple(tuple(int(value) for value in row)
                                             for row in predicted[:len(example.sentences)])))
        except (ValueError, IndexError, KeyError):
            predictions.append("<invalid>")
            invalid += 1
    return {**metric_answers(data.examples, predictions), "invalid_programs": invalid,
            "oracle_executor_exact_accuracy": 1.0}


# Training, preservation, and development-only checkpoint selection.
ROLE_PARAMETER_PREFIXES = ("trigger_head.", "link_queries.", "link_keys.", "none_roles.")


def parameter_subset_hash(model, names):
    """Hash a declared parameter subset to prove frozen weights stayed fixed."""

    parameters = dict(model.named_parameters())
    if not names or set(names) - parameters.keys():
        raise ValueError("Parameter hashes require an existing nonempty subset")
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(parameters[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_parser_for_roles(model):
    """Keep the optimizer, but stop encoder and event-type/activity updates."""
    if not getattr(model, "attachment_roles", False):
        raise ValueError("Role-only preservation requires the attachment parser")
    parameters = dict(model.named_parameters())
    role_names = [name for name in parameters if name == "link_distance_bias" or
                  name.startswith(ROLE_PARAMETER_PREFIXES)]
    frozen_names = [name for name in parameters if name not in role_names]
    if not role_names or any(not name.startswith(("encoder.", "heads.kind.", "heads.active."))
                             for name in frozen_names):
        raise ValueError("Unexpected parameters in the preservation architecture")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in role_names)
        if name in frozen_names:
            parameter.grad = None  # AdamW must also skip decay and stale moments.
    return {"trainable_names": sorted(role_names), "frozen_names": sorted(frozen_names),
            "trainable_parameters": sum(parameters[name].numel() for name in role_names),
            "frozen_parameters": sum(parameters[name].numel() for name in frozen_names),
            "boundary_frozen_sha256": parameter_subset_hash(model, frozen_names)}


def preservation_selection_key(scores):
    """Prefer checkpoints meeting fixed primary program gates before ranking."""
    programs, eligible = [], True
    for name, entry in scores.items():
        required = .99 if name in ("iid", "names", "retention") else .95
        for task in entry["per_task"].values():
            value = task["program_accuracy"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Checkpoint selection requires finite program accuracies")
            programs.append(value)
            eligible &= value >= required
    if not programs:
        raise ValueError("Checkpoint selection requires primary scores")
    return (eligible, min(programs), statistics.mean(programs),
            -statistics.mean(entry["loss"] for entry in scores.values()))


def train_component(model, mode, train, valid, args, seed, selection_sets=None,
                    paired=None, consistency_weight=0.0, source_weight=0.0,
                    freeze_after=0, capture_step=0):
    """Train one component under the shared sampling and selection protocol.

    Optional paired views, source supervision, and role-only freezing are
    explicit experiment arms. Checkpoints are selected from validation data.
    """

    is_parser = mode == "parser"
    steps = args.parser_steps if is_parser else args.steps
    device = train.rows.device
    if paired is not None and (not is_parser or args.batch_size % 2 or
                               len(paired.data.examples) != 2 * len(train.examples)):
        raise ValueError("Paired training needs a parser, an even batch and two views per parent")
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("Consistency weight must be finite and nonnegative")
    if consistency_weight and paired is None:
        raise ValueError("Consistency requires paired training")
    if (not math.isfinite(source_weight) or source_weight < 0 or
            (source_weight and (paired is None or paired.source is None))):
        raise ValueError("Source loss needs finite nonnegative weight and paired source labels")
    if source_weight and consistency_weight:
        raise ValueError("Source and consistency objectives are separate experiments")
    if (type(freeze_after) is not int or not 0 <= freeze_after < steps or
            type(capture_step) is not int or not 0 <= capture_step <= steps or
            (freeze_after and (not is_parser or not getattr(model, "attachment_roles", False)))):
        raise ValueError("Invalid parser freeze boundary or prefix capture step")
    generator = torch.Generator(device="cpu").manual_seed(seed + 10000)
    amount_generator = torch.Generator(device="cpu").manual_seed(seed + 20000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    metric = cache_parser if is_parser else lambda m, d: evaluate_core(m, mode, d)
    key_name = "event_accuracy" if is_parser else "macro_exact_accuracy"
    sync(device)
    start_time = time.perf_counter()
    def selection_metrics():
        current = metric(model, valid)
        if selection_sets is None:
            return current, (current[key_name], -current["loss"])
        scores = {"iid": current, **{name: metric(model, data)
                                   for name, data in selection_sets.items()}}
        programs = [task["program_accuracy"] for entry in scores.values() for task in entry["per_task"].values()]
        current = {**current, "selection_sets": scores}
        if getattr(args, "protocol", None) == "preservation":
            return current, preservation_selection_key(scores)
        return current, (min(programs), statistics.mean(programs),
                         -statistics.mean(entry["loss"] for entry in scores.values()))
    best_metrics, best_key = selection_metrics()
    best_state, best_step = state_copy(model), 0
    optimize_seconds = 0.0
    examples_seen, bytes_seen, retained_seen = 0, 0, 0
    consistency_total = 0.0
    source_total = 0.0
    preservation, prefix = None, None
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        if freeze_after and step == freeze_after + 1:
            preservation = {"freeze_after": freeze_after, **freeze_parser_for_roles(model),
                            "optimizer_reset": False, "learning_rate_schedule_changed": False}
        parent_batch = args.batch_size // 2 if paired is not None else args.batch_size
        indices_cpu = (scaling_indices(len(train.examples), parent_batch, generator, amount_generator)
                       if getattr(args, "protocol", None) in ("scaling", "binding", "polarity", "attachment", "preservation") else
                       torch.randint(len(train.examples), (parent_batch,), generator=generator))
        indices = indices_cpu.to(device)
        training_data = paired.data if paired is not None else train
        view_indices = torch.stack((2 * indices, 2 * indices + 1), dim=-1).flatten() if paired is not None else indices
        view_indices_cpu = (torch.stack((2 * indices_cpu, 2 * indices_cpu + 1), dim=-1).flatten()
                            if paired is not None else indices_cpu)
        bytes_seen += sum(len(training_data.examples[index].text.encode("utf-8")) for index in view_indices_cpu.tolist())
        examples_seen += len(view_indices)
        if getattr(args, "protocol", None) in ("scaling", "binding", "polarity", "attachment", "preservation"):
            retained_seen += int((indices_cpu < 3000).sum()) * (2 if paired is not None else 1)
        model.train()
        # The same original-example order is used by every answer-training path.
        warmup = min(1.0, step / min(100, max(1, steps // 10)))
        decay = .2 + .8 * (1 + math.cos(math.pi * step / steps)) / 2
        for group in optimizer.param_groups:
            group["lr"] = args.lr * warmup * decay
        sync(device)
        optimize_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if is_parser:
            surface = training_data.surface.take(view_indices)
            if paired is not None:
                options = {"return_token_features": True}
                if paired.source is not None:
                    options["return_source_links"] = True
                output = model(surface, **options)
            else:
                output = model(surface)
            loss = parser_loss(output, training_data.rows[view_indices, :surface.sentence_mask.shape[1]],
                               surface.sentence_mask)
            if paired is not None:
                consistency = polarity_feature_loss(output, paired.alignment[indices])
                consistency_total += float(consistency.detach())
                if consistency_weight:
                    loss = loss + consistency_weight * consistency
                if paired.source is not None:
                    from semantic_model import source_frame_loss
                    sentences, tokens = surface.ids.shape[1:]
                    source_loss = source_frame_loss(output,
                        paired.source.triggers[view_indices, :sentences, :tokens],
                        paired.source.mentions[view_indices, :sentences])
                    source_total += float(source_loss.detach())
                    if source_weight:
                        loss = loss + source_weight * source_loss
        else:
            loss = answer_loss(forward_core(model, mode, train, indices), train.targets[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {mode} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync(device)
        optimize_seconds += time.perf_counter() - optimize_start
        if capture_step and step == capture_step:
            prefix = {"step": step, "parser_sha256": tensor_state_hash(model)}
        if step % args.eval_every == 0 or step == steps:
            scores, key = selection_metrics()
            if key > best_key:
                best_key, best_metrics = key, scores
                best_state, best_step = state_copy(model), step
            print(json.dumps({"seed": seed, "mode": mode, "step": step,
                              key_name: scores[key_name], "loss": scores["loss"],
                              **({"programs": {name: entry["program_accuracy"] for name, entry in
                                                scores["selection_sets"].items()}}
                                 if selection_sets is not None else {})}), flush=True)
    if preservation is not None:
        preservation["final_frozen_sha256"] = parameter_subset_hash(model, preservation["frozen_names"])
        if preservation["final_frozen_sha256"] != preservation["boundary_frozen_sha256"]:
            raise AssertionError("Frozen parser parameters changed during role training")
    model.load_state_dict(best_state)
    if preservation is not None:
        preservation["selected_frozen_sha256"] = parameter_subset_hash(model, preservation["frozen_names"])
        if best_step >= freeze_after and preservation["selected_frozen_sha256"] != preservation["boundary_frozen_sha256"]:
            raise AssertionError("Selected checkpoint changed the preserved representation")
    model.eval()
    sync(device)
    peak = torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None
    return {"steps": steps, "selected_step": best_step, "validation": best_metrics,
            **({"prefix": prefix} if prefix is not None else {}),
            **({"preservation": preservation} if preservation is not None else {}),
            **({"selection_rule": "Primary 99/95 program eligibility first, then worst/mean programs and loss"}
               if getattr(args, "protocol", None) == "preservation" else {}),
            "parameters": parameter_count(model), "examples_seen": examples_seen,
            **({"parent_draws": examples_seen // 2, "consistency_weight": consistency_weight,
                "mean_feature_distance": consistency_total / steps,
                "paired_views_sha256": paired.report["views_sha256"]} if paired is not None else {}),
            **({"source_weight": source_weight, "mean_source_loss": source_total / steps,
                "source_supervision": paired.source.report} if paired is not None and paired.source is not None else {}),
            "original_bytes_seen": bytes_seen, "optimization_seconds": optimize_seconds,
            **({"retained_examples_seen": retained_seen, "new_examples_seen": examples_seen - retained_seen}
               if getattr(args, "protocol", None) in ("scaling", "binding", "polarity", "attachment", "preservation") else {}),
            "train_and_validation_seconds": time.perf_counter() - start_time,
            "peak_cuda_allocated_bytes": peak}


def evaluate_paths(models, data, device):
    """Evaluate parser, learned answer paths, and exact execution together."""

    results = {}
    parser = models["parser"].to(device)
    sync(device)
    started = time.perf_counter()
    results["parser"] = cache_parser(parser, data)
    sync(device)
    results["parser"]["evaluation_seconds"] = time.perf_counter() - started
    parser.cpu()
    for mode in available_modes(models):
        model = models["oracle" if mode == "parsed" else mode].to(device)
        sync(device)
        started = time.perf_counter()
        results[mode] = evaluate_core(model, mode, data)
        sync(device)
        results[mode]["core_evaluation_seconds"] = time.perf_counter() - started
        model.cpu()
    started = time.perf_counter()
    validator = StableReasoner.valid_queries if isinstance(models.get("oracle"), StableReasoner) else None
    results["executor"] = evaluate_executor(data, validator)
    results["executor"]["execution_seconds"] = time.perf_counter() - started
    return results


def train_seed(train_examples, valid_examples, args, seed):
    """Train every requested comparison path from one reproducible seed."""

    torch.manual_seed(seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    train = prepare(train_examples, args.device, args.representation)
    valid = prepare(valid_examples, args.device, args.representation)
    preparation_seconds = time.perf_counter() - started
    parser = SemanticParser(args.width, 2, representation=args.representation).to(args.device)
    training = {"parser": train_component(parser, "parser", train, valid, args, seed)}
    started = time.perf_counter()
    cache_parser(parser, train)
    cache_parser(parser, valid)
    sync(args.device)
    cache_seconds = time.perf_counter() - started
    models = {"parser": parser.cpu()}
    training_modes = ("raw", "oracle", "raw_aux") if args.architecture == "stable" else ("raw", "oracle", "hybrid", "raw_aux")
    for mode in training_modes:
        # Paired initializations are deterministic; source-aux starts from the
        # same trained parser encoder, then updates its own copied parameters.
        torch.manual_seed(seed + 1)
        if mode in ("raw", "raw_aux"):
            model = RawReasoner(args.width, 2, 2, representation=args.representation)
            if mode == "raw_aux":
                model.encoder.load_state_dict(models["parser"].encoder.state_dict())
        else:
            reasoner = {"attention": Reasoner, "bound": BoundReasoner, "stable": StableReasoner}[args.architecture]
            model = reasoner(args.width, 2, source=mode == "hybrid")
        model.to(args.device)
        training[mode] = train_component(model, mode, train, valid, args, seed)
        if mode == "oracle" and args.fit_quantities:
            previous = state_copy(model)
            before = training[mode]["validation"]
            sync(args.device)
            fit_started = time.perf_counter()
            fit = model.fit_quantities(train.rows, train.surface.sentence_mask, train.targets)
            after = evaluate_core(model, mode, valid)
            fit["accepted"] = ((after["macro_exact_accuracy"], -after["loss"]) >=
                               (before["macro_exact_accuracy"], -before["loss"]))
            if not fit["accepted"]:
                model.load_state_dict(previous)
            else:
                training[mode]["validation"] = after
            sync(args.device)
            fit.update({"fit_and_validation_seconds": time.perf_counter() - fit_started,
                        "extra_original_bytes_seen": sum(len(e.text.encode("utf-8")) for e in train.examples
                                                          if e.task == "accounting"),
                        "validation_before": before, "validation_after": after})
            training[mode]["quantity_fit"] = fit
            print(json.dumps({"seed": seed, "mode": "quantity_fit", "accepted": fit["accepted"],
                              "validation_accuracy": after["macro_exact_accuracy"]}), flush=True)
        models[mode] = model.cpu()
    validation = evaluate_paths(models, valid, args.device)
    params = {name: parameter_count(model) for name, model in models.items()}
    effective = {"raw": params["raw"], "oracle": params["oracle"],
                 "parsed": params["parser"] + params["oracle"],
                 "raw_aux": params["raw_aux"], "executor": params["parser"]}
    if "hybrid" in params:
        effective["hybrid"] = params["parser"] + params["hybrid"]
    return models, {"seed": seed, "training": training, "validation": validation,
                    "inference_parameters": effective,
                    "preparation_seconds": preparation_seconds,
                    "parser_cache_seconds": cache_seconds}


# Result aggregation, provenance, and the original structural/stability runner.
def aggregate(trials, key="validation", split=None):
    """Summarize accuracy across seeds without hiding per-task variation."""

    result = {}
    first = trials[0][key][split] if split is not None else trials[0][key]
    for mode in (mode for mode in (*LEARNED_MODES, "executor") if mode in first):
        entries = [trial[key][split][mode] if split is not None else trial[key][mode]
                   for trial in trials]
        values = [entry["macro_exact_accuracy"] for entry in entries]
        per_task = {}
        for task in ("accounting", "relations"):
            task_values = [entry["per_task"][task]["exact_accuracy"] for entry in entries]
            per_task[task] = {"mean": statistics.mean(task_values),
                              "sample_std": statistics.stdev(task_values) if len(task_values) > 1 else 0.0}
        result[mode] = {"mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "per_task": per_task}
    return result


def provenance():
    """Hash every retained source file recorded with an experiment result."""

    return {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ("semantic_tasks.py", "semantic_model.py", "semantic_experiment.py",
                         "test_semantic_v1.py")}


def persist(summary):
    """Update the one existing summary; callers run experiments sequentially."""
    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stability_gates(trials, key="test"):
    """Predeclared per-seed/per-task capability gates, not product guarantees."""
    thresholds = {shift: (.99 if shift == "iid" else .90 if shift == "combined" else .95)
                  for shift in STABILITY_SHIFTS if shift != "lexical_unseen"}
    if key == "development":
        thresholds = {shift: thresholds[shift] for shift in ("iid", "length", "numbers", "composition")}
    result = {}
    for mode in DEPLOYABLE_MODES:
        if mode not in trials[0][key]["iid"]:
            continue
        failures = []
        for trial in trials:
            for shift, threshold in thresholds.items():
                for task, metric in trial[key][shift][mode]["per_task"].items():
                    if metric["exact_accuracy"] < threshold:
                        failures.append({"seed": trial["seed"], "shift": shift, "task": task,
                                         "accuracy": metric["exact_accuracy"], "required": threshold})
        result[mode] = {"passed": not failures, "failures": failures}
    return {"thresholds": thresholds, "lexical_unseen": "Exploratory challenge outside supported-vocabulary gate",
            "modes": result}


def run(args):
    """Run the guarded structural or stability screen/confirmation protocol."""

    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seeds = args.seeds or ([args.seed] if args.seed is not None else
                           ([17] if args.phase == "screen" else [71, 83, 97]))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    started = time.perf_counter()
    stability = args.protocol == "stability"
    maker = make_stability_split if stability else make_split
    train_seed_id, valid_seed_id, test_seed_id = (72001, 72002, 73011) if stability else (62001, 62002, 63011)
    shifts = STABILITY_SHIFTS if stability else SHIFTS
    train = maker(args.train_size, train_seed_id)
    valid = maker(args.valid_size, valid_seed_id, exclude=(example.text for example in train))
    excluded_development = {example.text for example in train + valid}
    development = {}
    if stability:
        # Known failure axes are development evidence; held-out syntax and
        # lexical templates are never evaluated during screening/selection.
        for i, shift in enumerate(("length", "numbers", "composition")):
            examples = maker(200, 72101 + i, shift=shift, exclude=excluded_development)
            development[shift] = examples
            excluded_development.update(example.text for example in examples)
    run_result = {"phase": args.phase, "utc": datetime.now(timezone.utc).isoformat(),
                  "status": "training", "test_accessed": False,
                  "config": {**vars(args), "seeds": seeds}, "source_sha256": provenance(),
                  "data": {"train_seed": train_seed_id, "valid_seed": valid_seed_id,
                           "train_count": len(train), "valid_count": len(valid),
                           "train_sha256": split_fingerprint(train),
                           "valid_sha256": split_fingerprint(valid),
                           "development": {shift: {"seed": 72101 + i, "count": len(examples),
                                                    "sha256": split_fingerprint(examples)}
                                           for i, (shift, examples) in enumerate(development.items())},
                           "deduplication": "Exact raw and normalized input, within and across splits"},
                  "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                  "device": args.device, "threads": torch.get_num_threads(),
                                  "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None},
                  "trials": []}
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    structural = summary.setdefault("stabilization" if stability else "structural", {"runs": []})
    structural.setdefault("runs", []).append(run_result)
    persist(summary)
    bundles = []
    for seed in seeds:
        trial_start = time.perf_counter()
        models, trial = train_seed(train, valid, args, seed)
        if stability:
            trial["development"] = {"iid": trial["validation"]}
            for shift, examples in development.items():
                data = prepare(examples, args.device, args.representation)
                trial["development"][shift] = evaluate_paths(models, data, args.device)
                print(json.dumps({"seed": seed, "development_split": shift,
                                  "scores": {mode: trial["development"][shift][mode]["macro_exact_accuracy"]
                                             for mode in (*available_modes(models), "executor")}}), flush=True)
        trial["total_seconds"] = time.perf_counter() - trial_start
        bundles.append(models)
        run_result["trials"].append(trial)
        persist(summary)
    validation = aggregate(run_result["trials"])
    development_aggregate = ({shift: aggregate(run_result["trials"], "development", shift)
                              for shift in ("iid", *development)} if stability else {})
    modes = tuple(mode for mode in DEPLOYABLE_MODES if mode in validation)
    def selection_key(mode):
        quality = [metric[mode]["per_task"][task]["mean"]
                   for metric in development_aggregate.values() for task in ("accounting", "relations")]
        return (min(quality) if quality else validation[mode]["mean"],
                statistics.mean(quality) if quality else validation[mode]["mean"],
                -statistics.mean(trial["inference_parameters"][mode] for trial in run_result["trials"]))
    best_neural = max((mode for mode in modes if mode != "executor"),
                      key=selection_key)
    best_overall = max(modes, key=selection_key)
    def representative_score(index):
        trial = run_result["trials"][index]
        if stability:
            return min(entry[best_overall]["per_task"][task]["exact_accuracy"]
                       for entry in trial["development"].values() for task in ("accounting", "relations"))
        return trial["validation"][best_overall]["macro_exact_accuracy"]
    ranking = sorted(range(len(seeds)), key=lambda i:
                     representative_score(i))
    representative = ranking[len(ranking) // 2]
    selection = {"phase": args.phase, "best_neural_mode": best_neural,
                 "best_overall_mode": best_overall, "representative_seed": seeds[representative],
                 "criterion": ("Maximize worst mean per-task development accuracy across IID/length/numbers/composition, then mean, then fewer parameters; median worst-task seed; oracle excluded" if stability else
                               "Mean IID validation macro exact accuracy; ties prefer fewer inference parameters; median-validation seed; oracle excluded"),
                 "validation": validation, "development": development_aggregate}
    run_result["selection"] = selection
    run_result["status"] = "selection_frozen"
    if stability:
        run_result["development_gates"] = stability_gates(run_result["trials"], "development")
        ready = all(run_result["development_gates"]["modes"][mode]["passed"]
                    for mode in (best_neural, best_overall))
        if args.phase == "confirm" and not ready:
            run_result["status"] = "development_gate_failed"
            run_result["total_seconds"] = time.perf_counter() - started
            persist(summary)
            print(json.dumps({"phase": args.phase, "status": run_result["status"],
                              "test_accessed": False,
                              "failures": {mode: run_result["development_gates"]["modes"][mode]["failures"]
                                           for mode in (best_neural, best_overall)}}), flush=True)
            return run_result
    if args.phase == "confirm":
        metadata = {"config": {"width": args.width, "local_layers": 2, "layers": 2,
                                "architecture": args.architecture, "representation": args.representation},
                    "protocol": args.protocol,
                    "selected_mode": best_overall, "selection": selection,
                    "source_sha256": run_result["source_sha256"],
                    "scope": "Controlled-language structural experiment; no continual learning"}
        selected = bundles[representative]
        # Freeze/save selection before any final test examples are generated.
        save_checkpoint(CHECKPOINT, selected["parser"], selected["raw"], selected["oracle"],
                        selected.get("hybrid"), metadata, raw_aux=selected["raw_aux"])
        run_result["checkpoint_sha256"] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        structural["selection"] = selection
        persist(summary)
        run_result["test_accessed"] = True
        run_result["status"] = "evaluating_test"
        persist(summary)
        excluded = set(excluded_development)
        tests = {}
        for i, shift in enumerate(shifts):
            examples = maker(args.test_size, test_seed_id + i, shift=shift, exclude=excluded)
            excluded.update(example.text for example in examples)
            tests[shift] = examples
        run_result["data"]["tests"] = {shift: {"seed": test_seed_id + i, "count": len(examples),
                                                "sha256": split_fingerprint(examples)}
                                              for i, (shift, examples) in enumerate(tests.items())}
        for models, trial in zip(bundles, run_result["trials"]):
            trial["test"] = {}
            for shift, examples in tests.items():
                sync(args.device)
                preparation_start = time.perf_counter()
                data = prepare(examples, args.device, args.representation)
                sync(args.device)
                preparation_seconds = time.perf_counter() - preparation_start
                trial["test"][shift] = evaluate_paths(models, data, args.device)
                trial["test"][shift]["text_preparation_seconds"] = preparation_seconds
                persist(summary)
                print(json.dumps({"seed": trial["seed"], "test_split": shift,
                                  "scores": {mode: trial["test"][shift][mode]["macro_exact_accuracy"]
                                             for mode in (*available_modes(models), "executor")}}), flush=True)
        run_result["test_aggregate"] = {shift: aggregate(run_result["trials"], "test", shift)
                                         for shift in tests}
        if stability:
            run_result["capability_gates"] = stability_gates(run_result["trials"])
    run_result["total_seconds"] = time.perf_counter() - started
    run_result["notes"] = [
        "Raw inputs share visible sentence boundaries, first-mention name normalization, and literal number extraction.",
        "Oracle rows are a labeled upper-bound diagnostic; oracle uses fewer parameters than complete text paths.",
        "Parsed is the exact gold-trained oracle neural core with predicted rows; no separate answer fitting.",
        "Hybrid trains a new core on predicted rows plus frozen parser source features; this is a source_feature_residual, not exact-byte retrieval or copying.",
        "Raw_aux starts from the same supervised parser encoder then fine-tunes on answers; it controls for extra semantic-label supervision without an explicit row bottleneck.",
        "Parser is frozen for parsed/hybrid/executor inference; parser training/caching costs and parameter counts are reported separately.",
        "The exact executor is deterministic hybrid computation, not learned neural arithmetic or graph reasoning.",
        "Bound architecture, when selected, supplies exact entity-role binding, additive quantity memory, typed graph edges and three learned message steps. It learns update coefficients and message weights; this is a task-specific structural prior, not unrestricted reasoning.",
        "Name shift tests normalization invariance. Numbers/composition affect accounting/relations respectively; the other task is an unchanged control.",
        "Length means additional transfers or repeated relation facts, with four entities; relation composition separately holds out three-edge queries.",
        "Checkpoint/settings use validation only, with extra development stress axes under stability. Repeating confirmation reuses observed test seeds and is reproduction, not fresh evidence."
    ]
    if stability:
        run_result["notes"] += [
            "Stability training includes previously observed wording; final wording uses new syntax with familiar lexical meanings; lexical_unseen separately challenges absent verbs.",
            "Typed mode uses distinct identity markers and a quantity placeholder, retaining literal values separately; typed parser features cannot depend on quantity magnitude.",
            "Stable core uses only learned quantity/graph messages with explicit query-kind routing and no global correction; the removed source-residual path is not retrained or claimed as an additional model.",
            "IID checkpoint selection is followed by development stress selection before test. Capability gates are fixed per seed/task; test results do not change selected weights or settings.",
            "Architecture size is deliberately reduced; comparison demonstrates a structural design change, not equal-parameter scaling or compression-only causality."
        ]
        if args.fit_quantities:
            run_result["notes"].append("Quantity fitting uses accounting training labels only to solve the existing additive linear update coefficients; no prescribed debit/credit signs or test labels. Extra training exposure and fitting cost are recorded, with IID validation rollback if worse.")
    run_result["status"] = "complete"
    persist(summary)
    print(json.dumps({"phase": args.phase, "best_neural_mode": best_neural,
                      "best_overall_mode": best_overall,
                      "representative_seed": seeds[representative],
                      "mean_validation": {mode: values["mean"] for mode, values in validation.items()},
                      "total_seconds": run_result["total_seconds"]}), flush=True)
    return run_result


# Interpretation, scaling, binding, and polarity diagnostics.
def tensor_state_hash(model):
    """Hash a full tensor state for exact frozen-core comparisons."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parser_errors(data, limit=8):
    """Small diagnostic sample; full generated prompts are not logged."""
    errors, seen = [], set()
    predicted = data.predicted.detach().cpu().long().tolist()
    for example, rows in zip(data.examples, predicted):
        for sentence, expected, actual in zip(example.sentences, example.rows, rows):
            if tuple(actual) == expected:
                continue
            pattern = re.sub(r"\b[A-Z][a-z]+\b", "ENTITY", sentence)
            pattern = re.sub(r"(?<![A-Za-z0-9])-?\d+\b", "NUMBER", pattern)
            if pattern not in seen:
                errors.append({"sentence": sentence, "expected": expected, "predicted": actual})
                seen.add(pattern)
            if len(errors) == limit:
                return errors
    return errors


def interpretation_gates(trials, key="test", require_retention=False, strict=False):
    """Require both faithful programs and answers, independently per seed/task."""
    if not trials:
        raise ValueError("Interpretation gates require completed trials")
    thresholds = {"iid": .99, "names": .99, "wording": .95, "length": .95,
                  "numbers": .95, "composition": .95, "combined": .95 if strict else .90}
    if require_retention or any("retention" in trial[key] for trial in trials):
        thresholds["retention"] = .99
    failures = []
    for trial in trials:
        required_shifts = set(thresholds) - ({"names"} if key == "development" else set())
        if required_shifts - trial[key].keys():
            raise ValueError("Interpretation gates require every primary split")
        for shift, results in trial[key].items():
            if shift not in thresholds:
                continue
            required = thresholds[shift]
            if shift in ("wording", "combined") and "all_queries" not in results:
                raise ValueError("Interpretation gates require all-query world metrics")
            if strict and shift in ("wording", "combined") and "contrasts" not in results:
                raise ValueError("Binding gates require role/polarity contrast metrics")
            for task in ("accounting", "relations"):
                measures = {"program": results["parser"]["per_task"][task]["program_accuracy"],
                            **{mode: results[mode]["per_task"][task]["exact_accuracy"]
                               for mode in ("parsed", "executor")}}
                if "all_queries" in results:
                    measures["all_queries_world"] = results["all_queries"]["per_task"][task]["all_correct"]
                    if strict:
                        measures["all_queries_programs"] = results["all_queries"]["per_task"][task]["all_programs_correct"]
                if strict and "contrasts" in results:
                    measures["contrast_answers"] = results["contrasts"]["per_task"][task]["all_correct"]
                    measures["contrast_programs"] = results["contrasts"]["per_task"][task]["all_programs_correct"]
                for metric, value in measures.items():
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Interpretation gates require finite accuracies between zero and one")
                    if value < required:
                        failures.append({"seed": trial["seed"], "shift": shift, "task": task,
                                         "metric": metric, "accuracy": value, "required": required})
    return {"passed": not failures, "thresholds": thresholds, "failures": failures}


def evaluate_query_worlds(models, groups, device, changed_clauses=None, group_kinds=None):
    """Evaluate correlated query expansions as worlds, never as independent samples."""
    parser = models["parser"].to(device)
    core = models["oracle"].to(device)
    totals = {task: {"worlds": 0, "all_correct": 0, "all_programs_correct": 0, "queries": 0, "correct": 0}
              for task in ("accounting", "relations")}
    if not groups or (changed_clauses is not None and len(changed_clauses) != len(groups)):
        raise ValueError("Grouped evaluation needs nonempty groups and aligned changed clauses")
    if group_kinds is not None and len(group_kinds) != len(groups):
        raise ValueError("Contrast kinds must align with groups")
    by_kind = {}
    if changed_clauses is not None:
        for entry in totals.values():
            entry.update(target_rows_correct=0, roles_consistent=0)
    flat = [example for group in groups for example in group]
    data = prepare(flat, device, parser.representation, getattr(parser, "vocabulary", None))
    cache_parser(parser, data)
    programs = (((data.predicted == data.rows).all(-1) | ~data.surface.sentence_mask).all(-1)).cpu().tolist()
    answers = []
    with torch.no_grad():
        for start in range(0, len(flat), 128):
            indices = torch.arange(start, min(start + 128, len(flat)), device=device)
            surface = data.surface.take(indices)
            rows = data.predicted[indices, :surface.sentence_mask.shape[1]]
            answers.extend(decode_answers(core(rows, surface.sentence_mask),
                                          core.valid_queries(rows, surface.sentence_mask)))
    cursor = 0
    for group_index, group in enumerate(groups):
        hits = [answer == example.answer for answer, example in zip(answers[cursor:cursor + len(group)], group)]
        entry = totals[group[0].task]
        entry["worlds"] += 1
        entry["all_correct"] += all(hits)
        entry["all_programs_correct"] += all(programs[cursor:cursor + len(group)])
        entry["queries"] += len(hits)
        entry["correct"] += sum(hits)
        if group_kinds is not None:
            separate = by_kind.setdefault(group_kinds[group_index], {}).setdefault(group[0].task,
                {"worlds": 0, "all_correct_count": 0, "all_programs_correct_count": 0})
            separate["worlds"] += 1
            separate["all_correct_count"] += all(hits)
            separate["all_programs_correct_count"] += all(programs[cursor:cursor + len(group)])
        if changed_clauses is not None:
            clause = changed_clauses[group_index]
            if clause < 0 or any(clause >= len(example.rows) - 1 for example in group):
                raise ValueError("Polarity probe must identify a fact in every view")
            predicted = data.predicted[cursor:cursor + len(group), clause]
            target = data.rows[cursor:cursor + len(group), clause]
            entry["target_rows_correct"] += int((predicted == target).all())
            entry["roles_consistent"] += int((predicted[:, :4] == predicted[:1, :4]).all())
        cursor += len(group)
    parser.cpu()
    core.cpu()
    oracle_scores = evaluate_core(core.to(device), "oracle", data)
    core.cpu()
    return {"per_task": {task: {**entry, "all_correct_count": entry["all_correct"],
                                 "all_correct": entry["all_correct"] / entry["worlds"],
                                 "all_programs_correct": entry["all_programs_correct"] / entry["worlds"],
                                 **({"target_row_accuracy": entry["target_rows_correct"] / entry["worlds"],
                                     "role_consistency_accuracy": entry["roles_consistent"] / entry["worlds"]}
                                    if changed_clauses is not None else {}),
                                 "query_accuracy": entry["correct"] / entry["queries"]}
                          for task, entry in totals.items() if entry["worlds"]},
            "world_count": len(groups), "query_count": len(flat),
            **({"by_contrast": {kind: {task: {**entry,
                  "all_correct": entry["all_correct_count"] / entry["worlds"],
                  "all_programs_correct": entry["all_programs_correct_count"] / entry["worlds"]}
                  for task, entry in tasks.items()} for kind, tasks in by_kind.items()}}
               if group_kinds is not None else {}),
            "oracle_input_core": oracle_scores}


# Corpus construction keeps training, development, confirmation, and final
# worlds disjoint even when several protocol variants share a study.
def retention_split(count, seed, exclude=(), candidate_count=None):
    """Old supported grammar, with new presented worlds and balanced answers."""
    from semantic_tasks import interpretation_world_key
    excluded = set(exclude)
    worlds = {interpretation_world_key(text) for text in excluded}
    accounting = count // 2
    relations = count - accounting
    remaining = {"accounting": accounting, "yes": (relations + 1) // 2, "no": relations // 2}
    examples = []
    for example in make_stability_split(candidate_count or count * 8, seed, exclude=excluded):
        label = "accounting" if example.task == "accounting" else example.answer
        world = interpretation_world_key(example.text)
        if remaining[label] and world not in worlds:
            examples.append(example)
            worlds.add(world)
            remaining[label] -= 1
            if len(examples) == count:
                return examples
    raise ValueError("Not enough distinct old-style worlds for the retention split")


def scaling_data(train_size, diversity, valid_size, development_size):
    """Use one common exclusion universe for the entire capacity/data study."""
    from semantic_tasks import make_scaling_split
    if diversity not in ("narrow", "broad") or train_size not in (6000, 18000):
        raise ValueError("Scaling uses 6000 narrow/broad or 18000 broad new examples")
    if diversity == "narrow" and train_size != 6000:
        raise ValueError("The narrow control has a fixed 6000-example corpus")
    retained = make_stability_split(3000, 72001)
    old_texts = {example.text for example in retained}
    narrow = make_scaling_split(6000, 92001, exclude=old_texts, diversity="narrow")
    broad = make_scaling_split(18000, 92001, exclude=old_texts, diversity="broad")
    train = retained + (narrow if diversity == "narrow" else broad[:train_size])
    excluded = {example.text for example in retained + narrow + broad}
    valid = make_scaling_split(valid_size, 92002, exclude=excluded, diversity="broad")
    excluded.update(example.text for example in valid)
    development = {"iid": valid}
    development["retention"] = retention_split(400, 92401, exclude=excluded, candidate_count=9600)
    excluded.update(example.text for example in development["retention"])
    for i, shift in enumerate(("wording", "combined", "length", "numbers", "composition")):
        examples = make_scaling_split(development_size, 92101 + i, shift, exclude=excluded,
                                      stage="development", diversity="broad")
        development[shift] = examples
        excluded.update(example.text for example in examples)
    universe = {"narrow6000": split_fingerprint(narrow), "broad18000": split_fingerprint(broad),
                "retained3000": split_fingerprint(retained),
                "policy": "Every split excludes the union of all study training corpora, independent of selected cell"}
    return train, retained, development, excluded, universe



def binding_data(train_size, valid_size, development_size, balance_quantities=False):
    """Fresh compositional study with identical training/evaluation across parsers."""
    from semantic_tasks import make_binding_split
    if train_size != 6000:
        raise ValueError("Binding study fixes 6000 new plus 3000 retained examples")
    retained = make_stability_split(3000, 72001)
    excluded = {example.text for example in retained}
    baseline = make_binding_split(train_size, 110001, exclude=excluded)
    balanced = make_binding_split(train_size, 110001, exclude=excluded, balance_quantities=True)
    train = retained + (balanced if balance_quantities else baseline)
    excluded.update(example.text for example in baseline + balanced)
    valid = make_binding_split(valid_size, 110002, exclude=excluded)
    excluded.update(example.text for example in valid)
    development = {"iid": valid}
    development["retention"] = retention_split(400, 110401, exclude=excluded, candidate_count=9600)
    excluded.update(example.text for example in development["retention"])
    for i, shift in enumerate(("wording", "combined", "length", "numbers", "composition")):
        examples = make_binding_split(development_size, 110101 + i, shift,
                                      exclude=excluded, stage="development")
        development[shift] = examples
        excluded.update(example.text for example in examples)
    return train, retained, development, excluded


def binding_final_data(args, excluded):
    """Data-only construction for split isolation; never score models here."""
    from semantic_tasks import INTERPRETATION_SHIFTS, make_binding_split
    excluded, tests = set(excluded), {}
    for index, shift in enumerate(INTERPRETATION_SHIFTS):
        count = min(args.test_size, args.wording_test_size) if shift == "wording" else args.test_size
        examples = make_binding_split(count, 111011 + index, shift, exclude=excluded, stage="final")
        tests[shift] = examples
        excluded.update(example.text for example in examples)
    tests["retention"] = retention_split(min(args.test_size, args.retention_test_size), 111401,
                                         exclude=excluded, candidate_count=9600)
    return tests


# Paired polarity/source studies change one controlled property at a time.
def prepare_polarity_training(train, development, tests, device, vocabulary):
    """Keep counterfactual descendants in their source partition, with matched views."""
    from semantic_tasks import TRANSFER, polarity_pair, polarity_world_key, interpretation_world_key, surface_names
    holdouts = [(example, stage) for splits, stage in ((development, "development"), (tests, "final"))
                for examples in splits.values() for example in examples]
    holdout_origins = {stage: {polarity_world_key(example, stage) for example, partition in holdouts
                              if partition == stage} for stage in ("development", "final")}
    if holdout_origins["development"] & holdout_origins["final"]:
        raise ValueError("Development/final polarity-origin overlap")
    forbidden_origins = holdout_origins["development"] | holdout_origins["final"]
    forbidden_worlds = {interpretation_world_key(example.text) for example, _ in holdouts}
    train_origins = [polarity_world_key(example, "train") for example in train]
    overlap = set(train_origins) & forbidden_origins
    if overlap:
        raise ValueError(f"Training/holdout polarity-origin overlap: {len(overlap)} groups")
    rng, views, clauses = random.Random(120003), [], []
    report = {"parents": len(train), "paired_parents": 0, "identity_parents": 0,
              "colliding_augmentations": 0, "zero_amount_pairs": 0,
              "origin_overlap": 0, "development_final_origin_overlap": 0,
              "holdout_origin_groups": len(forbidden_origins),
              "pair_seed": 120003, "split_policy": "All polarity descendants retain their originating partition"}
    for example in train:
        candidates = [index for index, row in enumerate(example.rows) if row[0] == TRANSFER]
        clause = rng.choice(candidates) if candidates else -1
        pair = polarity_pair(example, clause, "train") if clause >= 0 else (example, example)
        if any(interpretation_world_key(view.text) in forbidden_worlds for view in pair):
            report["colliding_augmentations"] += 1
            pair, clause = (example, example), -1
        if surface_names(pair[0].text) != surface_names(pair[1].text):
            raise ValueError("Polarity transformation changed canonical identity order")
        report["paired_parents" if clause >= 0 else "identity_parents"] += 1
        report["zero_amount_pairs"] += int(clause >= 0 and example.rows[clause][3] == 0)
        views.extend(pair)
        clauses.append(clause)
    from semantic_model import fit_lexicon
    if fit_lexicon(example.text for example in views) != vocabulary:
        raise ValueError("Paired views changed the frozen training vocabulary")
    data = prepare(views, device, "lexical", vocabulary)
    alignment = polarity_alignment(data.surface, clauses)
    report.update(views=len(views), views_sha256=split_fingerprint(views),
                  original_parent_sha256=split_fingerprint(train),
                  aligned_mentions=int((alignment[:, :, 0] >= 0).sum()),
                  rendered_bytes=sum(len(example.text.encode("utf-8")) for example in views),
                  final_data_generated_for_isolation=True, final_model_evaluated=False)
    return PairedTraining(data, alignment, report)


def polarity_probe_sets(development, observed_cases):
    """Build direct polarity probes and reproduce every known failure once."""

    from semantic_tasks import direct_polarity_groups
    probes, observed, reconstructed = {}, [], []
    known = {(case["shift"], tuple(case["origin_key"]), case["clause_index"]) for case in observed_cases}
    if not known or len(known) != len(observed_cases):
        raise ValueError("Observed polarity counterexamples must have unique nonempty keys")
    for name in ("wording", "combined"):
        groups, coverage = direct_polarity_groups(development[name], stage="development")
        probes[name] = (groups, coverage)
        inclusive, _ = direct_polarity_groups(development[name], stage="development", include_zero=True)
        for group in inclusive:
            key = (name, group.origin_key, group.clause_index)
            if key in known:
                observed.append(group)
                reconstructed.append(key)
    if set(reconstructed) != known or len(reconstructed) != len(known):
        raise ValueError("Observed polarity counterexamples were not reconstructed exactly once")
    return probes, observed


def evaluate_polarity_groups(models, groups, coverage, device):
    """Score complete paired worlds, target rows, roles, and final answers."""

    result = evaluate_query_worlds(models, [group.examples for group in groups], device,
                                   [group.clause_index for group in groups])
    return {**result, "coverage": coverage}


def oracle_source_rows(output, surface, labels, replace_links=False):
    """Explicit labeled intervention; never used by ordinary text inference."""
    trigger = labels.triggers.to(output["trigger_log_probs"].dtype)
    annotated = trigger.any(-1)
    trigger = trigger / trigger.sum(-1, keepdim=True).clamp_min(1)
    links = output["link_log_probs"].exp()
    if replace_links:
        # Every co-referring predicate supports the same two semantic arguments.
        links = F.one_hot(labels.mentions.clamp_min(0), surface.ids.shape[-1]).to(links.dtype)
        links = links.unsqueeze(-2).expand_as(output["link_log_probs"])
    mentions = (links * trigger[:, :, None, :, None]).sum(-2)
    identities = F.one_hot(surface.entity_ids.clamp_min(0), 4).to(mentions.dtype)
    identities = identities * (surface.entity_ids >= 0).unsqueeze(-1)
    probabilities = torch.einsum("bsri,bsie->bsre", mentions, identities)
    corrected = dict(output)
    for role, key in enumerate(("a", "b")):
        logits = torch.cat((probabilities[:, :, role].clamp_min(1e-12).log(), output[key][..., 4:]), -1)
        corrected[key] = torch.where(annotated.unsqueeze(-1), logits, output[key])
    return parsed_rows(corrected, surface)


@torch.no_grad()
def evaluate_source_diagnostic(models, data, stage, device):
    """Measure predicted source links and a clearly labeled oracle ceiling."""
    labels = prepare_source_labels(data.examples, data.surface, stage)
    parser = models["parser"].to(device).eval()
    interventions = {name: torch.zeros_like(data.rows) for name in ("gold_predicates", "gold_predicates_and_links")}
    predicate_hits = link_hits = frame_hits = clauses = predicates = 0
    for start in range(0, len(data.examples), 128):
        indices = torch.arange(start, min(start + 128, len(data.examples)), device=device)
        surface = data.surface.take(indices)
        sentences, tokens = surface.ids.shape[1:]
        targets = SourceLabels(labels.triggers[indices, :sentences, :tokens],
                               labels.mentions[indices, :sentences], {})
        output = parser(surface, return_source_links=True)
        mask = targets.triggers
        annotated = mask.any(-1)
        chosen = output["trigger_log_probs"].argmax(-1)
        correct_predicate = mask.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        predicted_links = output["link_log_probs"].argmax(-1)
        correct_links = predicted_links == targets.mentions.unsqueeze(-1)
        at_chosen = correct_links.gather(-1, chosen[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1).all(-1)
        clauses += int(annotated.sum())
        predicates += int(mask.sum())
        predicate_hits += int((correct_predicate & annotated).sum())
        link_hits += int((correct_links & mask.unsqueeze(-2)).sum())
        frame_hits += int((correct_predicate & at_chosen & annotated).sum())
        for name, gold_links in (("gold_predicates", False), ("gold_predicates_and_links", True)):
            interventions[name][indices, :sentences] = oracle_source_rows(output, surface, targets, gold_links)
    parser.cpu()
    core = models["oracle"].to(device).eval()
    ceilings = {}
    for name, rows in interventions.items():
        intervention = replace(data, predicted=rows)
        ceilings[name] = {"parser": metric_parser(intervention),
                          "parsed": evaluate_core(core, "parsed", intervention),
                          "executor": evaluate_executor(intervention, core.valid_queries)}
    core.cpu()
    return {"scope": "Gold source/link interventions are diagnostic labels, never deployable inference inputs",
            "labels": labels.report, "transfer_clauses": clauses, "predicate_occurrences": predicates,
            "predicate_top1_accuracy": predicate_hits / clauses if clauses else None,
            "links_at_gold_predicates_accuracy": link_hits / (2 * predicates) if predicates else None,
            "source_frame_accuracy": frame_hits / clauses if clauses else None, "oracle_interventions": ceilings}


def polarity_gates(trials, key="development"):
    """Direct tests require correct roles, not just invariant wrong predictions."""
    if not trials:
        raise ValueError("Polarity gates require completed trials")
    failures = []
    for trial in trials:
        checks = [(f"{name}_direct_polarity", trial[key][name]["direct_polarity"], .95)
                  for name in ("wording", "combined")]
        if key == "development":
            checks.append(("observed_polarity", trial["observed_polarity"], 1.0))
        for name, result, required in checks:
            measures = result["per_task"]["accounting"]
            for metric in ("all_correct", "all_programs_correct", "target_row_accuracy", "role_consistency_accuracy"):
                value = measures[metric]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Polarity gates need finite accuracies")
                if value < required:
                    failures.append({"seed": trial["seed"], "shift": name, "task": "accounting",
                                     "metric": metric, "accuracy": value, "required": required})
    return {"passed": not failures, "failures": failures}


# Preservation gates protect known behavior while role parameters are repaired.
def preservation_routine_examples(development, records):
    """Reconstruct every recorded routine failure without adding training data."""
    if len(records) != 3:
        raise ValueError("Preservation requires all three known routine cases")
    indices, examples = set(), []
    for record in records:
        index = record["world_index"]
        if type(index) is not int or not 0 <= index < len(development["iid"]) or index in indices:
            raise ValueError("Known routine indices must be unique and in the frozen IID split")
        indices.add(index)
        example = development["iid"][index]
        if (example.text != record["text"] or example.answer != record["answer"] or
                example.task != record["task"] or not record["faults"]):
            raise ValueError("Known routine case does not match the frozen development example")
        clauses = set()
        for fault in record["faults"]:
            clause = fault["clause_index"]
            if (type(clause) is not int or not 0 <= clause < len(example.rows) or clause in clauses or
                    list(example.rows[clause]) != fault["expected"] or
                    example.sentences[clause] != fault["sentence"]):
                raise ValueError("Known routine clause does not match its recorded source and labels")
            clauses.add(clause)
        examples.append(example)
    return examples


def preservation_gates(trials):
    """All recorded routine failures must be fixed without changing other gates."""
    if not trials:
        raise ValueError("Preservation gates require completed trials")
    failures = []
    for trial in trials:
        result = trial["routine_regression"]
        values = {"program": result["parser"]["program_accuracy"],
                  **{mode: result[mode]["macro_exact_accuracy"] for mode in ("parsed", "executor")}}
        for metric, value in values.items():
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Known routine gate requires finite accuracies")
            if value < 1:
                failures.append({"seed": trial["seed"], "shift": "observed_routine_errors",
                                 "task": "accounting", "metric": metric, "accuracy": value, "required": 1.0})
    return {"passed": not failures, "failures": failures}


def polarity_diagnosis(args):
    """Audit every error in the exactly reproduced, already observed baseline."""
    from collections import Counter
    from semantic_tasks import transfer_provenance, interpretation_world_key, direct_polarity_groups
    torch.set_num_threads(4)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    baseline = summary["polarity"]["baseline"]
    models, _ = load_checkpoint(CANDIDATE)
    if tensor_state_hash(models["parser"]) != baseline["parser_sha256"]:
        raise ValueError("Diagnosis requires the reproduced width192 baseline; current candidate differs")
    _, _, development, _ = binding_data(6000, 400, 200, True)
    report = {"parser_sha256": baseline["parser_sha256"], "source_sha256": provenance(),
              "test_accessed": False, "sets": {}, "observed_cases": []}
    for name in ("wording", "combined"):
        examples = development[name]
        data = prepare(examples, args.device, "lexical", models["parser"].vocabulary)
        metrics = evaluate_paths(models, data, args.device)
        census, signatures, samples = Counter(), Counter(), []
        for world, (example, predicted) in enumerate(zip(examples, data.predicted.cpu().long().tolist())):
            for clause, (sentence, expected, actual) in enumerate(zip(example.sentences, example.rows, predicted)):
                census["clauses"] += 1
                if tuple(actual) == expected:
                    continue
                census["wrong_clauses"] += 1
                census["inactive_errors" if not expected[4] else "active_errors"] += 1
                for field, truth, value in zip(("kind", "a", "b", "value", "active"), expected, actual):
                    census[f"wrong_{field}"] += int(truth != value)
                trace = transfer_provenance(example, clause, stage="development") if expected[0] == 2 else None
                if trace:
                    signature = (trace.source_family, tuple(trace.lexical_fields), expected[4])
                    signatures[str(signature)] += 1
                    report["observed_cases"].append({"shift": name, "world_index": world,
                        "origin_key": interpretation_world_key(example.text), "clause_index": clause,
                        "amount": expected[3]})
                if len(samples) < 6:
                    samples.append({"sentence": sentence, "expected": expected, "predicted": actual,
                                    "family": trace.source_family if trace else None})
        groups, coverage = direct_polarity_groups(examples, stage="development")
        direct = evaluate_polarity_groups(models, groups, coverage, args.device)
        report["sets"][name] = {"metrics": metrics, "census": dict(census),
                               "signatures": dict(signatures), "examples": samples, "direct_polarity": direct,
                               "sha256": split_fingerprint(examples)}
    _, observed = polarity_probe_sets(development, report["observed_cases"])
    report["observed_polarity"] = evaluate_polarity_groups(models, observed,
        {"known_clauses": len(report["observed_cases"]), "zero_amounts_included": True}, args.device)
    if tensor_state_hash(models["parser"]) != baseline["parser_sha256"]:
        raise AssertionError("Diagnostic changed parser weights")
    summary["polarity"]["diagnosis"] = report
    persist(summary)
    print(json.dumps({"diagnosis": {name: entry["census"] for name, entry in report["sets"].items()},
                      "known_counterexamples": len(report["observed_cases"]), "test_accessed": False}), flush=True)
    return report


def development_rank(result):
    """Rank recipes using development programs only; final scores are ignored."""
    trials = result.get("trials", [])
    if not trials:
        raise ValueError("Recipe ranking requires completed development trials")
    shifts = {"iid", "retention", "wording", "combined", "length", "numbers", "composition"}
    values, counts = [], []
    for trial in trials:
        development = trial.get("development", {})
        if set(development) != shifts:
            raise ValueError("Recipe ranking requires the same complete development shifts")
        for entry in development.values():
            tasks = entry["parser"]["per_task"]
            if set(tasks) != {"accounting", "relations"}:
                raise ValueError("Recipe ranking requires both tasks in every shift")
            values.extend(task["program_accuracy"] for task in tasks.values())
        counts.append(trial["inference_parameters"]["parsed"])
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts) or len(set(counts)) != 1:
        raise ValueError("Recipe ranking requires one positive parameter count across seeds")
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Recipe ranking requires finite development accuracies")
    return min(values), statistics.mean(values), -counts[0]


# Candidate construction and guarded interpretation/preservation protocols.
def interpretation_parser(parser_type, width, layers, vocabulary):
    """Construct the declared parser variant without changing its interface."""

    from semantic_model import LexicalParser
    if parser_type == "legacy":
        return SemanticParser(width, layers, representation="typed")
    relative = parser_type in ("lexical_relative", "lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment")
    return LexicalParser(width, layers, vocabulary=vocabulary,
                         pooling="clause" if parser_type == "lexical_clause" else "pointer",
                         relative_positions=relative,
                         constrained_roles=parser_type in ("lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment"),
                         local_roles=parser_type == "lexical_local",
                         joint_roles=parser_type == "lexical_joint",
                         evidence_roles=parser_type == "lexical_evidence",
                         attachment_roles=parser_type == "lexical_attachment")



def interpretation_controls(args):
    """Name architecture, capacity and paired-data controls without final scores."""
    controls = [(name, name, args.width, "candidate") for name in args.comparison_parsers]
    controls += [(f"{args.parser_type}_width{width}", args.parser_type, width, "candidate")
                 for width in args.comparison_widths]
    if ((args.protocol == "polarity" and args.pair_consistency > 0) or
            (args.protocol == "attachment" and args.source_weight > 0)):
        controls.append(("paired_ce", args.parser_type, args.width, "paired_ce"))
    if args.protocol == "preservation":
        controls.append(("full_model", args.parser_type, args.width, "full_model"))
    if getattr(args, "comparison_unbalanced", False):
        if args.protocol != "binding" or not args.balance_quantities:
            raise ValueError("An unbalanced control requires balanced binding training")
        controls.append((f"{args.parser_type}_unbalanced", args.parser_type, args.width, "unbalanced"))
    return controls


def scaling_diagnostic_gate(trials):
    """The observed role-distance counterexample is a regression requirement."""
    failures = []
    for trial in trials:
        metrics = trial["role_distance_diagnostic"]
        values = {"program": metrics["parser"]["program_accuracy"],
                  **{mode: metrics[mode]["macro_exact_accuracy"] for mode in ("parsed", "executor")}}
        for metric, value in values.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Role-distance diagnostics require finite accuracies")
            if value < 1:
                failures.append({"seed": trial["seed"], "shift": "observed_role_distance",
                                 "task": "accounting", "metric": metric, "accuracy": value, "required": 1.0})
    return {"passed": bool(trials) and not failures, "failures": failures,
            "scope": "Two observed diagnostic prompts; regression requirement, not fresh generalization evidence"}


def run_interpretation(args):
    """Parser-only comparisons; all candidates share the existing frozen core."""
    from semantic_model import LexicalParser, fit_lexicon
    from semantic_tasks import (INTERPRETATION_SHIFTS, make_interpretation_split,
                                all_query_examples, interpretation_contrast_pairs)
    scaling = args.protocol == "scaling"
    preservation = args.protocol == "preservation"
    attachment = args.protocol in ("attachment", "preservation")
    polarity = args.protocol in ("polarity", "attachment", "preservation")
    binding = args.protocol in ("binding", "polarity", "attachment", "preservation")
    if binding:
        from semantic_tasks import make_binding_split
        if not args.retain_stability:
            raise ValueError("Binding comparisons require retained original training")
        make_examples = make_binding_split
    elif scaling:
        from semantic_tasks import make_scaling_split
        if not args.retain_stability:
            raise ValueError("Scaling comparisons must retain the same old training examples")
        make_examples = lambda count, seed, shift="iid", **options: make_scaling_split(
            count, seed, shift, diversity="broad", **options)
    else:
        make_examples = make_interpretation_split
    expansion_options = {"protocol": "binding" if binding else args.protocol, "diversity": "broad"} if scaling or binding else {}
    if polarity and (args.parser_type != ("lexical_attachment" if attachment else "lexical_constrained") or args.width != 192 or
                     args.parser_steps != 1500 or args.batch_size != 32 or not args.balance_quantities or
                     (args.train_size, args.valid_size, args.development_size, args.test_size,
                      args.wording_test_size, args.retention_test_size) != (6000, 400, 200, 600, 600, 160) or
                     args.comparison_parsers or args.comparison_widths or args.comparison_unbalanced):
        raise ValueError("Polarity/source study fixes the declared parser192, 1500 updates, batch32 and existing balanced splits")
    if attachment and args.pair_consistency != 0:
        raise ValueError("Attachment comparison excludes feature-consistency loss")
    if preservation and (args.freeze_after != 300 or args.source_weight != 0):
        raise ValueError("Preservation fixes freeze after300 and excludes source auxiliary loss")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    seeds = args.seeds or ([args.seed] if args.seed is not None else
                           ([23] if args.phase == "screen" else [101, 113, 127]))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    if args.phase == "confirm" and len(seeds) < 3:
        raise ValueError("Interpretation confirmation requires at least three seeds")
    if len(set(args.comparison_parsers)) != len(args.comparison_parsers) or args.parser_type in args.comparison_parsers:
        raise ValueError("Comparison parsers must be distinct from one another and the candidate")
    if any(width <= 0 or width % 4 or width == args.width for width in args.comparison_widths):
        raise ValueError("Comparison widths must be positive multiples of four and differ from the candidate")
    if len(set(args.comparison_widths)) != len(args.comparison_widths):
        raise ValueError("Comparison widths must be distinct")
    started = time.perf_counter()
    reference, reference_metadata = load_checkpoint(CHECKPOINT)
    if not isinstance(reference.get("oracle"), StableReasoner):
        raise ValueError("Interpretation requires the saved stable computation core")
    reference = {name: reference[name] for name in ("parser", "oracle")}
    core_hash = tensor_state_hash(reference["oracle"])
    universe = None
    if binding:
        train, retained, development, excluded = binding_data(
            args.train_size, args.valid_size, args.development_size, args.balance_quantities)
        valid = development["iid"]
    elif scaling:
        train, retained, development, excluded, universe = scaling_data(
            args.train_size, args.diversity, args.valid_size, args.development_size)
        valid = development["iid"]
    else:
        retained = make_stability_split(3000, 72001) if args.retain_stability else []
        train = retained + make_examples(args.train_size, 82001,
                                        exclude=(example.text for example in retained))
        excluded = {example.text for example in train}
        valid = make_examples(args.valid_size, 82002, exclude=excluded)
        excluded.update(example.text for example in valid)
        development = {"iid": valid}
        if args.retain_stability:
            development["retention"] = retention_split(600, 82401, exclude=excluded)
            excluded.update(example.text for example in development["retention"])
        for i, shift in enumerate(("wording", "combined", "length", "numbers", "composition")):
            examples = make_examples(args.development_size, 82101 + i, shift,
                                     exclude=excluded, stage="development")
            development[shift] = examples
            excluded.update(example.text for example in examples)
    vocabulary = fit_lexicon(example.text for example in train) if args.parser_type.startswith("lexical") else None
    if args.comparison_parsers and vocabulary is None:
        raise ValueError("Lexical comparison controls require a lexical candidate")
    representation = "lexical" if vocabulary is not None else "typed"
    prepared_train = prepare(train, args.device, representation, vocabulary)
    unbalanced_train = None
    if getattr(args, "comparison_unbalanced", False):
        interpretation_controls(args)  # Validate the requested paired control.
        unbalanced_examples = retained + make_binding_split(args.train_size, 110001,
                                                              exclude=(e.text for e in retained))
        if ([(e.rows, e.answer, e.task) for e in unbalanced_examples] !=
                [(e.rows, e.answer, e.task) for e in train] or
                fit_lexicon(e.text for e in unbalanced_examples) != vocabulary):
            raise AssertionError("Quantity placement control changed labels or the frozen vocabulary")
        unbalanced_train = prepare(unbalanced_examples, args.device, representation, vocabulary)
    prepared_dev = {name: prepare(examples, args.device, representation, vocabulary)
                    for name, examples in development.items()}
    paired, frozen_tests, direct_probes, observed_probes = None, None, None, None
    routine_regression = None
    if polarity:
        frozen_tests = binding_final_data(args, excluded)
        paired = prepare_polarity_training(train, development, frozen_tests, args.device, vocabulary)
        if attachment:
            paired.source = prepare_source_labels(paired.data.examples, paired.data.surface, stage="train")
            paired.report["source_supervision"] = paired.source.report
        previous = json.loads(SUMMARY.read_text(encoding="utf-8"))
        direct_probes, observed_probes = polarity_probe_sets(development, previous["polarity"]["diagnosis"]["observed_cases"])
        if preservation:
            routine_examples = preservation_routine_examples(development, previous["attachment"]["control_replay"]["routine_errors"])
            routine_regression = prepare(routine_examples, args.device, representation, vocabulary)
    result = {"phase": args.phase, "utc": datetime.now(timezone.utc).isoformat(),
              "status": "training", "test_accessed": False,
              "config": {**vars(args), "seeds": seeds, "representation": representation},
              "source_sha256": provenance(), "frozen_core_sha256": core_hash,
              "reference_checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "reference_selection": reference_metadata["selection"],
              "data": {"train": {"count": len(train), "seed": 110001 if binding else 92001 if scaling else 82001, "sha256": split_fingerprint(train)},
                       "retained_training": {"count": len(retained), "seed": 72001,
                                             "sha256": split_fingerprint(retained)},
                       "development": {name: {"count": len(examples), "sha256": split_fingerprint(examples)}
                                       for name, examples in development.items()},
                       "vocabulary": vocabulary,
                       **({"study_training_universe": universe} if scaling else {}),
                       "deduplication": "Raw and normalized prompts plus normalized presented facts excluding final query; abstract isomorphic worlds may recur"},
              "trials": [],
              "notes": ["Only parser weights change. The exact same previously trained computation core is frozen in every trial.",
                        "Training supervision supplies event types, roles and polarity; none is an inference input.",
                        "Development and final syntax use separate complete sentence families with familiar lexical ingredients.",
                        "Lexical clause uses identity-aware embeddings; pointer uses generic mention embeddings and learned role scores bound to visible identities.",
                        "Correct programs and all-query world accuracy guard against accidentally correct single answers.",
                        "The old checkpoint baseline uses its historical training data; retrained clause controls isolate gains from broader training."]}
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    section = summary.setdefault(args.protocol, {"runs": []})
    section.setdefault("runs", []).append(result)
    if polarity:
        result["paired_training"] = paired.report
        result["final_data_fingerprints"] = {name: split_fingerprint(examples) for name, examples in frozen_tests.items()}
    if preservation:
        result["data"]["routine_regression"] = {"count": len(routine_regression.examples),
            "sha256": split_fingerprint(routine_regression.examples), "scope": "Previously observed development failures; never training inputs"}
    persist(summary)
    bundles = []
    for seed in seeds:
        torch.manual_seed(seed)
        if str(args.device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        parser = interpretation_parser(args.parser_type, args.width, args.local_layers, vocabulary)
        parser.to(args.device)
        training = train_component(parser, "parser", prepared_train, prepared_dev["iid"], args, seed,
                                   selection_sets={name: data for name, data in prepared_dev.items() if name != "iid"},
                                   paired=paired, consistency_weight=args.pair_consistency if polarity else 0,
                                   source_weight=args.source_weight if attachment else 0,
                                   freeze_after=args.freeze_after if preservation else 0,
                                   capture_step=args.freeze_after if preservation else 0)
        core = copy.deepcopy(reference["oracle"]).requires_grad_(False)
        models = {"parser": parser.cpu(), "oracle": core}
        trial = {"seed": seed, "training": {"parser": training},
                 "inference_parameters": {"parser": parameter_count(parser), "oracle": parameter_count(core),
                                           "parsed": parameter_count(parser) + parameter_count(core),
                                           "executor": parameter_count(parser)}, "development": {}}
        for name, data in prepared_dev.items():
            trial["development"][name] = evaluate_paths(models, data, args.device)
            if name in ("wording", "combined"):
                trial["development"][name]["parser"]["error_examples"] = parser_errors(data)
            if name in ("wording", "combined"):
                groups = all_query_examples(development[name], seed=110201 if binding else 92201 if scaling else 82201,
                                            stage="development", shift=name, **expansion_options)
                trial["development"][name]["all_queries"] = evaluate_query_worlds(models, groups, args.device)
                pairs = interpretation_contrast_pairs(development[name], seed=110202 if binding else 92202 if scaling else 82202,
                                                      stage="development", shift=name, **expansion_options)
                trial["development"][name]["contrasts"] = evaluate_query_worlds(models, [pair[:2] for pair in pairs], args.device,
                    group_kinds=[pair[2] for pair in pairs] if polarity else None)
                if polarity:
                    trial["development"][name]["direct_polarity"] = evaluate_polarity_groups(
                        models, *direct_probes[name], args.device)
            print(json.dumps({"seed": seed, "development_split": name,
                              "program": trial["development"][name]["parser"]["program_accuracy"],
                              "parsed": trial["development"][name]["parsed"]["macro_exact_accuracy"],
                              "executor": trial["development"][name]["executor"]["macro_exact_accuracy"]}), flush=True)
        trial["validation"] = trial["development"]["iid"]
        if polarity:
            trial["observed_polarity"] = evaluate_polarity_groups(models, observed_probes,
                {"known_clauses": len(observed_probes), "zero_amounts_included": True}, args.device)
        if attachment:
            trial["source_diagnostics"] = {name: evaluate_source_diagnostic(models, prepared_dev[name],
                "development", args.device) for name in ("wording", "combined")}
        if preservation:
            trial["routine_regression"] = evaluate_paths(models, routine_regression, args.device)
            trial["routine_regression"]["parser"]["error_examples"] = parser_errors(routine_regression)
        if scaling or binding:
            from semantic_tasks import scaling_role_distance_pairs
            diagnostic = [example for pair in scaling_role_distance_pairs() for example in pair[:2]]
            diagnostic_data = prepare(diagnostic, args.device, representation, vocabulary)
            trial["role_distance_diagnostic"] = evaluate_paths(models, diagnostic_data, args.device)
            trial["role_distance_diagnostic"]["parser"]["error_examples"] = parser_errors(diagnostic_data)
        models["controls"] = {}
        trial["controls"] = {}
        for control_type, control_parser_type, control_width, training_view in interpretation_controls(args):
            control_train = unbalanced_train if training_view == "unbalanced" else prepared_train
            torch.manual_seed(seed)
            control_parser = interpretation_parser(control_parser_type, control_width,
                                                   args.local_layers, vocabulary).to(args.device)
            control_training = train_component(control_parser, "parser", control_train, prepared_dev["iid"], args, seed,
                                               selection_sets={name: data for name, data in prepared_dev.items() if name != "iid"},
                                               paired=paired, consistency_weight=0,
                                               capture_step=args.freeze_after if preservation else 0)
            if preservation and training["prefix"] != control_training["prefix"]:
                raise AssertionError("Candidate and full-model control differ before the freeze boundary")
            control_models = {"parser": control_parser.cpu(),
                              "oracle": copy.deepcopy(reference["oracle"]).requires_grad_(False)}
            control_development = {name: evaluate_paths(control_models, data, args.device)
                                   for name, data in prepared_dev.items()}
            if polarity:
                for name in ("wording", "combined"):
                    groups = all_query_examples(development[name], seed=110201, stage="development", shift=name,
                                                protocol="binding")
                    pairs = interpretation_contrast_pairs(development[name], seed=110202, stage="development", shift=name,
                                                          protocol="binding")
                    control_development[name]["all_queries"] = evaluate_query_worlds(control_models, groups, args.device)
                    control_development[name]["contrasts"] = evaluate_query_worlds(control_models, [pair[:2] for pair in pairs], args.device,
                        group_kinds=[pair[2] for pair in pairs])
                    control_development[name]["direct_polarity"] = evaluate_polarity_groups(
                        control_models, *direct_probes[name], args.device)
                    control_development[name]["parser"]["error_examples"] = parser_errors(prepared_dev[name])
            trial["controls"][control_type] = {"training": control_training,
                                              "parser_type": control_parser_type, "width": control_width,
                                              "training_view": training_view,
                                              "train_sha256": split_fingerprint(control_train.examples),
                                              "development": control_development,
                                              "parser_parameters": parameter_count(control_parser),
                                              "parser_sha256": tensor_state_hash(control_parser)}
            models["controls"][control_type] = control_models
            if attachment:
                trial["controls"][control_type]["source_diagnostics"] = {
                    name: evaluate_source_diagnostic(control_models, prepared_dev[name], "development", args.device)
                    for name in ("wording", "combined")}
            if polarity:
                control_trial = {"seed": seed, "development": control_development,
                    "observed_polarity": evaluate_polarity_groups(control_models, observed_probes,
                        {"known_clauses": len(observed_probes), "zero_amounts_included": True}, args.device),
                    "role_distance_diagnostic": evaluate_paths(control_models, diagnostic_data, args.device)}
                gates = interpretation_gates([control_trial], "development", True, strict=True)
                for extra in (polarity_gates([control_trial]), scaling_diagnostic_gate([control_trial])):
                    gates["failures"].extend(extra["failures"])
                    gates["passed"] &= extra["passed"]
                if preservation:
                    control_trial["routine_regression"] = evaluate_paths(control_models, routine_regression, args.device)
                    control_trial["routine_regression"]["parser"]["error_examples"] = parser_errors(routine_regression)
                    extra = preservation_gates([control_trial])
                    gates["failures"].extend(extra["failures"])
                    gates["passed"] &= extra["passed"]
                    trial["controls"][control_type]["routine_regression"] = control_trial["routine_regression"]
                trial["controls"][control_type].update(observed_polarity=control_trial["observed_polarity"],
                    role_distance_diagnostic=control_trial["role_distance_diagnostic"], development_gates=gates)
        if tensor_state_hash(core) != core_hash:
            raise AssertionError("Frozen computation weights changed")
        bundles.append(models)
        result["trials"].append(trial)
        persist(summary)
    result["development_aggregate"] = {name: aggregate(result["trials"], "development", name)
                                        for name in development}
    result["development_gates"] = interpretation_gates(result["trials"], "development", args.retain_stability, strict=binding)
    if scaling or binding:
        result["role_distance_gate"] = scaling_diagnostic_gate(result["trials"])
        result["development_gates"]["failures"].extend(result["role_distance_gate"]["failures"])
        result["development_gates"]["passed"] &= result["role_distance_gate"]["passed"]
    if polarity:
        result["direct_polarity_gates"] = polarity_gates(result["trials"])
        result["development_gates"]["failures"].extend(result["direct_polarity_gates"]["failures"])
        result["development_gates"]["passed"] &= result["direct_polarity_gates"]["passed"]
    if preservation:
        result["routine_regression_gates"] = preservation_gates(result["trials"])
        result["development_gates"]["failures"].extend(result["routine_regression_gates"]["failures"])
        result["development_gates"]["passed"] &= result["routine_regression_gates"]["passed"]
    ranking = sorted(range(len(seeds)), key=lambda i: (
        min(task["program_accuracy"] for entry in result["trials"][i]["development"].values()
            for task in entry["parser"]["per_task"].values()),
        statistics.mean(entry["parser"]["program_accuracy"] for entry in result["trials"][i]["development"].values())))
    representative = ranking[len(ranking) // 2]
    selection = {"representative_seed": seeds[representative], "best_neural_mode": "parsed",
                 "best_overall_mode": "executor", "criterion": "Parser checkpoints maximize worst then mean development full-program accuracy; median seed by that criterion; exact execution default; no final-test selection"}
    if preservation:
        selection["criterion"] = "Checkpoints prefer fixed primary 99/95 program eligibility, then worst/mean programs and loss; median seed by worst/mean development programs; every expanded and observed gate still required; no final-test selection"
    result["selection"] = selection
    result["selected_parser_sha256"] = [tensor_state_hash(models["parser"]) for models in bundles]
    result["status"] = "selection_frozen"
    persist(summary)
    if binding and provenance() != result["source_sha256"]:
        result["status"] = "source_changed_before_test"
        persist(summary)
        raise RuntimeError("Binding source changed during training; final test remains unopened")
    metadata = {"config": {"width": args.width, "local_layers": args.local_layers, "layers": 2,
                            "architecture": "stable", "representation": representation,
                            "parser_type": args.parser_type, "vocabulary": vocabulary},
                "protocol": args.protocol, "selected_mode": "executor", "selection": selection,
                "frozen_core_sha256": core_hash, "source_sha256": provenance(),
                "scope": "Controlled-language research; fixed four-entity schema; no continual learning"}
    if binding:
        metadata["release_status"] = "experimental binding candidate"
        metadata["evaluation_contract"] = {"routine_retention": .99, "novel_and_combined": .95,
                                           "per_task_per_seed": True, "expanded_programs_and_contrasts": True}
    if polarity:
        metadata["evaluation_contract"].update(direct_polarity=.95, observed_counterexamples=1.0)
        metadata["training_recipe"] = {"paired_views_sha256": paired.report["views_sha256"],
            "parser_updates": args.parser_steps, "rendered_batch_size": args.batch_size,
            "entity_feature_consistency_weight": args.pair_consistency,
            "objective": "Paired supervised event loss plus aligned entity-feature cosine distance"}
        if attachment:
            metadata["training_recipe"].update(source_supervision_weight=args.source_weight,
                objective="Paired event loss plus supervised source trigger and semantic argument links")
        if preservation:
            metadata["training_recipe"].update(freeze_after=args.freeze_after,
                trainable_after_freeze=[*ROLE_PARAMETER_PREFIXES, "link_distance_bias"],
                objective="Paired event CE; preserve encoder and kind/activity heads after the fixed prefix")
            metadata["evaluation_contract"]["observed_routine_errors"] = 1.0
    if (scaling or binding) and args.phase == "confirm":
        selected = bundles[representative]
        save_checkpoint(CANDIDATE, selected["parser"], None, selected["oracle"], None,
                        {**metadata, "promotion_status": "experimental; selected before final evaluation"})
        result["candidate_checkpoint"] = {"path": str(CANDIDATE.relative_to(ROOT)),
                                           "sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                                           "parser_sha256": result["selected_parser_sha256"][representative]}
        persist(summary)
    if args.phase == "confirm" and not result["development_gates"]["passed"]:
        result["status"] = "development_gate_failed"
    elif args.phase == "confirm":
        result["test_accessed"] = True
        result["status"] = "evaluating_test"
        persist(summary)
        tests = {}
        test_seed = 111011 if binding else 93011 if scaling else 83011
        retention_seed = 111401 if binding else 93401 if scaling else 83401
        for i, shift in enumerate(INTERPRETATION_SHIFTS):
            count = min(args.test_size, args.wording_test_size) if shift == "wording" else args.test_size
            examples = make_examples(count, test_seed + i, shift, exclude=excluded, stage="final")
            tests[shift] = examples
            excluded.update(example.text for example in examples)
        if args.retain_stability:
            count = min(args.test_size, args.retention_test_size)
            tests["retention"] = retention_split(count, retention_seed, exclude=excluded,
                                                 candidate_count=max(9600 if scaling or binding else 4800, count * 8))
        result["data"]["tests"] = {name: {"count": len(examples), "seed": retention_seed if name == "retention" else test_seed + i,
                                          "sha256": split_fingerprint(examples)}
                                    for i, (name, examples) in enumerate(tests.items())}
        if polarity and any(split_fingerprint(examples) != result["final_data_fingerprints"][name]
                            for name, examples in tests.items()):
            raise AssertionError("Final examples differ from the frozen data-only isolation check")
        result["reference_test"] = {}
        result["model_test_evaluated"] = True
        for name, examples in tests.items():
            old_parser = reference["parser"]
            data = prepare(examples, args.device, old_parser.representation, getattr(old_parser, "vocabulary", None))
            result["reference_test"][name] = evaluate_paths(reference, data, args.device)
        for models, trial in zip(bundles, result["trials"]):
            trial["test"] = {}
            for name, examples in tests.items():
                data = prepare(examples, args.device, representation, vocabulary)
                trial["test"][name] = evaluate_paths(models, data, args.device)
                if name in ("wording", "combined"):
                    trial["test"][name]["parser"]["error_examples"] = parser_errors(data)
                    groups = all_query_examples(examples, seed=111201 if binding else 93201 if scaling else 83201,
                                                stage="final", shift=name, **expansion_options)
                    trial["test"][name]["all_queries"] = evaluate_query_worlds(models, groups, args.device)
                    pairs = interpretation_contrast_pairs(examples, seed=111202 if binding else 93202 if scaling else 83202,
                                                          stage="final", shift=name, **expansion_options)
                    trial["test"][name]["contrasts"] = evaluate_query_worlds(models, [pair[:2] for pair in pairs], args.device,
                        group_kinds=[pair[2] for pair in pairs] if polarity else None)
                    if polarity:
                        from semantic_tasks import direct_polarity_groups
                        groups, coverage = direct_polarity_groups(examples, stage="final")
                        trial["test"][name]["direct_polarity"] = evaluate_polarity_groups(models, groups, coverage, args.device)
                print(json.dumps({"seed": trial["seed"], "test_split": name,
                                  "program": trial["test"][name]["parser"]["program_accuracy"],
                                  "parsed": trial["test"][name]["parsed"]["macro_exact_accuracy"],
                                  "executor": trial["test"][name]["executor"]["macro_exact_accuracy"]}), flush=True)
                persist(summary)
            for control_type, control_models in models["controls"].items():
                control_result = trial["controls"][control_type]
                control_result["test"] = {}
                for name, examples in tests.items():
                    data = prepare(examples, args.device, representation, vocabulary)
                    control_result["test"][name] = evaluate_paths(control_models, data, args.device)
                if tensor_state_hash(control_models["parser"]) != control_result["parser_sha256"]:
                    raise AssertionError("Control parser weights changed during final evaluation")
                if tensor_state_hash(control_models["oracle"]) != core_hash:
                    raise AssertionError("Control computation weights changed during final evaluation")
            if getattr(models["parser"], "constrained_roles", False):
                # Same learned weights, independently copied; remove only the
                # schema decoder to measure its contribution after selection.
                ablated = {"parser": copy.deepcopy(models["parser"]), "oracle": models["oracle"]}
                ablated["parser"].constrained_roles = False
                trial["unconstrained_decoder_ablation"] = {
                    name: evaluate_paths(ablated, prepare(tests[name], args.device, representation, vocabulary), args.device)
                    for name in ("wording", "combined")}
            if tensor_state_hash(models["oracle"]) != core_hash:
                raise AssertionError("Frozen computation weights changed during evaluation")
        if [tensor_state_hash(models["parser"]) for models in bundles] != result["selected_parser_sha256"]:
            raise AssertionError("Selected parser weights changed during final evaluation")
        if binding and (provenance() != result["source_sha256"] or
                        hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != result["candidate_checkpoint"]["sha256"]):
            result["status"] = "provenance_changed_during_test"
            persist(summary)
            raise RuntimeError("Binding source or candidate artifact changed during final evaluation")
        result["test_aggregate"] = {name: aggregate(result["trials"], "test", name) for name in tests}
        result["capability_gates"] = interpretation_gates(result["trials"], require_retention=args.retain_stability, strict=binding)
        if polarity:
            extra = polarity_gates(result["trials"], "test")
            result["capability_gates"]["failures"].extend(extra["failures"])
            result["capability_gates"]["passed"] &= extra["passed"]
        if result["capability_gates"]["passed"]:
            selected = bundles[representative]
            if binding:
                metadata["release_status"] = "semantic research V1; scoped validation passed"
                metadata["validation_seeds"] = seeds
                metadata["validation_source"] = f"results/semantic_v1.json:{args.protocol}"
            save_checkpoint(CHECKPOINT, selected["parser"], None, selected["oracle"], None, metadata)
            result["checkpoint_sha256"] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
            section["selection"] = selection
            result["promoted"] = True
        else:
            result["promoted"] = False
        result["status"] = "complete"
    else:
        result["status"] = "screen_complete"
    result["total_seconds"] = time.perf_counter() - started
    persist(summary)
    print(json.dumps({"phase": args.phase, "status": result["status"],
                      "development_gate_passed": result["development_gates"]["passed"],
                      "test_accessed": result["test_accessed"], "total_seconds": result["total_seconds"]}), flush=True)
    return result


# Reproducibility utilities, full-pipeline timing, and command-line routing.
def observed_stability_regression(args):
    """Replay already-observed historical sets; never use these as fresh evidence."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    models, metadata = load_checkpoint(CHECKPOINT)
    models = {name: models[name] for name in ("parser", "oracle")}
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    original = next(run for run in reversed(summary["stabilization"]["runs"])
                    if run["phase"] == "confirm" and run.get("test_accessed"))
    train = make_stability_split(original["data"]["train_count"], 72001)
    valid = make_stability_split(original["data"]["valid_count"], 72002, exclude=(e.text for e in train))
    excluded = {e.text for e in train + valid}
    for i, shift in enumerate(("length", "numbers", "composition")):
        examples = make_stability_split(200, 72101 + i, shift, exclude=excluded)
        excluded.update(e.text for e in examples)
    scores = {}
    for i, shift in enumerate(STABILITY_SHIFTS):
        expected = original["data"]["tests"][shift]
        examples = make_stability_split(expected["count"], 73011 + i, shift, exclude=excluded)
        excluded.update(e.text for e in examples)
        if split_fingerprint(examples) != expected["sha256"]:
            raise AssertionError("Historical regression does not reproduce the recorded split")
        parser = models["parser"]
        scores[shift] = evaluate_paths(models, prepare(examples, args.device, parser.representation,
                                                      getattr(parser, "vocabulary", None)), args.device)
    summary.setdefault("interpretation", {})["observed_stability_regression"] = {
        "scope": "Previously observed stability test sets, exact fingerprints reproduced; one current selected checkpoint versus historical three-seed summaries. Not fresh generalization evidence or a selection input.",
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "selection": metadata["selection"], "scores": scores,
        "historical_aggregate": original["test_aggregate"]}
    persist(summary)
    print(json.dumps({shift: {"program": metrics["parser"]["program_accuracy"],
                              "parsed": metrics["parsed"]["macro_exact_accuracy"],
                              "executor": metrics["executor"]["macro_exact_accuracy"]}
                      for shift, metrics in scores.items()}, indent=2))


def benchmark(args):
    """Actual batch-one text-to-answer timings, without gold rows or scoring."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    models, metadata = load_checkpoint(CHECKPOINT, args.device)
    protocol = metadata.get("protocol", "structural")
    if protocol in ("interpretation", "scaling", "binding", "polarity", "attachment", "preservation"):
        from semantic_tasks import make_interpretation_split, make_scaling_split, make_binding_split
        prompt_examples = (make_binding_split(24, 112002) if protocol in ("binding", "polarity", "attachment", "preservation") else
                           make_scaling_split(24, 94002) if protocol == "scaling" else
                           make_interpretation_split(24, 84002))
    else:
        prompt_examples = make_stability_split(24, 74002) if protocol == "stability" else make_split(8, 64002)
    prompts = [example.text for example in prompt_examples]
    measurements = {}
    for mode in (mode for mode in DEPLOYABLE_MODES if mode in (*available_modes(models), "executor")):
        elapsed, invalid, samples = [], 0, []
        for index, text in enumerate(prompts[:2] + prompts * 3):
            sync(args.device)
            started = time.perf_counter()
            try:
                answer = predict(models, text, mode, args.device)
            except ValueError:
                answer = "<invalid>"
            sync(args.device)
            seconds = time.perf_counter() - started
            if index >= 2:
                elapsed.append(seconds)
                invalid += answer == "<invalid>"
                if len(samples) < 2:
                    samples.append({"text": text, "answer": answer})
        measurements[mode] = {"requests": len(elapsed), "mean_ms": statistics.mean(elapsed) * 1000,
                              "median_ms": statistics.median(elapsed) * 1000,
                              "invalid_programs": invalid, "samples": samples}
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    result = {"checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "selected_mode": metadata["selected_mode"], "measurements": measurements,
              "device": args.device, "threads": torch.get_num_threads(),
              "source_sha256": provenance(),
              "unique_prompts": len(prompts), "repetitions": 3,
              "scope": "Warm batch-one text normalization, tensor preparation, actual neural/executor inference, and answer decoding. Excludes model loading/startup; no gold inputs, losses, or accuracy scoring. Fixed IID prompts, three repetitions per mode."}
    summary[{"stability": "stabilization"}.get(protocol, protocol)]["pipeline_benchmark"] = result
    persist(summary)
    print(json.dumps({mode: {key: value for key, value in row.items() if key != "samples"}
                      for mode, row in measurements.items()}, indent=2))


def main():
    """Validate command-line settings and dispatch exactly one protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--protocol", choices=("structural", "stability", "interpretation", "scaling", "binding", "polarity", "attachment", "preservation"), default="structural")
    parser.add_argument("--pair-consistency", type=float, default=.1,
                        help="Polarity-only feature consistency; zero selects paired ordinary supervision")
    parser.add_argument("--freeze-after", type=int, default=0,
                        help="Preservation-only boundary after which encoder and non-role heads stop updating")
    parser.add_argument("--source-weight", type=float, default=.1,
                        help="Attachment-only trigger and argument-link supervision weight")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Audit the reproduced polarity baseline without training or final model evaluation")
    parser.add_argument("--parser-type", choices=("legacy", "lexical_clause", "lexical_pointer", "lexical_relative", "lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment"), default="legacy")
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--comparison-parsers", nargs="*", choices=("lexical_clause", "lexical_pointer", "lexical_relative",
                                                                     "lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment"), default=[])
    parser.add_argument("--comparison-widths", nargs="*", type=int, default=[])
    parser.add_argument("--comparison-unbalanced", action="store_true",
                        help="Compare the same binding parser trained without quantity-placement variants")
    parser.add_argument("--diversity", choices=("narrow", "broad"), default="broad")
    parser.add_argument("--balance-quantities", action="store_true",
                        help="Binding-only matched quantity placement variation in eligible training clauses")
    parser.add_argument("--retain-stability", action="store_true",
                        help="Include original 3000 training examples and require fresh old-grammar retention checks")
    parser.add_argument("--development-size", type=int, default=200)
    parser.add_argument("--architecture", choices=("attention", "bound", "stable"), default="attention")
    parser.add_argument("--representation", choices=("bytes", "typed"), default="bytes")
    parser.add_argument("--fit-quantities", action="store_true")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--parser-steps", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--valid-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--wording-test-size", type=int, default=300,
                        help="Final wording has only162 distinct relation worlds; use at most324 balanced examples")
    parser.add_argument("--retention-test-size", type=int, default=160,
                        help="Old grammar has limited worlds left after strict train/development exclusion")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--regression-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.parser_steps, args.train_size, args.valid_size,
           args.test_size, args.batch_size, args.width, args.eval_every, args.local_layers,
           args.development_size, args.wording_test_size, args.retention_test_size) <= 0 or args.lr <= 0:
        parser.error("Counts and learning rate must be positive")
    if args.width % 4 or min(args.train_size, args.valid_size, args.test_size, args.development_size,
                            args.wording_test_size, args.retention_test_size) < 4:
        parser.error("Width must be divisible by four and splits need at least four examples")
    if args.fit_quantities and args.architecture != "stable":
        parser.error("Quantity fitting requires the stable additive architecture")
    if args.protocol == "interpretation" and min(args.wording_test_size, args.test_size) > 324:
        parser.error("Final wording supports at most324 balanced examples under distinct-world exclusion")
    if args.balance_quantities and args.protocol not in ("binding", "polarity", "attachment", "preservation"):
        parser.error("Quantity placement balancing belongs to the binding protocol")
    if args.comparison_unbalanced and (args.protocol != "binding" or not args.balance_quantities):
        parser.error("Unbalanced controls require --protocol binding --balance-quantities")
    if args.benchmark_only and args.regression_only:
        parser.error("Choose either benchmarking or observed regression")
    if not math.isfinite(args.pair_consistency) or args.pair_consistency < 0:
        parser.error("Pair consistency must be finite and nonnegative")
    if args.freeze_after < 0 or (args.freeze_after and args.protocol != "preservation"):
        parser.error("Freeze boundary belongs to the preservation protocol")
    if not math.isfinite(args.source_weight) or args.source_weight < 0:
        parser.error("Source weight must be finite and nonnegative")
    if args.diagnose_only and (args.protocol != "polarity" or args.benchmark_only or args.regression_only):
        parser.error("Polarity diagnosis is a separate action")
    if args.diagnose_only:
        polarity_diagnosis(args)
    elif args.regression_only:
        observed_stability_regression(args)
    elif args.benchmark_only:
        benchmark(args)
    elif args.protocol in ("interpretation", "scaling", "binding", "polarity", "attachment", "preservation"):
        run_interpretation(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
