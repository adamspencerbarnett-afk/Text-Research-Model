"""Controlled-text semantic parsing and reasoning models.

The main data flow is text -> SurfaceBatch -> parser -> semantic rows. Each
semantic row stores [event kind, entity A, entity B, quantity, active] and
can be passed to a learned reasoner or the separately labeled exact executor.

All learned paths share surface-only sentence, name, and numeric-literal
handling. Semantic rows are parser training labels, never text-inference inputs.
This module does not implement online or continual learning.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from semantic_tasks import surface_sentences, surface_names, execute


# Surface representations and shared capacity limits

ROOT = Path(__file__).resolve().parents[1]
CLS, PAD = 256, 257
ENTITY_START, QUANTITY = 258, 262
MAX_BYTES, MAX_SENTENCES = 160, 24
FIELD_SIZES = {"kind": 6, "a": 5, "b": 5, "active": 2}
REPRESENTATIONS = ("bytes", "typed")
LITERAL_PATTERN = re.compile(r"(?<![A-Za-z0-9])-?\d+\b")
TYPED_PATTERN = re.compile(r"\be([0-3])\b|(?<![A-Za-z0-9])-?\d+\b")
LEXICAL_PATTERN = re.compile(r"\be[0-3]\b|(?<![A-Za-z0-9])-?\d+\b|[A-Za-z]+|\S")
LEXICAL_RESERVED = ("<pad>", "<cls>", "<unk>", "<quantity>",
                    "<entity:0>", "<entity:1>", "<entity:2>", "<entity:3>")


def lexical_tokens(sentence):
    """Preserve words/order; mark only visible identities and integer literals."""
    result = []
    for token in LEXICAL_PATTERN.findall(sentence):
        if re.fullmatch(r"e[0-3]", token):
            result.append(f"<entity:{token[1]}>")
        elif LITERAL_PATTERN.fullmatch(token):
            result.append("<quantity>")
        else:
            result.append(token.lower())
    return tuple(result)


def validate_lexicon(vocabulary):
    """Reject vocabularies that would change reserved or learned token IDs."""
    if (not isinstance(vocabulary, dict) or
            any(not isinstance(token, str) or type(index) is not int for token, index in vocabulary.items()) or
            any(vocabulary.get(token) != index for index, token in enumerate(LEXICAL_RESERVED)) or
            set(vocabulary.values()) != set(range(len(vocabulary)))):
        raise ValueError("Lexical vocabulary requires fixed reserved IDs and contiguous unique word IDs")


def fit_lexicon(training_texts):
    """Fit vocabulary once from training text; later unknown words use UNK=2."""
    texts = list(training_texts)
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("Vocabulary fitting requires nonempty training text strings")
    words = set()
    for text in texts:
        if not 1 <= len(surface_names(text)) <= 4:
            raise ValueError("Controlled text requires one to four visible proper names")
        for sentence in surface_sentences(text):
            words.update(lexical_tokens(sentence))
    vocabulary = {token: index for index, token in enumerate(LEXICAL_RESERVED)}
    vocabulary.update((token, index) for index, token in
                      enumerate(sorted(words - vocabulary.keys()), start=len(vocabulary)))
    return vocabulary


@dataclass
class SurfaceBatch:
    """Padded text tensors shared by parsers and direct-answer controls.

    ``ids`` and ``token_mask`` use ``[batch, sentence, token]``; ``values`` and
    ``sentence_mask`` use ``[batch, sentence]``. ``entity_ids`` aligns visible
    lexical entity mentions with token positions and uses -1 elsewhere.
    """
    ids: torch.Tensor
    token_mask: torch.Tensor
    sentence_mask: torch.Tensor
    values: torch.Tensor
    representation: str = "bytes"
    entity_ids: torch.Tensor | None = None

    def take(self, indices):
        """Select examples and trim unused sentence and token padding."""
        mask = self.sentence_mask[indices]
        sentences = int(mask.sum(-1).max())
        tokens = int(self.token_mask[indices, :sentences].sum(-1).max())
        return SurfaceBatch(self.ids[indices, :sentences, :tokens],
                            self.token_mask[indices, :sentences, :tokens],
                            mask[:, :sentences], self.values[indices, :sentences],
                            self.representation,
                            self.entity_ids[indices, :sentences, :tokens] if self.entity_ids is not None else None)


def prepare_texts(texts, device="cpu", representation="bytes", vocabulary=None):
    """Validate and encode raw texts as a padded, surface-only batch.

    Byte input preserves UTF-8 bytes, typed input replaces visible entities and
    number positions with type markers, and lexical input uses a fitted word
    vocabulary. Integer magnitude remains in the separate ``values`` tensor.
    """
    if representation not in (*REPRESENTATIONS, "lexical"):
        raise ValueError("Unsupported surface representation")
    if representation == "lexical":
        validate_lexicon(vocabulary)
    texts = list(texts)
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("prepare_texts requires nonempty text strings, not oracle examples")
    # Split first so capacity failures are reported before allocating tensors.
    sentences = []
    for text in texts:
        if not 1 <= len(surface_names(text)) <= 4:
            raise ValueError("Controlled text requires one to four visible proper names")
        rows = surface_sentences(text)
        if not rows or len(rows) > MAX_SENTENCES:
            raise ValueError("Sentence count outside the supported capacity")
        sentences.append(rows)
    # Encode each clause independently; parsers later emit one semantic row per
    # clause, including the final question.
    encoded_sentences = []
    for rows in sentences:
        encoded_rows = []
        for sentence in rows:
            if representation == "bytes":
                encoded = [CLS] + list(sentence.encode("utf-8"))
            elif representation == "lexical":
                encoded = [1] + [vocabulary.get(token, 2) for token in lexical_tokens(sentence)]
            else:
                # Entity references and quantities occupy distinct namespaces.
                # Only the scalar side channel retains the literal magnitude.
                encoded, cursor = [CLS], 0
                for match in TYPED_PATTERN.finditer(sentence):
                    encoded.extend(sentence[cursor:match.start()].encode("utf-8"))
                    encoded.append(ENTITY_START + int(match.group(1))
                                   if match.group(1) is not None else QUANTITY)
                    cursor = match.end()
                encoded.extend(sentence[cursor:].encode("utf-8"))
            encoded_rows.append(encoded)
        encoded_sentences.append(encoded_rows)
    width = max(len(row) for rows in encoded_sentences for row in rows)
    if width > MAX_BYTES + 1:
        raise ValueError("A sentence exceeds byte capacity; input is never silently truncated")
    count = max(map(len, sentences))
    ids = torch.full((len(texts), count, width), 0 if representation == "lexical" else PAD, dtype=torch.long)
    tokens = torch.zeros_like(ids, dtype=torch.bool)
    mask = torch.zeros((len(texts), count), dtype=torch.bool)
    values = torch.zeros((len(texts), count))
    entity_ids = torch.full_like(ids, -1) if representation == "lexical" else None
    # Every padded sentence retains a CLS key so attention is well-defined.
    ids[:, :, 0], tokens[:, :, 0] = 1 if representation == "lexical" else CLS, True
    for i, rows in enumerate(sentences):
        for j, sentence in enumerate(rows):
            encoded = encoded_sentences[i][j]
            ids[i, j, :len(encoded)] = torch.tensor(encoded)
            if entity_ids is not None:
                entity_ids[i, j, :len(encoded)] = torch.tensor([token - 4 if 4 <= token < 8 else -1
                                                               for token in encoded])
            tokens[i, j, :len(encoded)] = True
            mask[i, j] = True
            literals = LITERAL_PATTERN.findall(sentence)
            if len(literals) > 1:
                raise ValueError("This controlled benchmark supports one quantity per sentence")
            value = int(literals[0]) if literals else 0
            if representation in ("typed", "lexical") and abs(value) > 2 ** 24:
                raise ValueError("Typed integer literals must lie between -16777216 and 16777216 for exact float32 storage")
            values[i, j] = value
    return SurfaceBatch(*(tensor.to(device) for tensor in (ids, tokens, mask, values)),
                        representation=representation,
                        entity_ids=entity_ids.to(device) if entity_ids is not None else None)


def oracle_tensor(examples, device="cpu"):
    """Pad gold semantic rows for supervised training and diagnostics."""
    rows = torch.zeros((len(examples), max(len(e.rows) for e in examples), 5))
    rows[:, :, 1:3] = -1
    for i, example in enumerate(examples):
        rows[i, :len(example.rows)] = torch.tensor(example.rows)
    return rows.to(device)


def answer_targets(examples, device="cpu"):
    """Build task-type, numeric-answer, and boolean-answer supervision."""
    # Targets are never passed into forward()/prepare_texts().
    return torch.tensor([[int(e.task == "relations"),
                          float(e.answer) if e.task == "accounting" else 0,
                          int(e.answer == "yes")] for e in examples], device=device)


# Neural encoders and semantic parsers


class AttentionBlock(nn.Module):
    """Four-head self-attention followed by a residual feed-forward block."""
    def __init__(self, width):
        super().__init__()
        if width % 4:
            raise ValueError("Width must be divisible by four")
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, width * 3), nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, valid):
        n, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(n, length, 3, 4, width // 4).permute(2, 0, 3, 1, 4)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        x = x + self.proj(z.transpose(1, 2).reshape(n, length, width))
        return x + self.ff(self.norm2(x))


class RelativeAttentionBlock(AttentionBlock):
    """Learn signed token-distance preferences without absolute positions."""
    def __init__(self, width, max_distance=32):
        super().__init__(width)
        if max_distance <= 0:
            raise ValueError("Relative attention distance must be positive")
        self.max_distance = max_distance
        self.relative_bias = nn.Parameter(torch.zeros(4, 2 * max_distance + 1))

    def forward(self, x, valid):
        count, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(count, length, 3, 4, width // 4).permute(2, 0, 3, 1, 4)
        positions = torch.arange(length, device=x.device)
        distances = (positions[None, :] - positions[:, None]).clamp(-self.max_distance,
                                                                    self.max_distance) + self.max_distance
        bias = self.relative_bias[:, distances].unsqueeze(0).expand(count, -1, -1, -1)
        attention_mask = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x + self.proj(z.transpose(1, 2).reshape(count, length, width))
        return x + self.ff(self.norm2(x))


class SurfaceEncoder(nn.Module):
    """Reduce each byte or typed clause to one contextual feature vector."""
    def __init__(self, width=48, layers=2, representation="bytes"):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError("Unsupported surface representation")
        self.width = width
        self.representation = representation
        self.embedding = nn.Embedding(258 if representation == "bytes" else 263, width)
        self.position = nn.Embedding(MAX_BYTES + 1, width)
        self.blocks = nn.ModuleList(AttentionBlock(width) for _ in range(layers))
        self.literal = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, surface, include_values=True):
        if surface.representation != self.representation:
            raise ValueError("Surface representation does not match this encoder")
        n, sentences, length = surface.ids.shape
        x = self.embedding(surface.ids.reshape(-1, length))
        x = x + self.position(torch.arange(length, device=x.device))
        valid = surface.token_mask.reshape(-1, length)
        for block in self.blocks:
            x = block(x, valid)
        features = self.norm(x[:, 0]).reshape(n, sentences, -1)
        if include_values:
            features = features + self.literal(surface.values.unsqueeze(-1) / 16)
        return features * surface.sentence_mask.unsqueeze(-1)


class SemanticParser(nn.Module):
    """Legacy clause parser with one classifier per semantic-row field."""
    def __init__(self, width=48, layers=2, representation="bytes"):
        super().__init__()
        self.representation = representation
        self.encoder = SurfaceEncoder(width, layers, representation)
        if representation == "typed":
            # Keep these rows for state-compatible transfer into the raw
            # control, which does consume quantities. Parsing never uses them.
            self.encoder.literal.requires_grad_(False)
        self.heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in FIELD_SIZES.items()})

    def forward(self, surface):
        features = self.encoder(surface, include_values=self.representation != "typed")
        return {**{name: head(features) for name, head in self.heads.items()}, "features": features}


class LexicalEncoder(nn.Module):
    """Word-level contextual features without numerical-magnitude input."""
    def __init__(self, width, layers, vocabulary, generic_entities=False, relative_positions=False):
        super().__init__()
        validate_lexicon(vocabulary)
        self.width, self.representation = width, "lexical"
        self.generic_entities = generic_entities
        self.relative_positions = relative_positions
        self.embedding = nn.Embedding(len(vocabulary) - (3 if generic_entities else 0), width)
        self.position = None if relative_positions else nn.Embedding(MAX_BYTES + 1, width)
        block = RelativeAttentionBlock if relative_positions else AttentionBlock
        self.blocks = nn.ModuleList(block(width) for _ in range(layers))
        self.norm = nn.LayerNorm(width)

    def raw_embeddings(self, surface):
        """Embed lexical IDs, optionally sharing one marker across entities."""
        if (surface.representation != "lexical" or surface.entity_ids is None or
                surface.entity_ids.shape != surface.ids.shape):
            raise ValueError("Lexical encoding requires token-aligned visible entity mentions")
        ids = surface.ids
        if self.generic_entities:
            # Canonical IDs affect only the pointer's output binding. The
            # contextual network sees the same entity marker for every name.
            ids = torch.where(ids >= 8, ids - 3, ids)
            ids = ids.masked_fill((surface.ids >= 4) & (surface.ids < 8), 4)
        return self.embedding(ids)

    def forward(self, surface):
        """Return contextual token features shaped [batch, sentence, token, width]."""
        count, sentences, length = surface.ids.shape
        x = self.raw_embeddings(surface).reshape(-1, length, self.width)
        if self.position is not None:
            x = x + self.position(torch.arange(length, device=x.device))
        valid = surface.token_mask.reshape(-1, length)
        for block in self.blocks:
            x = block(x, valid)
        return (self.norm(x).reshape(count, sentences, length, -1) *
                surface.sentence_mask[:, :, None, None])


class LexicalParser(nn.Module):
    """Compare clause classification with learned contextual role pointers.

    Both variants receive the same words and visible entity candidates. Clause
    classification sees identity-specific embeddings; pointer classification
    uses a shared entity embedding and binds mention scores back to source IDs.
    No verb, role, polarity, or event type is supplied by preprocessing.
    """
    def __init__(self, width=48, layers=2, vocabulary=None, pooling="pointer", relative_positions=False,
                 constrained_roles=False, local_roles=False, joint_roles=False, evidence_roles=False,
                 attachment_roles=False):
        super().__init__()
        if pooling not in ("clause", "pointer"):
            raise ValueError("Lexical pooling must be clause or pointer")
        if relative_positions and pooling != "pointer":
            raise ValueError("Relative lexical attention is a pointer candidate")
        if constrained_roles and not (relative_positions and pooling == "pointer"):
            raise ValueError("Schema-constrained roles require the relative pointer candidate")
        if local_roles and not constrained_roles:
            raise ValueError("Local role features require the constrained relative pointer candidate")
        if joint_roles and not constrained_roles:
            raise ValueError("Joint role features require the constrained relative pointer candidate")
        if local_roles and joint_roles:
            raise ValueError("Local-only and joint role features are mutually exclusive")
        if evidence_roles and not constrained_roles:
            raise ValueError("Lexical role evidence requires the constrained relative pointer candidate")
        if evidence_roles and (local_roles or joint_roles):
            raise ValueError("Lexical role evidence cannot be combined with local or joint role heads")
        if attachment_roles and not constrained_roles:
            raise ValueError("Source attachment requires the constrained relative pointer candidate")
        if attachment_roles and (local_roles or joint_roles or evidence_roles):
            raise ValueError("Source attachment cannot be combined with other optional role heads")
        validate_lexicon(vocabulary)
        self.vocabulary = dict(vocabulary)
        self.representation, self.pooling = "lexical", pooling
        self.relative_positions = relative_positions
        self.constrained_roles = constrained_roles
        self.local_roles = local_roles
        self.joint_roles = joint_roles
        self.evidence_roles = evidence_roles
        self.attachment_roles = attachment_roles
        self.encoder = LexicalEncoder(width, layers, self.vocabulary, generic_entities=pooling == "pointer",
                                      relative_positions=relative_positions)
        self.heads = nn.ModuleDict({"kind": nn.Linear(width, 6), "active": nn.Linear(width, 2)})
        if pooling == "clause":
            self.role_heads = nn.ModuleDict({"a": nn.Linear(width, 5), "b": nn.Linear(width, 5)})
        else:
            if attachment_roles:
                self.trigger_head = nn.Linear(width, 1)
                self.link_queries = nn.Linear(width, 32)
                self.link_keys = nn.Linear(width, 16)
                distances = torch.arange(-32, 33, dtype=torch.float32)
                self.link_distance_bias = nn.Parameter(-distances.abs().repeat(2, 1) / 8)
            elif evidence_roles:
                self.role_queries = nn.Linear(width, 32)
                self.role_keys = nn.Linear(width, 16)
                self.role_values = nn.Linear(width, 2)
                # Begin with a symmetric locality prior, then learn each signed
                # distance independently. Neither role receives a direction rule.
                distances = torch.arange(-32, 33, dtype=torch.float32)
                self.role_distance_bias = nn.Parameter(-distances.abs().repeat(2, 1) / 8)
            else:
                self.mention_roles = (nn.Sequential(nn.Linear((7 if joint_roles else 6) * width, width),
                                                    nn.GELU(), nn.Linear(width, 2))
                                      if local_roles or joint_roles else nn.Linear(width, 2))
            self.none_roles = nn.Linear(width, 2)

    def local_role_features(self, surface):
        """Fixed nearby words plus an orderless lexical context, never labels."""
        raw = self.encoder.raw_embeddings(surface)
        raw = raw * surface.token_mask.unsqueeze(-1)
        word_mask = surface.token_mask & (surface.ids != 1)
        context = (raw * word_mask.unsqueeze(-1)).sum(2) / word_mask.sum(-1, keepdim=True).clamp_min(1)
        length = surface.ids.shape[-1]
        padded = F.pad(raw, (0, 0, 2, 2))
        neighbors = [padded[:, :, 2 + offset:2 + offset + length] for offset in (-2, -1, 0, 1, 2)]
        return torch.cat((*neighbors, context.unsqueeze(2).expand(-1, -1, length, -1)), dim=-1), context

    def evidence_role_scores(self, surface, tokens):
        """Bind shared lexical role values to mentions using contextual queries."""
        count, sentences, length, _ = tokens.shape
        raw = self.encoder.raw_embeddings(surface)
        queries = self.role_queries(tokens).reshape(-1, length, 2, 16).transpose(1, 2)
        keys = self.role_keys(raw).reshape(-1, length, 16)
        values = self.role_values(raw).reshape(-1, length, 2).transpose(1, 2)
        positions = torch.arange(length, device=tokens.device)
        distances = (positions[None, :] - positions[:, None]).clamp(-32, 32) + 32
        logits = torch.matmul(queries, keys[:, None].transpose(-1, -2)) / 4
        logits = logits + self.role_distance_bias[:, distances].unsqueeze(0)
        # The existing CLS sentinel stays valid even in a padded sentence.
        # Invalid keys never contribute, and no row softmax is entirely masked.
        valid = surface.token_mask.reshape(-1, length)
        weights = logits.masked_fill(~valid[:, None, None, :], float("-inf")).softmax(-1)
        scores = (weights * values[:, :, None, :]).sum(-1).transpose(1, 2)
        return scores.reshape(count, sentences, length, 2)

    def attachment_role_scores(self, surface, tokens):
        """Mix predicted semantic source frames into one event's argument scores."""
        count, sentences, length, _ = tokens.shape
        valid = surface.token_mask.reshape(-1, length)
        trigger_logits = self.trigger_head(tokens).reshape(-1, length)
        trigger_log_probs = trigger_logits.masked_fill(~valid, float("-inf")).log_softmax(-1)
        queries = self.link_queries(tokens).reshape(-1, length, 2, 16).transpose(1, 2)
        keys = self.link_keys(tokens).reshape(-1, length, 16)
        positions = torch.arange(length, device=tokens.device)
        distances = (positions[None, :] - positions[:, None]).clamp(-32, 32) + 32
        logits = torch.matmul(queries, keys[:, None].transpose(-1, -2)) / 4
        logits = logits + self.link_distance_bias[:, distances].unsqueeze(0)
        mentions = (surface.entity_ids.reshape(-1, length) >= 0) & valid
        # A sentence without entities uses only the existing CLS sentinel. It
        # supplies no actual entity candidate, but keeps every softmax defined.
        mention_keys = mentions.clone()
        mention_keys[:, 0] |= ~mentions.any(-1)
        link_log_probs = logits.masked_fill(~mention_keys[:, None, None, :],
                                           float("-inf")).log_softmax(-1)
        # Non-candidate columns need finite placeholders before logsumexp:
        # reducing an all-minus-infinity column gives undefined gradients.
        safe_links = link_log_probs.masked_fill(~mention_keys[:, None, None, :], -1e4)
        scores = torch.logsumexp(trigger_log_probs[:, None, :, None] + safe_links, dim=2)
        return (scores.transpose(1, 2).reshape(count, sentences, length, 2),
                trigger_log_probs.reshape(count, sentences, length),
                link_log_probs.reshape(count, sentences, 2, length, length))

    def forward(self, surface, *, return_token_features=False, return_source_links=False):
        """Predict event type, activity, and both entity roles for every clause."""
        if return_source_links and not self.attachment_roles:
            raise ValueError("Source-link outputs require the attachment parser")
        tokens = self.encoder(surface)
        features = tokens[:, :, 0]
        output = {name: head(features) for name, head in self.heads.items()}
        # Role outputs may select only entities visibly mentioned in the clause;
        # the fifth class represents a role that the event schema does not use.
        candidates = [(surface.entity_ids == entity) & surface.token_mask for entity in range(4)]
        available = torch.stack([mask.any(-1) for mask in candidates], dim=-1)
        available = torch.cat((available, torch.ones_like(available[:, :, :1])), dim=-1)
        if self.pooling == "clause":
            output.update({name: head(features).masked_fill(~available, -1e4)
                           for name, head in self.role_heads.items()})
        else:
            if self.attachment_roles:
                scores, trigger_log_probs, link_log_probs = self.attachment_role_scores(surface, tokens)
                none = self.none_roles(features)
                if return_source_links:
                    output.update(trigger_log_probs=trigger_log_probs, link_log_probs=link_log_probs)
            elif self.evidence_roles:
                scores = self.evidence_role_scores(surface, tokens)
                none = self.none_roles(features)
            elif self.joint_roles:
                local, _ = self.local_role_features(surface)
                # Nearby lexical roles and ordered long-range context remain
                # available to the same learned mention scorer.
                scores = self.mention_roles(torch.cat((local, tokens), dim=-1))
                none = self.none_roles(features)
            elif self.local_roles:
                local, context = self.local_role_features(surface)
                scores = self.mention_roles(local)
                none = self.none_roles(context)
            else:
                scores = self.mention_roles(tokens)
                none = self.none_roles(features)
            for role, name in enumerate(("a", "b")):
                entity_scores = [torch.logsumexp(scores[:, :, :, role].masked_fill(~mask, -1e4), dim=-1)
                                 for mask in candidates]
                logits = torch.stack((*entity_scores, none[:, :, role]), dim=-1)
                output[name] = logits.masked_fill(~available, -1e4)
        return {**output, "features": features,
                **({"constrained_roles": True} if self.constrained_roles else {}),
                **({"token_features": tokens} if return_token_features else {})}


# Parser training and semantic-row decoding


def parser_loss(output, rows, mask):
    """Sum cross-entropy losses for the four categorical parser fields."""
    targets = {"kind": rows[:, :, 0].long(), "a": rows[:, :, 1].long().remainder(5),
               "b": rows[:, :, 2].long().remainder(5), "active": rows[:, :, 4].long()}
    return sum(F.cross_entropy(output[name][mask], target[mask]) for name, target in targets.items())


def source_frame_loss(output, trigger_mask, mention_targets):
    """Training-only trigger CE plus conditional semantic-argument link CE.

    Labels use token positions including CLS. Every annotated predicate shares
    both event-argument targets; these are semantic links, not dependency heads.
    Unannotated clauses have an empty trigger mask and two -1 mention targets.
    The experiment applies its fixed auxiliary weight outside this helper.
    """
    if "trigger_log_probs" not in output or "link_log_probs" not in output:
        raise ValueError("Source-frame loss requires requested attachment outputs")
    triggers, links = output["trigger_log_probs"], output["link_log_probs"]
    if (triggers.ndim != 3 or trigger_mask.shape != triggers.shape or
            trigger_mask.dtype != torch.bool or mention_targets.dtype != torch.long or
            mention_targets.shape != (*triggers.shape[:2], 2) or
            links.shape != (*triggers.shape[:2], 2, triggers.shape[-1], triggers.shape[-1]) or
            trigger_mask.device != triggers.device or mention_targets.device != triggers.device):
        raise ValueError("Source-frame labels require aligned trigger masks and two mention positions")
    annotated = trigger_mask.any(-1)
    if (bool(trigger_mask[:, :, 0].any()) or
            bool((mention_targets[~annotated] != -1).any()) or
            bool(((mention_targets[annotated] <= 0) |
                  (mention_targets[annotated] >= triggers.shape[-1])).any()) or
            bool((mention_targets[annotated][:, 0] == mention_targets[annotated][:, 1]).any())):
        raise ValueError("Source-frame targets must name visible non-CLS tokens in annotated clauses")
    if not bool(annotated.any()):
        # Finite CLS entries preserve the graph even when this batch has no links.
        return triggers[:, :, 0].sum() * 0
    gold = trigger_mask[annotated]
    count = gold.sum(-1)
    selected_triggers = triggers[annotated]
    selected_links = links[annotated].gather(
        -1, mention_targets[annotated][:, :, None, None].expand(-1, -1, triggers.shape[-1], 1)
    ).squeeze(-1)
    link_mask = gold[:, None, :].expand_as(selected_links)
    if (not bool(torch.isfinite(selected_triggers[gold]).all()) or
            not bool(torch.isfinite(selected_links[link_mask]).all())):
        raise ValueError("Source-frame labels point to masked trigger or entity tokens")
    trigger_ce = -(selected_triggers.masked_fill(~gold, 0).sum(-1) / count).mean()
    link_ce = -(selected_links.masked_fill(~link_mask, 0).sum(-1) / count[:, None]).mean()
    return trigger_ce + link_ce


def parsed_rows(output, surface):
    """Decode parser logits into ``[kind, a, b, quantity, active]`` rows."""
    kind = output["kind"].argmax(-1)
    a, b = output["a"].argmax(-1), output["b"].argmax(-1)
    if output.get("constrained_roles", False):
        if surface.entity_ids is None:
            raise ValueError("Constrained roles require visible lexical entity mentions")
        available = torch.stack([((surface.entity_ids == entity) & surface.token_mask).any(-1)
                                 for entity in range(4)], dim=-1)
        unary = (kind == 1) | (kind == 4)
        binary = (kind == 2) | (kind == 3) | (kind == 5)
        unary_a = output["a"][:, :, :4].masked_fill(~available, float("-inf")).argmax(-1)
        unary_a = unary_a.masked_fill(~available.any(-1), 4)
        legal_pairs = available.unsqueeze(-1) & available.unsqueeze(-2)
        legal_pairs = legal_pairs & ~torch.eye(4, device=kind.device, dtype=torch.bool)
        pair_scores = output["a"][:, :, :4].unsqueeze(-1) + output["b"][:, :, :4].unsqueeze(-2)
        assignment = pair_scores.masked_fill(~legal_pairs, float("-inf")).flatten(-2).argmax(-1)
        legal = legal_pairs.flatten(-2).any(-1)
        pair_a = (assignment // 4).masked_fill(~legal, 4)
        pair_b = (assignment % 4).masked_fill(~legal, 4)
        # The declared event schema determines arity, not role direction. That
        # direction maximizes learned scores over distinct visible participants.
        a = torch.where(unary, unary_a, torch.where(binary, pair_a, 4))
        b = torch.where(binary, pair_b, 4)
    rows = torch.stack((kind, a.masked_fill(a == 4, -1),
                        b.masked_fill(b == 4, -1), surface.values,
                        output["active"].argmax(-1)), dim=-1).float()
    rows[:, :, 0] *= surface.sentence_mask
    return rows


# Learned reasoning cores


class AnswerCore(nn.Module):
    """Bidirectional event reasoning with four learned entity-memory slots.

    These slots are learned vectors, not a hard-coded accounting/graph solver.
    The exact executor is called only by separately labeled diagnostics.
    """
    def __init__(self, width=48, layers=2):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(1, 5, width) * .02)
        self.position = nn.Embedding(MAX_SENTENCES, width)
        self.blocks = nn.ModuleList(AttentionBlock(width) for _ in range(layers))
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, 4)

    def forward(self, features, mask):
        n, sentences, _ = features.shape
        features = features + self.position(torch.arange(sentences, device=features.device))
        x = torch.cat((self.memory.expand(n, -1, -1), features), dim=1)
        valid = torch.cat((torch.ones((n, 5), device=mask.device, dtype=torch.bool), mask), dim=1)
        for block in self.blocks:
            x = block(x, valid)
        return self.head(self.norm(x[:, 0]))


class Reasoner(nn.Module):
    """Embed predicted semantic rows before global learned answer reasoning."""
    def __init__(self, width=48, layers=2, source=False):
        super().__init__()
        self.kind, self.a, self.b, self.active = (nn.Embedding(size, 8) for size in (6, 5, 5, 2))
        self.mix = nn.Sequential(nn.Linear(33, width), nn.GELU())
        self.core = AnswerCore(width, layers)
        self.source = nn.Linear(width, width, bias=False) if source else None
        if self.source is not None:
            nn.init.zeros_(self.source.weight)

    def forward(self, rows, sentence_mask, features=None):
        z = torch.cat((self.kind(rows[:, :, 0].long()), self.a(rows[:, :, 1].long().remainder(5)),
                       self.b(rows[:, :, 2].long().remainder(5)), self.active(rows[:, :, 4].long()),
                       rows[:, :, 3:4] / 16), dim=-1)
        z = self.mix(z)
        if self.source is not None:
            if features is None:
                raise ValueError("Source residual requires features from the actual text")
            z = z + self.source(features)
        return self.core(z, sentence_mask)


class BoundReasoner(Reasoner):
    """Bind typed events to entity slots and learn updates from answer labels.

    The controlled schema supplies four identities, source/destination roles,
    directed BEFORE edges, and a final query sentence. Accounting amounts feed
    learned additive messages; no debit, credit, initialization, or negation
    coefficient is prescribed. Graph computation uses three shared learned
    message updates, not a graph traversal or an exact answer executor.
    """
    def __init__(self, width=48, layers=2, source=False):
        super().__init__(width, layers, source)
        self.binding_width = 8
        # Twelve event/polarity combinations, each with two semantic roles.
        self.quantity_messages = nn.Embedding(6 * 2 * 2, self.binding_width)
        nn.init.normal_(self.quantity_messages.weight, std=.1)
        self.quantity_readout = nn.Linear(self.binding_width, 1)
        self.graph_seed = nn.Parameter(torch.randn(self.binding_width) * .2)
        self.graph_self = nn.Linear(self.binding_width, self.binding_width, bias=False)
        self.graph_message = nn.Linear(self.binding_width, self.binding_width, bias=False)
        self.graph_readout = nn.Linear(self.binding_width, 1)
        self.residual_gates = nn.Parameter(torch.zeros(2))

    def forward(self, rows, sentence_mask, features=None):
        count, sentences, _ = rows.shape
        kind = rows[:, :, 0].long()
        active = rows[:, :, 4].long()
        source, destination = rows[:, :, 1].long(), rows[:, :, 2].long()
        source_valid = (source >= 0) & (source < 4)
        destination_valid = (destination >= 0) & (destination < 4)
        source = source.masked_fill(~source_valid, 4)
        destination = destination.masked_fill(~destination_valid, 4)
        # Invalid identities never alias a real slot, including in the global
        # residual path. Slot four is an always-zero dummy, not a fifth entity.
        safe_rows = rows.clone()
        safe_rows[:, :, 1] = source.masked_fill(source == 4, -1)
        safe_rows[:, :, 2] = destination.masked_fill(destination == 4, -1)
        base = super().forward(safe_rows, sentence_mask, features)
        positions = torch.arange(sentences, device=rows.device).expand(count, -1)
        last = positions.masked_fill(~sentence_mask, -1).amax(-1).clamp_min(0)
        batch = torch.arange(count, device=rows.device)
        query_a, query_b = source[batch, last], destination[batch, last]
        live_slots = rows.new_tensor([1, 1, 1, 1, 0]).view(1, 5, 1)

        event_index = (kind * 2 + active) * 2
        amounts = rows[:, :, 3:4] / 16 * sentence_mask.unsqueeze(-1)
        source_messages = self.quantity_messages(event_index) * amounts
        destination_messages = self.quantity_messages(event_index + 1) * amounts
        balances = rows.new_zeros((count, 5, self.binding_width))
        balances.scatter_add_(1, source.unsqueeze(-1).expand_as(source_messages), source_messages)
        balances.scatter_add_(1, destination.unsqueeze(-1).expand_as(destination_messages),
                              destination_messages)
        balances = balances * live_slots
        number = self.quantity_readout(balances[batch, query_a]).squeeze(-1)

        # Explicitly typed edge binding is an architectural prior. Repeated
        # assertions denote the same edge and do not increase its multiplicity.
        edges = ((kind == 3) & (active == 1) & source_valid & destination_valid & sentence_mask)
        adjacency = rows.new_zeros((count, 25))
        adjacency.scatter_add_(1, source * 5 + destination, edges.to(rows.dtype))
        adjacency = adjacency.reshape(count, 5, 5).clamp_max(1)
        state = F.one_hot(query_a, 5).to(rows.dtype).unsqueeze(-1) * self.graph_seed
        state = state * live_slots
        for _ in range(3):
            incoming = torch.bmm(adjacency.transpose(1, 2), state)
            state = F.relu(self.graph_self(state) + self.graph_message(incoming)) * live_slots
        relation = self.graph_readout(state[batch, query_b]).squeeze(-1)
        gates = self.residual_gates.tanh()
        return torch.cat((base[:, :2],
                          (number + gates[0] * base[:, 2]).unsqueeze(-1),
                          (relation + gates[1] * base[:, 3]).unsqueeze(-1)), dim=-1)


class StableReasoner(nn.Module):
    """Learn bound updates while preserving the final query's operation type.

    This keeps the previous eight-dimensional additive and graph branches, with
    three shared graph updates. It removes the global attention, type classifier,
    and unrestricted corrections. Width/layers remain accepted for the common
    factory interface; they do not alter this fixed 346-parameter core.
    """
    def __init__(self, width=48, layers=2, source=False):
        super().__init__()
        if source:
            raise ValueError("StableReasoner has no source-residual branch")
        self.binding_width = 8
        self.quantity_messages = nn.Embedding(24, self.binding_width)
        nn.init.normal_(self.quantity_messages.weight, std=.1)
        self.quantity_readout = nn.Linear(self.binding_width, 1)
        self.graph_seed = nn.Parameter(torch.randn(self.binding_width) * .2)
        self.graph_self = nn.Linear(self.binding_width, self.binding_width, bias=False)
        self.graph_message = nn.Linear(self.binding_width, self.binding_width, bias=False)
        self.graph_readout = nn.Linear(self.binding_width, 1)

    @staticmethod
    def valid_queries(rows, sentence_mask):
        """Mark batches containing exactly one supported active final query."""
        positions = torch.arange(rows.shape[1], device=rows.device).expand(rows.shape[0], -1)
        last = positions.masked_fill(~sentence_mask, -1).amax(-1).clamp_min(0)
        final = rows[torch.arange(rows.shape[0], device=rows.device), last]
        kind, source, destination = final[:, 0].long(), final[:, 1].long(), final[:, 2].long()
        queries = ((rows[:, :, 0] == 4) | (rows[:, :, 0] == 5)) & sentence_mask
        return (sentence_mask.any(-1) & (queries.sum(-1) == 1) &
                ((kind == 4) | (kind == 5)) & (source >= 0) & (source < 4) &
                ((kind == 4) | ((destination >= 0) & (destination < 4) & (destination != source))) &
                (final[:, 4] == 1))

    @torch.no_grad()
    def fit_quantities(self, rows, sentence_mask, targets):
        """Fit additive coefficients from explicit training answers only.

        This optional CPU solver never runs during inference. Schema roles
        determine feature placement; supervision determines every coefficient
        and the intercept. Unobserved roles receive the minimum-norm solution,
        so reported rank matters when assessing coverage of the training data.
        """
        if (rows.ndim != 3 or rows.shape[-1] != 5 or
                sentence_mask.shape != rows.shape[:2] or sentence_mask.dtype != torch.bool or
                targets.ndim != 2 or targets.shape != (rows.shape[0], 3)):
            raise ValueError("Quantity fitting requires rows [N,S,5], boolean mask [N,S], and targets [N,3]")
        if rows.shape[1] == 0 or not torch.isfinite(rows).all() or not torch.isfinite(targets).all():
            raise ValueError("Quantity fitting requires finite training rows and targets")
        rows = rows.detach().to(device="cpu", dtype=torch.float64)
        sentence_mask = sentence_mask.detach().cpu()
        targets = targets.detach().to(device="cpu", dtype=torch.float64)
        accounting = targets[:, 0] == 0
        if not accounting.any():
            raise ValueError("Quantity fitting requires accounting training examples")
        rows, sentence_mask, targets = rows[accounting], sentence_mask[accounting], targets[accounting]
        count, sentences, _ = rows.shape
        last = torch.arange(sentences).expand(count, -1).masked_fill(~sentence_mask, -1).amax(-1).clamp_min(0)
        batch = torch.arange(count)
        if not self.valid_queries(rows, sentence_mask).all() or not (rows[batch, last, 0] == 4).all():
            raise ValueError("Accounting training rows require one supported final balance query")
        live_kind, live_active = rows[:, :, 0][sentence_mask], rows[:, :, 4][sentence_mask]
        if (not ((live_kind >= 0) & (live_kind <= 5) & (live_kind == live_kind.round())).all() or
                not ((live_active == 0) | (live_active == 1)).all()):
            raise ValueError("Quantity fitting requires supported event types and activity flags")
        kind = rows[:, :, 0].long().masked_fill(~sentence_mask, 0)
        active = rows[:, :, 4].long().masked_fill(~sentence_mask, 0)
        event_index = (kind * 2 + active) * 2
        query_entity = rows[batch, last, 1].unsqueeze(-1)
        amounts = rows[:, :, 3] / 16 * sentence_mask
        design = torch.zeros((count, 25), dtype=torch.float64)
        for role, field in enumerate((1, 2)):
            contributions = amounts * (rows[:, :, field] == query_entity)
            design[:, :24].scatter_add_(1, event_index + role, contributions)
        design[:, 24] = 1
        expected = targets[:, 1] / 16
        fitted = torch.linalg.lstsq(design, expected, driver="gelsd")
        coefficients = fitted.solution
        self.quantity_messages.weight.zero_()
        self.quantity_messages.weight[:, 0].copy_(coefficients[:24])
        self.quantity_readout.weight.zero_()
        self.quantity_readout.weight[0, 0] = 1
        self.quantity_readout.bias.copy_(coefficients[24:].reshape_as(self.quantity_readout.bias))
        deployed = torch.cat((self.quantity_messages.weight[:, 0].detach().cpu().double(),
                              self.quantity_readout.bias.detach().cpu().double()))
        return {"method": "cpu_float64_minimum_norm_least_squares",
                "examples": count, "design_columns": 25,
                "rank": int(fitted.rank),
                "training_mse_normalized": float((design @ coefficients - expected).square().mean()),
                "stored_coefficient_mse_normalized": float((design @ deployed - expected).square().mean()),
                "quantity_coefficients": deployed[:24].tolist(), "intercept": float(deployed[24])}

    def forward(self, rows, sentence_mask, features=None):
        count, sentences, _ = rows.shape
        # Masked padding and missing identities cannot alias a real entity.
        kind = rows[:, :, 0].long().masked_fill(~sentence_mask, 0).clamp(0, 5)
        active = rows[:, :, 4].long().masked_fill(~sentence_mask, 0).clamp(0, 1)
        source, destination = rows[:, :, 1].long(), rows[:, :, 2].long()
        source_valid = (source >= 0) & (source < 4)
        destination_valid = (destination >= 0) & (destination < 4)
        source = source.masked_fill(~source_valid, 4)
        destination = destination.masked_fill(~destination_valid, 4)
        positions = torch.arange(sentences, device=rows.device).expand(count, -1)
        last = positions.masked_fill(~sentence_mask, -1).amax(-1).clamp_min(0)
        batch = torch.arange(count, device=rows.device)
        query_a, query_b = source[batch, last], destination[batch, last]
        live_slots = rows.new_tensor([1, 1, 1, 1, 0]).view(1, 5, 1)

        event_index = (kind * 2 + active) * 2
        amounts = rows[:, :, 3:4] / 16 * sentence_mask.unsqueeze(-1)
        source_messages = self.quantity_messages(event_index) * amounts
        destination_messages = self.quantity_messages(event_index + 1) * amounts
        balances = rows.new_zeros((count, 5, self.binding_width))
        balances.scatter_add_(1, source.unsqueeze(-1).expand_as(source_messages), source_messages)
        balances.scatter_add_(1, destination.unsqueeze(-1).expand_as(destination_messages),
                              destination_messages)
        balances = balances * live_slots
        number = self.quantity_readout(balances[batch, query_a]).squeeze(-1)

        edges = ((kind == 3) & (active == 1) & source_valid & destination_valid & sentence_mask)
        adjacency = rows.new_zeros((count, 25))
        adjacency.scatter_add_(1, source * 5 + destination, edges.to(rows.dtype))
        adjacency = adjacency.reshape(count, 5, 5).clamp_max(1)
        state = F.one_hot(query_a, 5).to(rows.dtype).unsqueeze(-1) * self.graph_seed
        state = state * live_slots
        for _ in range(3):
            incoming = torch.bmm(adjacency.transpose(1, 2), state)
            state = F.relu(self.graph_self(state) + self.graph_message(incoming)) * live_slots
        relation = self.graph_readout(state[batch, query_b]).squeeze(-1)

        query_kind = kind[batch, last]
        types = torch.stack((query_kind == 4, query_kind == 5), dim=-1).to(rows.dtype) * 20 - 10
        # Invalid parses remain finite during evaluation. Callers mark them with
        # valid_queries(), and text inference rejects them before decoding.
        return torch.cat((types, number.unsqueeze(-1), relation.unsqueeze(-1)), dim=-1)


class RawReasoner(nn.Module):
    """Direct text-to-answer control that bypasses semantic-row parsing."""
    def __init__(self, width=48, local_layers=2, layers=2, representation="bytes"):
        super().__init__()
        self.representation = representation
        self.encoder = SurfaceEncoder(width, local_layers, representation)
        self.core = AnswerCore(width, layers)

    def forward(self, surface):
        return self.core(self.encoder(surface), surface.sentence_mask)


def answer_loss(predictions, targets):
    """Train task selection plus the relevant numeric or boolean answer head."""
    typ = targets[:, 0].long()
    loss = F.cross_entropy(predictions[:, :2], typ)
    accounting, relations = typ == 0, typ == 1
    if accounting.any():
        # Keep integer-accuracy gradients material alongside the boolean task.
        loss = loss + 16 * F.mse_loss(predictions[accounting, 2], targets[accounting, 1] / 16)
    if relations.any():
        loss = loss + F.binary_cross_entropy_with_logits(predictions[relations, 3], targets[relations, 2])
    return loss


def decode_answers(predictions, valid_queries=None):
    """Convert answer logits to public strings, optionally rejecting bad parses."""
    data = predictions.detach().cpu()
    answers = [str(round(float(row[2]) * 16)) if int(row[:2].argmax()) == 0 else
               ("yes" if row[3] >= 0 else "no") for row in data]
    if valid_queries is not None:
        valid = valid_queries.detach().cpu().tolist()
        answers = [answer if allowed else "<invalid>" for answer, allowed in zip(answers, valid)]
    return answers


# Checkpoint persistence and public inference


def save_checkpoint(path, parser, raw, oracle, hybrid, metadata, raw_aux=None):
    """Save component states and the architecture data needed to rebuild them."""
    config = metadata["config"]
    if isinstance(parser, LexicalParser):
        config = {**config, "parser_type": ("lexical_attachment" if parser.attachment_roles else
                                             "lexical_evidence" if parser.evidence_roles else
                                             "lexical_joint" if parser.joint_roles else
                                             "lexical_local" if parser.local_roles else
                                             "lexical_constrained" if parser.constrained_roles else
                                             "lexical_relative" if parser.relative_positions else
                                             f"lexical_{parser.pooling}"),
                  "representation": "lexical", "vocabulary": dict(parser.vocabulary)}
        metadata = {**metadata, "config": config}
    models = {"parser": parser, "raw": raw, "oracle": oracle, "hybrid": hybrid}
    if raw_aux is not None:
        models["raw_aux"] = raw_aux
    state = {"format_version": 1, "config": config, "metadata": metadata,
             "states": {name: {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        for name, model in models.items() if model is not None}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild saved components on ``device`` and return them with metadata."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format_version"] != 1:
        raise ValueError("Unsupported structural checkpoint")
    c = state["config"]
    architecture = c.get("architecture", "attention")
    representation = c.get("representation", "bytes")
    if architecture not in ("attention", "bound", "stable"):
        raise ValueError("Unsupported reasoning architecture")
    parser_type = c.get("parser_type", "legacy")
    if parser_type not in ("legacy", "lexical_clause", "lexical_pointer", "lexical_relative",
                            "lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence",
                            "lexical_attachment"):
        raise ValueError("Unsupported semantic parser architecture")
    def parser_factory():
        """Recreate the parser variant recorded in checkpoint configuration."""
        if parser_type == "legacy":
            return SemanticParser(c["width"], c["local_layers"], representation)
        return LexicalParser(c["width"], c["local_layers"], c["vocabulary"],
                             pooling="pointer" if parser_type in ("lexical_relative", "lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment") else
                             parser_type.removeprefix("lexical_"),
                             relative_positions=parser_type in ("lexical_relative", "lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment"),
                             constrained_roles=parser_type in ("lexical_constrained", "lexical_local", "lexical_joint", "lexical_evidence", "lexical_attachment"),
                             local_roles=parser_type == "lexical_local",
                             joint_roles=parser_type == "lexical_joint",
                             evidence_roles=parser_type == "lexical_evidence",
                             attachment_roles=parser_type == "lexical_attachment")
    reasoner = {"attention": Reasoner, "bound": BoundReasoner, "stable": StableReasoner}[architecture]
    factories = {
        "parser": parser_factory,
        "raw": lambda: RawReasoner(c["width"], c["local_layers"], c["layers"], representation),
        "raw_aux": lambda: RawReasoner(c["width"], c["local_layers"], c["layers"], representation),
        "oracle": lambda: reasoner(c["width"], c["layers"]),
        "hybrid": lambda: reasoner(c["width"], c["layers"], source=True),
    }
    if set(state["states"]) - factories.keys():
        raise ValueError("Unknown component in structural checkpoint")
    models = {name: factories[name]() for name in state["states"]}
    for name, model in models.items():
        model.load_state_dict(state["states"][name])
        model.architecture = architecture
        model.to(device).eval()
    return models, state["metadata"]


@torch.no_grad()
def predict(models, text, mode="hybrid", device="cpu"):
    """Run one supported inference path for a single controlled-language text.

    ``raw`` predicts directly from text; ``parsed`` uses parser rows with the
    row-only learned reasoner; ``hybrid`` also supplies parser features; and
    ``executor`` applies deterministic operations to the predicted rows.
    """
    if mode == "oracle":
        raise ValueError("Oracle rows are diagnostic labels, not a text inference mode")
    stable = any(isinstance(component, StableReasoner) or
                 getattr(component, "architecture", None) == "stable"
                 for component in models.values())
    if stable and (not isinstance(text, str) or text.count("?") != 1 or not text.rstrip().endswith("?")):
        raise ValueError("Stable inference requires exactly one final question")
    if mode in ("raw", "raw_aux"):
        if mode not in models:
            raise ValueError(f"This checkpoint has no {mode} component")
        surface = prepare_texts([text], device, models[mode].representation,
                                vocabulary=getattr(models[mode], "vocabulary", None))
        return decode_answers(models[mode](surface))[0]
    if mode not in ("parsed", "hybrid", "executor"):
        raise ValueError("Unknown inference mode")
    if "parser" not in models:
        raise ValueError("This checkpoint has no text parser")
    surface = prepare_texts([text], device, models["parser"].representation,
                            vocabulary=getattr(models["parser"], "vocabulary", None))
    output = models["parser"](surface)
    rows = parsed_rows(output, surface)
    if stable and not bool(StableReasoner.valid_queries(rows, surface.sentence_mask).all()):
        raise ValueError("The parsed final query is missing, unsupported, or has invalid entities")
    if mode == "executor":
        return execute([tuple(int(v) for v in row) for row in rows[0].cpu().tolist()])
    key = "oracle" if mode == "parsed" else "hybrid"
    if key not in models:
        raise ValueError(f"This checkpoint has no {mode} reasoning path")
    model = models[key]
    return decode_answers(model(rows, surface.sentence_mask,
                                output["features"] if mode == "hybrid" else None))[0]


def main():
    """Load one checkpoint and run the public single-request CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/semantic_candidate.pt",
                        help="Checkpoint path (default: current experimental candidate)")
    parser.add_argument("--text", required=True)
    parser.add_argument("--mode", choices=("raw", "raw_aux", "parsed", "hybrid", "executor"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    models, metadata = load_checkpoint(args.model, args.device)
    mode = args.mode or metadata["selected_mode"]
    print(json.dumps({"mode": mode, "answer": predict(models, args.text, mode, args.device),
                      "scope": "Controlled-language research model, not unrestricted language understanding."}))


if __name__ == "__main__":
    main()
