"""Pure, deterministic controlled-language tasks for the structural comparison.

Rows are supervision/diagnostic oracle input, never features for a text parser.
All ordinary words are lowercase and proper names are capitalized. Surface
identity normalization consequently needs no generator roster or answer labels.
The names split checks this convention's invariance, not learned name discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import random
import re
import string
from typing import Iterable

# Every program is a sequence of five-integer rows.  ``kind`` selects the
# operation; entity fields contain first-mention IDs; ``value`` stores a coin
# amount; and ``active`` distinguishes events that happened from negated ones.
PAD, INITIAL, TRANSFER, BEFORE, QUERY_BALANCE, QUERY_BEFORE = range(6)
KINDS = ("pad", "initial", "transfer", "before", "query_balance", "query_before")
ROW_FIELDS = ("kind", "entity_a", "entity_b", "value", "active")
TRAIN_NAMES = ("Alice", "Bob", "Clara", "David", "Elena", "Felix", "Grace", "Henry")
HELDOUT_NAMES = ("Inez", "Jamal", "Kiran", "Leila", "Marek", "Nadia", "Omar", "Petra")
SHIFTS = ("iid", "names", "wording", "length", "numbers", "composition")
STABILITY_PROTOCOL = "stability_v1"
STABILITY_SHIFTS = ("iid", "names", "wording", "length", "numbers", "composition",
                    "combined", "lexical_unseen")
STABILITY_HELDOUT_NAMES = ("Quinn", "Ravi", "Sana", "Tomas", "Uma", "Vera", "Wesley", "Yara")
INTERPRETATION_PROTOCOL = "interpretation_v1"
INTERPRETATION_SHIFTS = STABILITY_SHIFTS
INTERPRETATION_NAMES = ("Amina", "Bruno", "Cyrus", "Dalia", "Esme", "Farid", "Gwen", "Hugo")
SCALING_PROTOCOL = "scaling_v1"
SCALING_SHIFTS = STABILITY_SHIFTS
SCALING_PAIRED_PREFIX = 6000
BINDING_PROTOCOL = "binding_v1"
BINDING_SHIFTS = STABILITY_SHIFTS
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+\b")
SENTENCE_PATTERN = re.compile(r"[^.?]+[.?]")
Row = tuple[int, int, int, int, int]


# Core task schema and surface normalization

@dataclass(frozen=True)
class Example:
    """One generated task in both human-readable and executable forms.

    ``text`` is what a model reads. ``rows`` are the construction-time meaning
    labels used for supervision and diagnostics. They are deliberately kept
    separate so inference cannot obtain the answer from hidden metadata.
    """

    text: str
    sentences: tuple[str, ...]
    rows: tuple[Row, ...]
    answer: str
    task: str


def surface_names(text: str) -> tuple[str, ...]:
    """Names in actual surface order; no hidden entity IDs are consulted."""
    return tuple(dict.fromkeys(NAME_PATTERN.findall(text)))


def split_sentences(text: str) -> tuple[str, ...]:
    """Split controlled prose and reject incomplete or unparsed leftovers."""
    matches = tuple(match.group().strip() for match in SENTENCE_PATTERN.finditer(text))
    if " ".join(matches) != " ".join(text.split()):
        raise ValueError("Text must consist of complete controlled-language sentences")
    return matches


def surface_sentences(text: str) -> tuple[str, ...]:
    """Tokenize visible sentences, replacing proper names by first-mention IDs.

    The returned features are identical for a consistent renaming of entities.
    Numbers are literal strings and no event type or semantic role is supplied.
    """
    identities = {name: f"e{i}" for i, name in enumerate(surface_names(text))}
    return tuple(NAME_PATTERN.sub(lambda match: identities[match.group()], sentence)
                 for sentence in split_sentences(text))


# Baseline sentence rendering and task generation

def _initial(a: str, amount: int, style: int, wording: bool) -> str:
    templates = (("{n} coins belong to {a} initially.",
                  "the starting balance of {a} is {n} coins.") if wording else
                 ("{a} starts with {n} coins.", "initially {a} holds {n} coins.",
                  "{a} has {n} coins at the start."))
    return templates[style % len(templates)].format(a=a, n=amount)


def _transfer(a: str, b: str, amount: int, active: int, style: int, wording: bool) -> str:
    if wording:
        templates = (("{a} hands {n} coins over to {b}.",
                      "{n} coins move from {a} into the possession of {b}.") if active else
                     ("{a} never hands {n} coins over to {b}.",
                      "no movement of {n} coins from {a} to {b} occurs."))
    else:
        templates = (("{a} gives {b} {n} coins.", "{b} receives {n} coins from {a}.",
                      "{a} transfers {n} coins to {b}.") if active else
                     ("{a} does not give {b} {n} coins.",
                      "{b} does not receive {n} coins from {a}.",
                      "{a} does not transfer {n} coins to {b}."))
    return templates[style % len(templates)].format(a=a, b=b, n=amount)


def _balance_query(a: str, style: int, wording: bool) -> str:
    templates = (("what is the final coin balance of {a}?",
                  "after these events how many coins belong to {a}?") if wording else
                 ("how many coins does {a} have now?", "what is {a}'s balance now?",
                  "how many coins is {a} holding now?"))
    return templates[style % len(templates)].format(a=a)


def _before(a: str, b: str, style: int, wording: bool) -> str:
    templates = (("{a} precedes {b}.", "{b} follows {a}.") if wording else
                 ("{a} is before {b}.", "{b} is after {a}.",
                  "{a} comes before {b}.", "{b} comes after {a}."))
    return templates[style % len(templates)].format(a=a, b=b)


def _before_query(a: str, b: str, style: int, wording: bool) -> str:
    templates = (("does {a} precede {b}?", "does {b} follow {a}?") if wording else
                 ("is {a} before {b}?", "is {b} after {a}?",
                  "does {a} come before {b}?"))
    return templates[style % len(templates)].format(a=a, b=b)


def _finish(sentences: list[str], named_rows: list[tuple], answer: str, task: str) -> Example:
    """Package rendered sentences and replace visible names with local IDs."""
    text = " ".join(sentences)
    identities = {name: index for index, name in enumerate(surface_names(text))}
    rows = tuple((kind, identities.get(a, -1), identities.get(b, -1), value, active)
                 for kind, a, b, value, active in named_rows)
    return Example(text, tuple(sentences), rows, answer, task)


def _accounting(rng: random.Random, shift: str) -> Example:
    """Generate one balance-tracking world and its final balance question."""
    names = rng.sample(HELDOUT_NAMES if shift == "names" else TRAIN_NAMES, 4)
    wording = shift == "wording"
    initial_bounds, amount_bounds = ((10, 16), (5, 8)) if shift == "numbers" else ((2, 9), (1, 4))
    balances = {name: rng.randint(*initial_bounds) for name in names}
    sentences, rows = [], []
    for name in rng.sample(names, len(names)):
        amount = balances[name]
        sentences.append(_initial(name, amount, rng.randrange(12), wording))
        rows.append((INITIAL, name, None, amount, 1))
    count = rng.randint(8, 10) if shift == "length" else rng.randint(2, 4)
    for _ in range(count):
        a, b = rng.sample(names, 2)
        amount, active = rng.randint(*amount_bounds), int(rng.random() >= .3)
        sentences.append(_transfer(a, b, amount, active, rng.randrange(12), wording))
        rows.append((TRANSFER, a, b, amount, active))
        if active:
            balances[a] -= amount
            balances[b] += amount
    query = rng.choice(names)
    sentences.append(_balance_query(query, rng.randrange(12), wording))
    rows.append((QUERY_BALANCE, query, None, 0, 1))
    return _finish(sentences, rows, str(balances[query]), "accounting")


def _relations(rng: random.Random, shift: str, wanted_yes: bool) -> Example:
    """Generate one ordering graph and a yes/no reachability question."""
    names = rng.sample(HELDOUT_NAMES if shift == "names" else TRAIN_NAMES, 4)
    wording = shift == "wording"
    order = rng.sample(names, len(names))
    facts = list(zip(order, order[1:]))
    if shift == "length":
        # Duplicate true assertions lengthen the input without adding entities,
        # changing reachability, or making the requested reasoning path longer.
        facts += rng.choices(facts, k=rng.randint(5, 7))
    rng.shuffle(facts)
    sentences, rows = [], []
    for a, b in facts:
        sentences.append(_before(a, b, rng.randrange(12), wording))
        rows.append((BEFORE, a, b, 0, 1))
    # IID questions need one or two edges. Composition alone holds out a path
    # of three edges; all axes keep four entities. Negation occurs in transfers.
    distance = 3 if shift == "composition" else rng.randint(1, 2)
    start = rng.randrange(len(order) - distance)
    left, right = order[start], order[start + distance]
    if not wanted_yes:
        left, right = right, left
    sentences.append(_before_query(left, right, rng.randrange(12), wording))
    rows.append((QUERY_BEFORE, left, right, 0, 1))
    return _finish(sentences, rows, "yes" if wanted_yes else "no", "relations")


def make_split(count: int, seed: int, shift: str = "iid",
               exclude: Iterable[str] = ()) -> list[Example]:
    """Return mixed tasks with unique prompts, balanced tasks and relation labels.

    Number shift applies only to accounting and composition only to relations;
    the other task is an unchanged control in those splits. Length adds transfer
    events or repeated relation facts, retaining four entities and query distance.
    Names, wording, and numeric ranges otherwise keep their training distributions.
    No files are read/written and no tensor frameworks or old experiments run.
    """
    if count < 0 or shift not in SHIFTS:
        raise ValueError("Nonnegative count and a supported shift are required")
    rng, seen, examples = random.Random(seed), set(exclude), []
    seen_surfaces = {surface_sentences(text) for text in seen}
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("Could not generate enough unique prompts")
        index = len(examples)
        # Alternation keeps the two task types and yes/no relation labels balanced.
        example = (_accounting(rng, shift) if index % 2 == 0 else
                   _relations(rng, shift, (index // 2) % 2 == 0))
        normalized = surface_sentences(example.text)
        # Exclude exact prompts and renamed copies of the same visible prompt.
        if example.text in seen or normalized in seen_surfaces:
            continue
        seen.add(example.text)
        seen_surfaces.add(normalized)
        examples.append(example)
    rng.shuffle(examples)
    return examples


# Stability protocol: familiar wording, isolated shifts, and combined shifts

# Fresh syntax is defined before training and is never selected by the IID
# generator. Its content words are already present in development templates.
# The lexical challenge deliberately adds absent verbs and is reported separately.
_STABILITY_SYNTAX = {
    "initial": ("at the start {a} has {n} coins.",
                "{a} initially holds {n} coins.",
                "{a}'s starting balance is {n} coins.",
                "at the start {n} coins belong to {a}."),
    "transfer": ("{a} gives {n} coins to {b}.",
                 "from {a} {b} receives {n} coins.",
                 "{a} transfers to {b} {n} coins.",
                 "{n} coins move to {b} from {a}."),
    "inactive_transfer": ("{a} does not give {n} coins to {b}.",
                          "from {a} {b} does not receive {n} coins.",
                          "{a} does not transfer to {b} {n} coins.",
                          "{n} coins never move to {b} from {a}."),
    "balance_query": ("{a} now has how many coins?",
                      "how many coins belong to {a} now?",
                      "what is the coin balance of {a} after these events?",
                      "what is the balance of {a} now?"),
    "before": ("before {b} comes {a}.", "after {a} comes {b}.",
               "before {b} is {a}.", "after {a} is {b}."),
    "before_query": ("before {b} does {a} come?", "after {a} does {b} come?",
                     "does {a} come before {b} now?", "does {b} come after {a}?"),
}
_STABILITY_LEXICAL = {
    "initial": ("{a} owns {n} coins initially.", "{a} possesses {n} coins at the start."),
    "transfer": ("{a} donates {n} coins to {b}.", "{b} obtains {n} coins from {a}."),
    "inactive_transfer": ("{a} does not donate {n} coins to {b}.",
                          "{b} does not obtain {n} coins from {a}."),
    "balance_query": ("how many coins does {a} possess now?",
                      "how many coins does {a} own now?"),
    "before": ("{a} predates {b}.", "{b} succeeds {a} in the order."),
    "before_query": ("does {a} predate {b}?", "does {b} succeed {a} in the order?"),
}
_STABILITY_INITIAL_VALUES = tuple(range(-50, 2)) + tuple(range(10, 51))
_STABILITY_TRANSFER_VALUES = (0,) + tuple(range(5, 21))


def _stability_sentence(rng: random.Random, event: str, family: str,
                        a: str, b: str | None = None, amount: int = 0) -> str:
    """Render one labeled event into a controlled-language sentence."""
    if family != "development":
        templates = _STABILITY_SYNTAX if family == "wording" else _STABILITY_LEXICAL
        return rng.choice(templates[event]).format(a=a, b=b, n=amount)
    # Previously tested wording is now explicitly training/development data.
    # Equal family probability includes all old templates without changing the
    # old renderer functions or their reproducible random-number consumption.
    old_wording, style = bool(rng.randrange(2)), rng.randrange(12)
    if event == "initial":
        return _initial(a, amount, style, old_wording)
    if event in ("transfer", "inactive_transfer"):
        return _transfer(a, b, amount, int(event == "transfer"), style, old_wording)
    if event == "balance_query":
        return _balance_query(a, style, old_wording)
    if event == "before":
        return _before(a, b, style, old_wording)
    if event == "before_query":
        return _before_query(a, b, style, old_wording)
    raise ValueError("Unsupported stability event")


def _stability_accounting(rng: random.Random, shift: str) -> Example:
    shifted_names = shift in ("names", "combined")
    names = rng.sample(STABILITY_HELDOUT_NAMES if shifted_names else TRAIN_NAMES, 4)
    family = ("wording" if shift in ("wording", "combined") else
              "lexical_unseen" if shift == "lexical_unseen" else "development")
    shifted_numbers = shift in ("numbers", "combined")
    balances = {name: rng.choice(_STABILITY_INITIAL_VALUES) if shifted_numbers else rng.randint(2, 9)
                for name in names}
    sentences, rows = [], []
    for name in rng.sample(names, len(names)):
        value = balances[name]
        sentences.append(_stability_sentence(rng, "initial", family, name, amount=value))
        rows.append((INITIAL, name, None, value, 1))
    count = rng.randint(8, 16) if shift in ("length", "combined") else rng.randint(2, 4)
    for _ in range(count):
        a, b = rng.sample(names, 2)
        amount = rng.choice(_STABILITY_TRANSFER_VALUES) if shifted_numbers else rng.randint(1, 4)
        active = int(rng.random() >= .3)
        event = "transfer" if active else "inactive_transfer"
        sentences.append(_stability_sentence(rng, event, family, a, b, amount))
        rows.append((TRANSFER, a, b, amount, active))
        if active:
            balances[a] -= amount
            balances[b] += amount
    query = rng.choice(names)
    sentences.append(_stability_sentence(rng, "balance_query", family, query))
    rows.append((QUERY_BALANCE, query, None, 0, 1))
    return _finish(sentences, rows, str(balances[query]), "accounting")


def _stability_relations(rng: random.Random, shift: str, wanted_yes: bool) -> Example:
    shifted_names = shift in ("names", "combined")
    names = rng.sample(STABILITY_HELDOUT_NAMES if shifted_names else TRAIN_NAMES, 4)
    family = ("wording" if shift in ("wording", "combined") else
              "lexical_unseen" if shift == "lexical_unseen" else "development")
    order = rng.sample(names, 4)
    facts = list(zip(order, order[1:]))
    if shift in ("length", "combined"):
        facts += rng.choices(facts, k=rng.randint(5, 13))
    rng.shuffle(facts)
    sentences, rows = [], []
    for a, b in facts:
        sentences.append(_stability_sentence(rng, "before", family, a, b))
        rows.append((BEFORE, a, b, 0, 1))
    distance = 3 if shift in ("composition", "combined") else rng.randint(1, 2)
    start = rng.randrange(4 - distance)
    a, b = order[start], order[start + distance]
    if not wanted_yes:
        a, b = b, a
    sentences.append(_stability_sentence(rng, "before_query", family, a, b))
    rows.append((QUERY_BEFORE, a, b, 0, 1))
    return _finish(sentences, rows, "yes" if wanted_yes else "no", "relations")


def make_stability_split(count: int, seed: int, shift: str = "iid",
                         exclude: Iterable[str] = ()) -> list[Example]:
    """Generate the separate stability protocol without altering old results.

    IID uses all previously observed wording as development material, initial
    balances 2..9, transfers 1..4, and two to four transfer events. Fresh wording
    rearranges familiar words; lexical_unseen separately uses absent verbs.
    Numbers holds initial values in [-50, 1] or [10, 50] and transfer amounts
    in {0} or [5, 20], excluding each role's training range. Negative starting
    balances represent debt; transfers remain nonnegative, including no-op zero.
    Length uses 8..16 transfers or 8..16 repeated graph assertions. Composition
    holds out three-edge questions. Combined joins new names, syntax, amounts,
    length, and composition. Four entities and at most 21 clauses remain fixed.
    Numbers changes only accounting and composition only relations; the other
    task in those splits remains a control. Names tests surface normalization.

    No final seeds are created implicitly. Callers must freeze candidate settings
    before requesting held-out splits and exclude earlier raw prompts; both raw
    and normalized inputs are deduplicated. This function does no model work,
    accesses no files, and returns the existing Example/semantic-row contract.
    """
    if count < 0 or shift not in STABILITY_SHIFTS:
        raise ValueError("Nonnegative count and a supported stability shift are required")
    rng, seen, examples = random.Random(seed), set(exclude), []
    seen_surfaces = {surface_sentences(text) for text in seen}
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("Could not generate enough unique stability prompts")
        index = len(examples)
        example = (_stability_accounting(rng, shift) if index % 2 == 0 else
                   _stability_relations(rng, shift, (index // 2) % 2 == 0))
        normalized = surface_sentences(example.text)
        if example.text in seen or normalized in seen_surfaces:
            continue
        seen.add(example.text)
        seen_surfaces.add(normalized)
        examples.append(example)
    rng.shuffle(examples)
    return examples


# Interpretation protocol: test whether wording preserves semantic roles

# Each entry is a complete syntactic arrangement, with lexical choices sampled
# independently. Passive, fronting and relative/cleft ingredients occur in
# training; development/final hold out their full arrangements/combinations.
# Positive and negative transfer entries align by index for minimal contrasts.
INTERPRETATION_FAMILIES = {
    "train": {
        "initial": (
            "{a} {own} {n} coins at the start.", "initially {a} {own} {n} coins.",
            "the starting balance of {a} is {n} coins.", "{n} coins belong to {a} initially.",
            "{a} starts with {n} coins.", "at the start {n} coins are held by {a}.",
            "it is {a} who {own} {n} coins initially.",
            "{a} is in possession of {n} coins at the start.",
            "the number of coins held by {a} initially is {n}.",
            "there are {n} coins held by {a} at the start."),
        "transfer": (
            "{a} {give} {b} {n} coins.", "{a} {give} {n} coins to {b}.",
            "{b} {receive} {n} coins from {a}.", "{n} coins are {given} by {a} to {b}.",
            "{b} is {given} {n} coins by {a}.", "to {b} {a} {give} {n} coins.",
            "from {a} {b} {receive} {n} coins.", "it is {a} who {give} {n} coins to {b}.",
            "it is {n} coins that {a} {give} to {b}.",
            "the transfer of {n} coins goes from {a} to {b}.",
            "{a} is the one who {give} {b} {n} coins."),
        "inactive_transfer": (
            "{a} does not {give_base} {b} {n} coins.", "{a} does not {give_base} {n} coins to {b}.",
            "{b} does not {receive_base} {n} coins from {a}.",
            "{n} coins are not {given} by {a} to {b}.",
            "{b} is not {given} {n} coins by {a}.", "to {b} {a} does not {give_base} {n} coins.",
            "from {a} {b} does not {receive_base} {n} coins.",
            "it is {a} who does not {give_base} {n} coins to {b}.",
            "it is {n} coins that {a} does not {give_base} to {b}.",
            "no transfer of {n} coins goes from {a} to {b}.",
            "{a} is the one who does not {give_base} {b} {n} coins."),
        "balance_query": (
            "how many coins does {a} have now?", "what is {a}'s balance now?",
            "how many coins belong to {a} now?", "what is the final coin balance of {a}?",
            "{a} now has how many coins?", "what is the number of coins held by {a} now?",
            "how many coins are held by {a} now?", "what is the balance that {a} has now?"),
        "before": (
            "{a} is before {b}.", "{b} is after {a}.", "{a} {precede} {b}.", "{b} {follow} {a}.",
            "before {b} comes {a}.", "after {a} comes {b}.",
            "it is {a} who comes before {b}.", "it is {b} who comes after {a}.",
            "the one before {b} is {a}.", "{b} is the one after {a}."),
        "before_query": (
            "is {a} before {b}?", "is {b} after {a}?", "does {a} come before {b}?",
            "does {b} come after {a}?", "does {a} {precede_base} {b}?",
            "does {b} {follow_base} {a}?", "before {b} does {a} come?", "after {a} does {b} come?",
            "is {a} the one before {b}?", "is {b} the one after {a}?",
            "is it {a} who comes before {b}?", "is it {b} who comes after {a}?"),
    },
    "development": {
        "initial": ("the coins that {a} {own} at the start number {n}.",
                    "there are {n} coins in the possession of {a} initially."),
        "transfer": ("{n} coins are {given} to {b} by {a}.",
                     "to {b} {n} coins are {given} by {a}.",
                     "it is from {a} that {b} {receive} {n} coins."),
        "inactive_transfer": ("{n} coins are not {given} to {b} by {a}.",
                              "to {b} {n} coins are not {given} by {a}.",
                              "it is from {a} that {b} does not {receive_base} {n} coins."),
        "balance_query": ("what is the number of coins that {a} has now?",
                          "the balance of {a} is what now?"),
        "before": ("{a} is the one who comes before {b}.",
                   "{b} is the one who comes after {a}.", "it is before {b} that {a} comes."),
        "before_query": ("is {a} the one who comes before {b}?",
                         "is {b} the one who comes after {a}?", "is it before {b} that {a} comes?"),
    },
    "final": {
        "initial": ("what {a} {own} at the start is {n} coins.",
                    "the {n} coins held initially are in the possession of {a}."),
        "transfer": ("from {a} {n} coins are {given} to {b}.",
                     "it is to {b} that {n} coins are {given} by {a}.",
                     "{b} is the one who {receive} from {a} {n} coins."),
        "inactive_transfer": ("from {a} {n} coins are not {given} to {b}.",
                              "it is to {b} that {n} coins are not {given} by {a}.",
                              "{b} is the one who does not {receive_base} from {a} {n} coins."),
        "balance_query": ("what balance is now held by {a}?",
                          "the coins that {a} has now number how many?"),
        "before": ("it is after {a} that {b} comes.",
                   "the one who comes before {b} is {a}.", "the one who comes after {a} is {b}."),
        "before_query": ("is it after {a} that {b} comes?",
                         "is the one who comes before {b} {a}?", "is the one who comes after {a} {b}?"),
    },
}
INTERPRETATION_LEXICON = {
    "own": ("has", "holds", "owns", "possesses"),
    "give": (("gives", "give", "given"), ("hands", "hand", "handed"),
             ("passes", "pass", "passed"), ("transfers", "transfer", "transferred")),
    "receive": (("receives", "receive"), ("obtains", "obtain")),
    "precede": (("precedes", "precede"), ("predates", "predate")),
    "follow": (("follows", "follow"), ("succeeds", "succeed")),
}
_INTERPRETATION_UNSEEN = {
    "own": ("retains",), "give": (("remits", "remit", "remitted"),),
    "receive": (("acquires", "acquire"),), "precede": (("antedates", "antedate"),),
    "follow": (("postdates", "postdate"),),
}


def interpretation_world_key(text: str) -> tuple[str, ...]:
    """Identity-normalized presented facts, omitting the final query.

    Equivalent worlds expressed with different wording remain allowed; this
    prevents the same presented world leaking through a different query.
    """
    return surface_sentences(text)[:-1]


def _interpretation_fields(rng, unseen=False):
    lexicon = _INTERPRETATION_UNSEEN if unseen else INTERPRETATION_LEXICON
    give, receive = rng.choice(lexicon["give"]), rng.choice(lexicon["receive"])
    precede, follow = rng.choice(lexicon["precede"]), rng.choice(lexicon["follow"])
    return {"own": rng.choice(lexicon["own"]), "give": give[0], "give_base": give[1], "given": give[2],
            "receive": receive[0], "receive_base": receive[1], "precede": precede[0],
            "precede_base": precede[1], "follow": follow[0], "follow_base": follow[1]}


def interpretation_sentence(rng, event, a, b=None, amount=0, stage="train", unseen=False,
                            family_index=None):
    """Render labels into source text for data construction, never inference.

    The event/family identifiers are not returned to or used by a model. Verb
    inflections share one lexical draw so polarity contrasts preserve the lemma.
    """
    templates = INTERPRETATION_FAMILIES[stage][event]
    index = rng.randrange(len(templates)) if family_index is None else family_index
    return templates[index].format(a=a, b=b, n=amount, **_interpretation_fields(rng, unseen))


def _interpretation_style(shift, stage):
    if stage not in INTERPRETATION_FAMILIES:
        raise ValueError("Interpretation stage must be train, development, or final")
    if shift in ("wording", "combined"):
        if stage == "train":
            raise ValueError("Wording/combined require an explicit development or final stage")
        return stage
    return "train"


def _interpretation_world(rng, shift, stage, task, wanted_yes):
    """Sample a latent world, then render it with the requested grammar family."""
    family = _interpretation_style(shift, stage)
    names = rng.sample(INTERPRETATION_NAMES if shift in ("names", "combined") else TRAIN_NAMES, 4)
    sentences, rows = [], []

    def render(event, a, b=None, amount=0):
        return interpretation_sentence(rng, event, a, b, amount, family, shift == "lexical_unseen")

    if task == "accounting":
        wide = shift in ("numbers", "combined")
        balances = {name: rng.choice(_STABILITY_INITIAL_VALUES) if wide else rng.randint(2, 9)
                    for name in names}
        for name in rng.sample(names, 4):
            sentences.append(render("initial", name, amount=balances[name]))
            rows.append((INITIAL, name, None, balances[name], 1))
        count = rng.randint(8, 16) if shift in ("length", "combined") else rng.randint(2, 4)
        for _ in range(count):
            a, b = rng.sample(names, 2)
            amount = rng.choice(_STABILITY_TRANSFER_VALUES) if wide else rng.randint(1, 4)
            active = rng.randrange(2)
            sentences.append(render("transfer" if active else "inactive_transfer", a, b, amount))
            rows.append((TRANSFER, a, b, amount, active))
            if active:
                balances[a] -= amount
                balances[b] += amount
        query = rng.choice(names)
        sentences.append(render("balance_query", query))
        rows.append((QUERY_BALANCE, query, None, 0, 1))
        return _finish(sentences, rows, str(balances[query]), task)
    order = rng.sample(names, 4)
    facts = list(zip(order, order[1:]))
    if shift in ("length", "combined"):
        facts += rng.choices(facts, k=rng.randint(5, 13))
    rng.shuffle(facts)
    for a, b in facts:
        sentences.append(render("before", a, b))
        rows.append((BEFORE, a, b, 0, 1))
    distance = 3 if shift in ("composition", "combined") else rng.randint(1, 2)
    start = rng.randrange(4 - distance)
    a, b = order[start], order[start + distance]
    if not wanted_yes:
        a, b = b, a
    sentences.append(render("before_query", a, b))
    rows.append((QUERY_BEFORE, a, b, 0, 1))
    return _finish(sentences, rows, "yes" if wanted_yes else "no", task)


def make_interpretation_split(count: int, seed: int, shift: str = "iid",
                              exclude: Iterable[str] = (), stage: str = "train") -> list[Example]:
    """Fresh interpretation protocol; old generators remain unchanged.

    IID/isolated numeric, length and composition axes always use training syntax.
    Wording/combined require stage=development or final and use disjoint whole
    sentence-family arrangements. Final vocabulary is covered by training;
    lexical_unseen changes verb vocabulary separately using training syntax.
    Initial/transfer ranges, four identities and 21-clause maximum match stability.
    Tasks and relation answers are balanced, transfer activity is equiprobable,
    and role orientations vary independently of renderer choice. Full prompts
    and their normalized presented facts are deduplicated within/across splits.
    No family labels, row labels, solutions or hidden identifiers enter raw text.
    """
    if count < 0 or shift not in INTERPRETATION_SHIFTS:
        raise ValueError("Nonnegative count and a supported interpretation shift are required")
    _interpretation_style(shift, stage)
    rng, seen, examples = random.Random(seed), set(exclude), []
    normalized = {surface_sentences(text) for text in seen}
    worlds = {interpretation_world_key(text) for text in seen}
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("Could not generate enough unique interpretation worlds")
        index = len(examples)
        example = _interpretation_world(rng, shift, stage,
                                        "accounting" if index % 2 == 0 else "relations",
                                        (index // 2) % 2 == 0)
        key, world = surface_sentences(example.text), interpretation_world_key(example.text)
        if example.text in seen or key in normalized or world in worlds:
            continue
        seen.add(example.text)
        normalized.add(key)
        worlds.add(world)
        examples.append(example)
    rng.shuffle(examples)
    return examples


def _interpretation_answer(rows):
    """Construction-time labels for query/contrast expansion, independent of execute."""
    balances, edges = {}, set()
    for kind, a, b, value, active in rows[:-1]:
        if kind == INITIAL:
            balances[a] = value
        elif kind == TRANSFER and active:
            balances[a] -= value
            balances[b] += value
        elif kind == BEFORE and active:
            edges.add((a, b))
    kind, a, b, _, _ = rows[-1]
    if kind == QUERY_BALANCE:
        return str(balances[a])
    for _ in range(3):
        edges |= {(x, z) for x, y in edges for other, z in edges if y == other}
    return "yes" if (a, b) in edges else "no"


def all_query_examples(examples, seed=0, stage="train", shift="iid",
                       protocol="interpretation", diversity="broad") -> list[list[Example]]:
    """Expand already-separated worlds into four or twelve queries, grouped by world.

    This is evaluation/data construction, not inference. Do not independently
    split the returned queries: facts within each group are intentionally shared.
    """
    rng, groups = random.Random(seed), []
    family, render, _ = _expansion_style(protocol, shift, stage, diversity)
    for example in examples:
        names = surface_names(example.text)
        pairs = ((a, -1) for a in range(4)) if example.task == "accounting" else (
            (a, b) for a in range(4) for b in range(4) if a != b)
        group = []
        for a, b in pairs:
            kind = QUERY_BALANCE if b == -1 else QUERY_BEFORE
            event = "balance_query" if b == -1 else "before_query"
            query = render(rng, event, names[a], names[b] if b != -1 else None,
                           stage=family, unseen=shift == "lexical_unseen")
            rows = example.rows[:-1] + ((kind, a, b, 0, 1),)
            sentences = example.sentences[:-1] + (query,)
            group.append(Example(" ".join(sentences), sentences, rows,
                                 _interpretation_answer(rows), example.task))
        groups.append(group)
    return groups


def interpretation_contrast_pairs(examples, seed=0, stage="train", shift="iid",
                                  protocol="interpretation", diversity="broad"):
    """Return (base, changed, role|polarity) pairs after splitting base worlds.

    Accounting contrasts alter one nonzero transfer and query its sender; role
    swaps reverse debit/credit while polarity toggles only occurrence. Relation
    contrasts reverse one final query. Paired lexical draws are held constant.
    """
    rng, pairs = random.Random(seed), []
    family, render, families = _expansion_style(protocol, shift, stage, diversity)
    for example in examples:
        names = surface_names(example.text)
        if example.task == "relations":
            kind, a, b, value, active = example.rows[-1]
            render_seed = rng.randrange(2**31)
            variants = []
            for left, right in ((a, b), (b, a)):
                rows = example.rows[:-1] + ((kind, left, right, value, active),)
                query = render(random.Random(render_seed), "before_query", names[left], names[right],
                               stage=family, unseen=shift == "lexical_unseen")
                sentences = example.sentences[:-1] + (query,)
                variants.append(Example(" ".join(sentences), sentences, rows,
                                        _interpretation_answer(rows), example.task))
            pairs.append((*variants, "role"))
            continue
        candidates = [i for i, row in enumerate(example.rows) if row[0] == TRANSFER and row[3] != 0]
        if not candidates:
            continue
        index = rng.choice(candidates)
        _, a, b, amount, _ = example.rows[index]
        for contrast in ("role", "polarity"):
            render_seed, family_index = rng.randrange(2**31), rng.randrange(len(families[family]["transfer"]))
            variants = []
            for changed in (False, True):
                left, right = (b, a) if changed and contrast == "role" else (a, b)
                active = int(not (changed and contrast == "polarity"))
                rows = list(example.rows)
                rows[index], rows[-1] = (TRANSFER, left, right, amount, active), (QUERY_BALANCE, a, -1, 0, 1)
                sentences = list(example.sentences)
                sentences[index] = render(random.Random(render_seed),
                    "transfer" if active else "inactive_transfer", names[left], names[right], amount,
                    stage=family, unseen=shift == "lexical_unseen", family_index=family_index)
                sentences[-1] = render(random.Random(render_seed), "balance_query", names[a], stage=family)
                rows, sentences = tuple(rows), tuple(sentences)
                variants.append(Example(" ".join(sentences), sentences, rows,
                                        _interpretation_answer(rows), example.task))
            pairs.append((*variants, contrast))
    return pairs


# Scaling protocol: compare narrow and broad language coverage on matched worlds

# The narrow arm retains the previous training arrangements. The broad arm
# treats every previously observed interpretation family as training material,
# then adds combinations before this protocol's fresh development/final study.
# No final model predictions informed these definitions.
SCALING_FAMILIES = {
    "narrow": {event: tuple(templates) for event, templates in INTERPRETATION_FAMILIES["train"].items()},
    "broad": {event: tuple(dict.fromkeys(template
        for stage in ("train", "development", "final")
        for template in INTERPRETATION_FAMILIES[stage][event]))
        for event in INTERPRETATION_FAMILIES["train"]},
    "development": {
        "initial": (
            "the starting number of coins held by {a} is {n}.",
            "the {n} coins that {a} {own} are the starting balance.",
            "initially in the possession of {a} are {n} coins.",
            "at the start the balance held by {a} is {n} coins."),
        "transfer": (
            "by {a} {n} coins are {given} to {b}.",
            "it is {b} who is {given} {n} coins by {a}.",
            "{n} coins are what {b} {receive} from {a}.",
            "the one who {receive} from {a} the {n} coins is {b}.",
            "the {n} coins {given} by {a} are {given} to {b}.",
            "it is {a} who {give} to {b} the {n} coins."),
        "inactive_transfer": (
            "by {a} {n} coins are not {given} to {b}.",
            "it is {b} who is not {given} {n} coins by {a}.",
            "{n} coins are what {b} does not {receive_base} from {a}.",
            "the one who does not {receive_base} from {a} the {n} coins is {b}.",
            "the {n} coins not {given} by {a} are not {given} to {b}.",
            "it is {a} who does not {give_base} to {b} the {n} coins."),
        "balance_query": (
            "what is the balance now in the possession of {a}?",
            "how many coins now belong to the one who is {a}?",
            "the number of coins now held by {a} is what?",
            "what number of coins is in the possession of {a} now?"),
        "before": (
            "in the order {a} comes before {b}.",
            "in the order {b} comes after {a}.",
            "it is {a} who is the one before {b}.",
            "it is {b} who is the one after {a}.",
            "the one who is before {b} is {a}.",
            "the one who is after {a} is {b}.",
            "before {b} is the one who is {a}.",
            "after {a} is the one who is {b}."),
        "before_query": (
            "in the order does {a} come before {b}?",
            "in the order does {b} come after {a}?",
            "is it {a} who is the one before {b}?",
            "is it {b} who is the one after {a}?",
            "is the one who is before {b} {a}?",
            "is the one who is after {a} {b}?",
            "before {b} is the one who is {a}?",
            "after {a} is the one who is {b}?"),
    },
    "final": {
        "initial": (
            "at the start the coins in the possession of {a} number {n}.",
            "the balance that {a} has at the start is {n} coins.",
            "{n} is the number of coins that {a} holds initially.",
            "initially the number of coins held by {a} is {n}."),
        "transfer": (
            "the {n} coins that {b} {receive} are {given} by {a}.",
            "the one who {give} {n} coins to {b} is {a}.",
            "it is {n} coins that are {given} to {b} by {a}.",
            "{b} {receive} from {a} the {n} coins that are {given}.",
            "the one who {receive} the {n} coins {given} by {a} is {b}.",
            "{n} coins are {given} by the one who is {a} to {b}."),
        "inactive_transfer": (
            "the {n} coins that {b} does not {receive_base} are not {given} by {a}.",
            "the one who does not {give_base} {n} coins to {b} is {a}.",
            "it is {n} coins that are not {given} to {b} by {a}.",
            "{b} does not {receive_base} from {a} the {n} coins that are not {given}.",
            "the one who does not {receive_base} the {n} coins not {given} by {a} is {b}.",
            "{n} coins are not {given} by the one who is {a} to {b}."),
        "balance_query": (
            "how many coins are now in the possession of the one who is {a}?",
            "the balance that is now held by {a} is what?",
            "what is the number now of coins in the possession of {a}?",
            "what number of coins does the one who is {a} have now?"),
        "before": (
            "{a} comes before the one who is {b}.",
            "{b} comes after the one who is {a}.",
            "the one who is {a} comes before {b}.",
            "the one who is {b} comes after {a}.",
            "{a} is before the one who is {b} in the order.",
            "{b} is after the one who is {a} in the order.",
            "it is {a} who comes before the one who is {b}.",
            "it is {b} who comes after the one who is {a}."),
        "before_query": (
            "does {a} come before the one who is {b}?",
            "does {b} come after the one who is {a}?",
            "does the one who is {a} come before {b}?",
            "does the one who is {b} come after {a}?",
            "is {a} before the one who is {b} in the order?",
            "is {b} after the one who is {a} in the order?",
            "is it {a} who comes before the one who is {b}?",
            "is it {b} who comes after the one who is {a}?"),
    },
}
_SCALING_BROAD_ADDITIONS = {
    "initial": (
        "{a} is the one who initially {own} {n} coins.",
        "the number of coins in the possession of {a} at the start is {n}.",
        "at the start {n} is the number of coins held by {a}.",
        "{n} coins are what {a} {own} initially."),
    "transfer": (
        "{a} is the one who {give} {n} coins to {b}.",
        "{b} is the one who {receive} {n} coins from {a}.",
        "it is {a} who is the one who {give} {b} {n} coins.",
        "it is {b} who {receive} from {a} {n} coins.",
        "{n} coins are the coins that {a} {give} to {b}.",
        "to {b} are {given} {n} coins by {a}.",
        "{n} coins are {given} by the one who is {a} to the one who is {b}.",
        "{n} coins are {given} to the one who is {b} by the one who is {a}.",
        "the one who is {a} {give} the one who is {b} {n} coins.",
        "the one who is {b} {receive} {n} coins from the one who is {a}.",
        "from the one who is {a} the one who is {b} {receive} {n} coins."),
    "inactive_transfer": (
        "{a} is the one who does not {give_base} {n} coins to {b}.",
        "{b} is the one who does not {receive_base} {n} coins from {a}.",
        "it is {a} who is the one who does not {give_base} {b} {n} coins.",
        "it is {b} who does not {receive_base} from {a} {n} coins.",
        "{n} coins are the coins that {a} does not {give_base} to {b}.",
        "to {b} are not {given} {n} coins by {a}.",
        "{n} coins are not {given} by the one who is {a} to the one who is {b}.",
        "{n} coins are not {given} to the one who is {b} by the one who is {a}.",
        "the one who is {a} does not {give_base} the one who is {b} {n} coins.",
        "the one who is {b} does not {receive_base} {n} coins from the one who is {a}.",
        "from the one who is {a} the one who is {b} does not {receive_base} {n} coins."),
    "balance_query": (
        "how many coins does the one who is {a} have now?",
        "what is the number of coins in the possession of {a} now?",
        "what is now the balance of {a}?",
        "the number of coins that {a} has now is what?"),
    "before": (
        "{a} is before {b} in the order.", "{b} is after {a} in the order.",
        "the one before {b} in the order is {a}.", "the one after {a} in the order is {b}.",
        "{a} is the one before {b} in the order.", "{b} is the one after {a} in the order.",
        "it is before {b} in the order that {a} comes.",
        "it is after {a} in the order that {b} comes.",
        "the one who is {a} {precede} the one who is {b}.",
        "the one who is {b} {follow} the one who is {a}."),
    "before_query": (
        "is {a} before {b} in the order?", "is {b} after {a} in the order?",
        "is the one before {b} in the order {a}?", "is the one after {a} in the order {b}?",
        "is {a} the one before {b} in the order?", "is {b} the one after {a} in the order?",
        "is it before {b} in the order that {a} comes?",
        "is it after {a} in the order that {b} comes?"),
}
for _event, _templates in _SCALING_BROAD_ADDITIONS.items():
    SCALING_FAMILIES["broad"][_event] += _templates
for _stage in ("narrow", "broad"):
    SCALING_FAMILIES[_stage]["before"] += (
        "{a} {precede} {b} in the order.", "{b} {follow} {a} in the order.")
    SCALING_FAMILIES[_stage]["before_query"] += (
        "does {a} {precede_base} {b} in the order?", "does {b} {follow_base} {a} in the order?")


def _scaling_style(shift, stage, diversity):
    if stage not in ("train", "development", "final") or diversity not in ("narrow", "broad"):
        raise ValueError("Scaling requires a supported stage and narrow/broad diversity")
    if shift in ("wording", "combined"):
        if stage == "train":
            raise ValueError("Wording/combined require explicit development or final stage")
        return stage
    return diversity


def scaling_sentence(rng, event, a, b=None, amount=0, stage="broad", unseen=False,
                     family_index=None, diversity="broad"):
    """Construction-only rendering; lexical draws precede the family draw.

    A separate RNG for each clause makes corresponding narrow/broad examples
    share lexical draws even though their family sets have different sizes.
    """
    if stage == "train":
        stage = diversity
    templates = SCALING_FAMILIES[stage][event]
    fields = _interpretation_fields(rng, unseen)
    index = rng.randrange(len(templates)) if family_index is None else family_index
    return templates[index].format(a=a, b=b, n=amount, **fields)


def _expansion_style(protocol, shift, stage, diversity):
    """Select the renderer and grammar family for query/contrast expansion."""
    if protocol == "interpretation":
        return _interpretation_style(shift, stage), interpretation_sentence, INTERPRETATION_FAMILIES
    if protocol == "scaling":
        return _scaling_style(shift, stage, diversity), scaling_sentence, SCALING_FAMILIES
    if protocol == "binding":
        return _binding_style(shift, stage), binding_sentence, BINDING_FAMILIES
    raise ValueError("Unsupported expansion protocol")


def _scaling_latent(rng, shift, task, wanted_yes):
    """Sample facts/query independently of surface grammar and its RNG use."""
    names = rng.sample(INTERPRETATION_NAMES if shift in ("names", "combined") else TRAIN_NAMES, 4)
    rows = []
    if task == "accounting":
        wide = shift in ("numbers", "combined")
        balances = {name: rng.choice(_STABILITY_INITIAL_VALUES) if wide else rng.randint(2, 9)
                    for name in names}
        for name in rng.sample(names, 4):
            rows.append((INITIAL, name, None, balances[name], 1))
        count = rng.randint(8, 16) if shift in ("length", "combined") else rng.randint(2, 4)
        for _ in range(count):
            a, b = rng.sample(names, 2)
            amount = rng.choice(_STABILITY_TRANSFER_VALUES) if wide else rng.randint(1, 4)
            active = rng.randrange(2)
            rows.append((TRANSFER, a, b, amount, active))
            if active:
                balances[a] -= amount
                balances[b] += amount
        query = rng.choice(names)
        rows.append((QUERY_BALANCE, query, None, 0, 1))
        return rows, str(balances[query])
    order = rng.sample(names, 4)
    facts = list(zip(order, order[1:]))
    if shift in ("length", "combined"):
        facts += rng.choices(facts, k=rng.randint(5, 13))
    rng.shuffle(facts)
    rows.extend((BEFORE, a, b, 0, 1) for a, b in facts)
    distance = 3 if shift in ("composition", "combined") else rng.randint(1, 2)
    start = rng.randrange(4 - distance)
    a, b = order[start], order[start + distance]
    if not wanted_yes:
        a, b = b, a
    rows.append((QUERY_BEFORE, a, b, 0, 1))
    return rows, "yes" if wanted_yes else "no"


def _scaling_render(rows, answer, task, seed, family, unseen, renderer=scaling_sentence):
    """Render fixed semantic rows without changing their answer or identities."""
    rng, sentences = random.Random(seed), []
    events = {INITIAL: "initial", TRANSFER: "transfer", BEFORE: "before",
              QUERY_BALANCE: "balance_query", QUERY_BEFORE: "before_query"}
    for kind, a, b, amount, active in rows:
        event = "inactive_transfer" if kind == TRANSFER and not active else events[kind]
        sentences.append(renderer(random.Random(rng.randrange(2**31)), event, a, b, amount,
                                  stage=family, unseen=unseen))
    return _finish(sentences, rows, answer, task)


def scaling_relation_capacity(stage="train", diversity="broad"):
    """Upper bound for unique three-edge presented worlds, before exclusions.

    Canonical name normalization leaves six fact orders, with one independent
    fact rendering per edge. Repeated-fact length/combined worlds are larger.
    """
    family = diversity if stage == "train" else stage
    forms = set()
    fields = _interpretation_fields(random.Random(0))
    for precede in INTERPRETATION_LEXICON["precede"]:
        for follow in INTERPRETATION_LEXICON["follow"]:
            fields.update(precede=precede[0], precede_base=precede[1],
                          follow=follow[0], follow_base=follow[1])
            forms.update(template.format(a="Alice", b="Bob", n=1, **fields)
                         for template in SCALING_FAMILIES[family]["before"])
    return 6 * len(forms) ** 3


def make_scaling_split(count: int, seed: int, shift: str = "iid", exclude: Iterable[str] = (),
                       stage: str = "train", diversity: str = "broad") -> list[Example]:
    """Paired diversity and nested data-amount protocol with fresh syntax.

    The first 6,000 narrow/broad examples share the same latent named events,
    query, amounts, polarity and per-clause lexical draws. Rejection checks both
    renderings, preventing grammar-specific rejection from breaking that match.
    Beyond this prefix only broad examples are supported. Output order is not
    shuffled, so a smaller request is an exact prefix of the larger request.
    Tasks alternate and relation labels alternate; the trainer should shuffle.

    Whole development/final arrangements are distinct from both training arms
    and all previous interpretation families. World separation and task bounds
    match interpretation_v1. Existing generators and fingerprints are unchanged.
    The caller must exclude the union of maximum training arms from every shared
    validation/test set, even for models trained on a smaller training prefix.
    """
    if count < 0 or shift not in SCALING_SHIFTS:
        raise ValueError("Nonnegative count and a supported scaling shift are required")
    family = _scaling_style(shift, stage, diversity)
    if family == "narrow" and count > SCALING_PAIRED_PREFIX:
        raise ValueError("Narrow scaling supports at most 6,000 paired examples")
    if shift not in ("length", "combined"):
        cap = scaling_relation_capacity(stage if family in ("development", "final") else "train", diversity)
        if count // 2 > cap:
            raise ValueError("Requested scaling split exceeds the unique relation-world capacity")
    excluded = set(exclude)
    rng, examples = random.Random(seed), []
    seen = {"narrow": set(excluded), "broad": set(excluded), "held": set(excluded)}
    excluded_worlds = {interpretation_world_key(text) for text in excluded}
    worlds = {key: set(excluded_worlds) for key in seen}
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("Could not generate enough unique scaling worlds after exclusions")
        index = len(examples)
        task = "accounting" if index % 2 == 0 else "relations"
        rows, answer = _scaling_latent(rng, shift, task, (index // 2) % 2 == 0)
        render_seed = rng.randrange(2**31)
        paired = family in ("narrow", "broad") and index < SCALING_PAIRED_PREFIX
        render_families = ("narrow", "broad") if paired else (family,)
        variants = {style: _scaling_render(rows, answer, task, render_seed, style,
                                          shift == "lexical_unseen") for style in render_families}
        if any(example.text in seen[style if paired or style == "broad" else "held"] or
               interpretation_world_key(example.text) in worlds[style if paired or style == "broad" else "held"]
               for style, example in variants.items()):
            continue
        for style, example in variants.items():
            key = style if paired or style == "broad" else "held"
            seen[key].add(example.text)
            worlds[key].add(interpretation_world_key(example.text))
        examples.append(variants[family])
    return examples


def scaling_role_distance_pairs():
    """Observed architecture diagnostic, excluded from promotion test families.

    Swapping distant by/to preserves every entity's +/-2 token window and the
    clause word multiset, but reverses transfer roles. A parser restricted to
    those features cannot distinguish both roles even with greater width.
    """
    sentences = ["Alice starts with 5 coins.", "Bob starts with 3 coins.",
                 "Clara starts with 4 coins.", "David starts with 2 coins."]
    rows = [(INITIAL, name, None, amount, 1)
            for name, amount in zip(("Alice", "Bob", "Clara", "David"), (5, 3, 4, 2))]
    variants = []
    for left, right, a, b, answer in (("by", "to", "Alice", "Bob", "3"),
                                    ("to", "by", "Bob", "Alice", "7")):
        transfer = f"{left} the one who is Alice 2 coins are given {right} the one who is Bob."
        variants.append(_finish(sentences + [transfer, "how many coins does Alice have now?"],
            rows + [(TRANSFER, a, b, 2, 1), (QUERY_BALANCE, "Alice", None, 0, 1)], answer, "accounting"))
    return [(*variants, "role_distance")]


# Binding protocol: stress who did what to whom across harder sentence forms

# Binding training absorbs observed scaling syntax; its new arrangements vary
# predicate order and the distance between role words and entity mentions.
# Conjoined give/receive predicates describe one transfer of the same coins,
# not two updates. Each clause has at most one literal and two name mentions.
BINDING_FAMILIES = {
    "train": {event: tuple(dict.fromkeys(template
        for stage in ("narrow", "broad", "development", "final")
        for template in SCALING_FAMILIES[stage][event]))
        for event in INTERPRETATION_FAMILIES["train"]},
    "development": {
        "initial": (
            "the one who initially {own} {n} coins is {a}.",
            "the starting balance of the one who is {a} is {n} coins.",
            "it is the one who is {a} who {own} {n} coins at the start.",
            "the coins held initially by {a} number {n}."),
        "balance_query": (
            "what is the number of coins that the one who is {a} has now?",
            "how many coins belong now to the one who is {a}?",
            "what balance is held now by the one who is {a}?",
            "what is now the coin balance that {a} has?"),
        "before": (
            "before the one who is {b} comes {a} in the order.",
            "after the one who is {a} comes {b} in the order.",
            "it is the one who is {a} who comes before {b}.",
            "it is the one who is {b} who comes after {a}.",
            "{a} is the one who comes before the one who is {b}.",
            "{b} is the one who comes after the one who is {a}.",
            "the one who comes before the one who is {b} is {a}.",
            "the one who comes after the one who is {a} is {b}."),
        "before_query": (
            "before the one who is {b} does {a} come in the order?",
            "after the one who is {a} does {b} come in the order?",
            "is it the one who is {a} who comes before {b}?",
            "is it the one who is {b} who comes after {a}?",
            "is {a} the one who comes before the one who is {b}?",
            "is {b} the one who comes after the one who is {a}?",
            "is the one who comes before the one who is {b} {a}?",
            "is the one who comes after the one who is {a} {b}?"),
    },
    "final": {
        "initial": (
            "it is {a} who is the one with {n} coins at the start.",
            "the number of coins that the one who is {a} initially {own} is {n}.",
            "what the one who is {a} has at the start is {n} coins.",
            "the one who is {a} initially has the {n} coins in possession."),
        "balance_query": (
            "the number of coins held by the one who is {a} now is what?",
            "the one who is {a} now has what number of coins?",
            "what number of coins is now held by the one who is {a}?",
            "the coins now in the possession of the one who is {a} number how many?"),
        "before": (
            "before the one who is {b} comes the one who is {a}.",
            "after the one who is {a} comes the one who is {b}.",
            "it is before the one who is {b} that the one who is {a} comes.",
            "it is after the one who is {a} that the one who is {b} comes.",
            "the one who is {a} is the one who comes before {b}.",
            "the one who is {b} is the one who comes after {a}.",
            "{a} is the one before the one who is {b} in the order.",
            "{b} is the one after the one who is {a} in the order."),
        "before_query": (
            "before the one who is {b} does the one who is {a} come?",
            "after the one who is {a} does the one who is {b} come?",
            "is it before the one who is {b} that the one who is {a} comes?",
            "is it after the one who is {a} that the one who is {b} comes?",
            "is the one who is {a} the one who comes before {b}?",
            "is the one who is {b} the one who comes after {a}?",
            "is {a} the one before the one who is {b} in the order?",
            "is {b} the one after the one who is {a} in the order?"),
    },
}
_BINDING_TRAIN_ADDITIONS = {
    "initial": (
        "{a} is the one with {n} coins at the start.",
        "the one with {n} coins initially is {a}.",
        "initially {n} coins are held by the one who is {a}.",
        "the starting balance held by the one who is {a} is {n} coins.",
        "it is {a} who initially {own} the {n} coins.",
        "at the start {a} has the {n} coins in possession."),
    "balance_query": (
        "what is the balance of the one who is {a} now?",
        "the one who is {a} has how many coins now?",
        "how many coins are held now by the one who is {a}?",
        "what number of coins does {a} have now?",
        "what is the number of coins held now by {a}?",
        "what is the balance that the one who is {a} has now?"),
    "before": (
        "the one who is {a} is before the one who is {b}.",
        "the one who is {b} is after the one who is {a}.",
        "{a} {precede} the one who is {b}.", "the one who is {a} {precede} {b}.",
        "{b} {follow} the one who is {a}.", "the one who is {b} {follow} {a}.",
        "it is {a} who is before the one who is {b}.",
        "it is {b} who is after the one who is {a}.",
        "before {b} comes the one who is {a}.",
        "after {a} comes the one who is {b}.",
        "the one who comes before {b} is the one who is {a}.",
        "the one who comes after {a} is the one who is {b}."),
    "before_query": (
        "is the one who is {a} before the one who is {b}?",
        "is the one who is {b} after the one who is {a}?",
        "does {a} {precede_base} the one who is {b}?",
        "does the one who is {a} {precede_base} {b}?",
        "does {b} {follow_base} the one who is {a}?",
        "does the one who is {b} {follow_base} {a}?",
        "is it {a} who is before the one who is {b}?",
        "is it {b} who is after the one who is {a}?",
        "before {b} does the one who is {a} come?",
        "after {a} does the one who is {b} come?",
        "is the one who comes before {b} the one who is {a}?",
        "is the one who comes after {a} the one who is {b}?"),
}
_BINDING_TRANSFERS = {
    "train": (
        "{a} {give} {n} coins and {b} {receive} the coins.",
        "{b} {receive} {n} coins and {a} {give} the coins.",
        "{a} is the one who {give} {n} coins and {b} is the one who {receive} the coins.",
        "{b} is the one who {receive} {n} coins and {a} is the one who {give} the coins.",
        "the one who {give} {n} coins is {a} and the one who {receive} the coins is {b}.",
        "the one who {receive} {n} coins is {b} and the one who {give} the coins is {a}.",
        "{n} coins are {given} by {a} and are {given} to {b}.",
        "{n} coins are {given} to {b} and are {given} by {a}.",
        "the coins that {a} {give} are the {n} coins that {b} {receive}.",
        "the coins that {b} {receive} are the {n} coins that {a} {give}.",
        "the one who is {a} {give} {n} coins to {b}.",
        "{a} {give} {n} coins to the one who is {b}.",
        "the one who is {b} {receive} {n} coins from {a}.",
        "{b} {receive} {n} coins from the one who is {a}.",
        "{n} coins are {given} by {a} to the one who is {b}.",
        "{n} coins are {given} to the one who is {b} by {a}.",
        "it is by {a} that {n} coins are {given} to {b}.",
        "it is to the one who is {b} that {a} {give} {n} coins.",
        "by the one who is {a} {n} coins are {given} to {b}.",
        "to the one who is {b} {n} coins are {given} by {a}.",
        "the one who {give} the {n} coins is {a} and {b} {receive} the coins.",
        "{b} {receive} the {n} coins and the one who {give} the coins is {a}."),
    "development": (
        "it is {a} who {give} the coins and {b} who {receive} {n} coins.",
        "{b} is the one who {receive} the {n} coins that {a} {give}.",
        "the {n} coins {given} by the one who is {a} are {given} to {b}.",
        "it is from the one who is {a} that the one who is {b} {receive} {n} coins.",
        "the one who {give} the coins is {a} and the one who {receive} {n} coins is {b}.",
        "it is {n} coins that the one who is {a} {give} to the one who is {b}."),
    "final": (
        "the coins that the one who is {b} {receive} are {given} by {a} and number {n}.",
        "it is the one who is {a} who {give} {n} coins and the one who is {b} who {receive} the coins.",
        "the one who {receive} the coins is {b} and {a} is the one who {give} {n} coins.",
        "the one who {give} the coins is {a} and {b} is the one who {receive} {n} coins.",
        "what the one who is {b} {receive} from {a} is {n} coins.",
        "the one who is {b} {receive} the {n} coins that are {given} by the one who is {a}."),
}
for _event, _templates in _BINDING_TRAIN_ADDITIONS.items():
    BINDING_FAMILIES["train"][_event] += _templates
for _stage, _templates in _BINDING_TRANSFERS.items():
    # Positive/negative complete families stay aligned for construction-time
    # polarity contrasts. This string rewriting never runs at model inference.
    _negative = tuple(template.replace("{give}", "does not {give_base}")
        .replace("{receive}", "does not {receive_base}").replace("{given}", "not {given}")
        for template in _templates)
    if _stage == "train":
        BINDING_FAMILIES[_stage]["transfer"] += _templates
        BINDING_FAMILIES[_stage]["inactive_transfer"] += _negative
    else:
        BINDING_FAMILIES[_stage]["transfer"] = _templates
        BINDING_FAMILIES[_stage]["inactive_transfer"] = _negative


def _binding_quantity_mirror(template):
    """Swap complete object phrases, preserving a leading definite article."""
    if " and " not in template or "the coins" not in template or "{n} coins" not in template:
        return None
    phrase = "the {n} coins" if "the {n} coins" in template else "{n} coins"
    return template.replace(phrase, "__AMOUNT__").replace("the coins", phrase).replace("__AMOUNT__", "the coins")


# Only training arrangements can be mirrored. In particular, one giver-first
# postposed-subject mirror is already reserved for development and is excluded.
# Full template strings make the augmentation auditable without stable indices.
BINDING_QUANTITY_VARIANTS = {}
for _event in ("transfer", "inactive_transfer"):
    _reserved = set(BINDING_FAMILIES["development"][_event]) | set(BINDING_FAMILIES["final"][_event])
    for _template in BINDING_FAMILIES["train"][_event]:
        _mirror = _binding_quantity_mirror(_template)
        if _mirror is not None and _mirror not in _reserved:
            BINDING_QUANTITY_VARIANTS[_template] = _mirror


def _binding_style(shift, stage):
    if stage not in BINDING_FAMILIES:
        raise ValueError("Binding stage must be train, development, or final")
    if shift in ("wording", "combined"):
        if stage == "train":
            raise ValueError("Wording/combined require explicit development or final stage")
        return stage
    return "train"


def binding_sentence(rng, event, a, b=None, amount=0, stage="train", unseen=False,
                     family_index=None, balance_quantities=False):
    """Render training/evaluation data only, without exposing grammar labels."""
    templates = BINDING_FAMILIES[stage][event]
    fields = _interpretation_fields(rng, unseen)
    index = rng.randrange(len(templates)) if family_index is None else family_index
    template = templates[index]
    if balance_quantities and stage == "train" and template in BINDING_QUANTITY_VARIANTS:
        # All original lexical/family draws precede this extra draw. The caller
        # independently seeds each clause, so later clauses remain unchanged.
        if rng.randrange(2):
            template = BINDING_QUANTITY_VARIANTS[template]
    return template.format(a=a, b=b, n=amount, **fields)


def binding_relation_capacity(stage="train"):
    """Three-edge presented-world upper bound before explicit exclusions."""
    forms, fields = set(), _interpretation_fields(random.Random(0))
    for precede in INTERPRETATION_LEXICON["precede"]:
        for follow in INTERPRETATION_LEXICON["follow"]:
            fields.update(precede=precede[0], precede_base=precede[1],
                          follow=follow[0], follow_base=follow[1])
            forms.update(template.format(a="Alice", b="Bob", n=1, **fields)
                         for template in BINDING_FAMILIES[stage]["before"])
    return 6 * len(forms) ** 3


def make_binding_split(count: int, seed: int, shift: str = "iid", exclude: Iterable[str] = (),
                       stage: str = "train", balance_quantities: bool = False) -> list[Example]:
    """Fresh binding arrangements with exactly the existing semantic contract.

    Training includes observed language and new variations in predicate order,
    argument distance and conjoined transfer descriptions. Development/final
    combine those ingredients into separate complete arrangements. The new
    conjunction word is covered in training; lexical_unseen remains separate.
    All labels are generated from independent latent facts, never inferred by
    matching the resulting prose. Text inference receives no family/row labels.

    Tasks and relation labels alternate; transfer polarity is equiprobable.
    Counts nest by prefix; ordinary trainer shuffling avoids ordered batches.
    Exclusions remove normalized presented worlds regardless of final question.
    Optional quantity balancing changes only object placement in seven paired
    conjunction arrangements and their negatives; held-out wording is unchanged.
    Default rendering retains every original binding draw and fingerprint.
    """
    if count < 0 or shift not in BINDING_SHIFTS:
        raise ValueError("Nonnegative count and a supported binding shift are required")
    family = _binding_style(shift, stage)
    if shift not in ("length", "combined") and count // 2 > binding_relation_capacity(family):
        raise ValueError("Requested binding split exceeds the unique relation-world capacity")
    seen = set(exclude)
    worlds = {interpretation_world_key(text) for text in seen}
    rng, examples, attempts = random.Random(seed), [], 0
    def render(*args, **kwargs):
        return binding_sentence(*args, **kwargs, balance_quantities=balance_quantities)
    while len(examples) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("Could not generate enough unique binding worlds after exclusions")
        index = len(examples)
        task = "accounting" if index % 2 == 0 else "relations"
        rows, answer = _scaling_latent(rng, shift, task, (index // 2) % 2 == 0)
        example = _scaling_render(rows, answer, task, rng.randrange(2**31), family,
                                  shift == "lexical_unseen", renderer=render)
        key = interpretation_world_key(example.text)
        if example.text in seen or key in worlds:
            continue
        seen.add(example.text)
        worlds.add(key)
        examples.append(example)
    return examples


def binding_paraphrase_examples(examples, seed=0):
    """Training-only equivalent programs in a second, independently drawn form.

    Returned examples must stay in the same partition as their source worlds.
    Named facts and answers are unchanged; visible first-mention IDs may change.
    Callers choose whether to include these and must report the extra exposure.
    """
    rng, result = random.Random(seed), []
    for example in examples:
        names = surface_names(example.text)
        rows = [(kind, names[a], names[b] if b >= 0 else None, value, active)
                for kind, a, b, value, active in example.rows]
        result.append(_scaling_render(rows, example.answer, example.task, rng.randrange(2**31),
                                      "train", False, renderer=binding_sentence))
    return result


# Diagnostic metadata for role and polarity experiments

@dataclass(frozen=True)
class TransferProvenance:
    """Construction/audit metadata, never supplied as model input.

    Character spans and zero-based token positions refer to original_sentence,
    excluding the model's CLS token. Only visible lexical fields are retained;
    irrelevant random draws cannot be recovered and do not affect either form.
    source_family is a canonical matching renderer; equivalent_families records
    aliases when the original generator's identity cannot be uniquely recovered.
    """
    origin_key: tuple[str, ...]
    clause_index: int
    source_family: str
    equivalent_families: tuple[str, ...]
    positive_template: str
    negative_template: str
    lexical_fields: tuple[tuple[str, str], ...]
    sender: str
    recipient: str
    amount: int
    original_active: int
    original_sentence: str
    entity_spans: tuple[tuple[str, int, int], ...]
    entity_positions: tuple[tuple[str, int], ...]
    predicate_positions: tuple[tuple[str, int], ...]

    def render(self, active: int) -> str:
        if type(active) is not int or active not in (0, 1):
            raise ValueError("Polarity must be integer zero or one")
        template = self.positive_template if active else self.negative_template
        return template.format(a=self.sender, b=self.recipient, n=self.amount,
                               **dict(self.lexical_fields))


@dataclass(frozen=True)
class TransferSourceFrame:
    """Semantic event labels kept separate from ordinary model inputs.

    Both named roles apply at every predicate, including implied arguments;
    these are not direct grammatical attachments. Token positions are local
    to sentence, zero-based, and exclude the model's CLS token.
    """
    clause_index: int
    origin_key: tuple[str, ...]
    source_family: str
    sentence: str
    predicate_positions: tuple[int, ...]
    sender_position: int
    recipient_position: int


@dataclass(frozen=True)
class PolarityGroup:
    """Four views of one original transfer, retaining the parent's split key."""
    origin_key: tuple[str, ...]
    clause_index: int
    provenance: TransferProvenance
    # Sender inactive/active, followed by recipient inactive/active.
    examples: tuple[Example, Example, Example, Example]


def _polarity_families(stage):
    if stage not in BINDING_FAMILIES:
        raise ValueError("Polarity reconstruction requires train, development, or final")
    families = []
    for selected in dict.fromkeys((stage, "train")):
        positive, negative = (BINDING_FAMILIES[selected][event]
                              for event in ("transfer", "inactive_transfer"))
        if len(positive) != len(negative):
            raise ValueError("Polarity template families must remain aligned")
        for index, (yes, no) in enumerate(zip(positive, negative)):
            families.append((f"binding/{selected}/{index}", yes, no))
            if selected == "train" and yes in BINDING_QUANTITY_VARIANTS:
                if no not in BINDING_QUANTITY_VARIANTS:
                    raise ValueError("Quantity mirrors must preserve both polarities")
                families.append((f"binding/train/{index}/quantity_mirror",
                                 BINDING_QUANTITY_VARIANTS[yes], BINDING_QUANTITY_VARIANTS[no]))
    # Retained examples use the original generator's five explicit pairs,
    # including 'never hands' and 'no movement ... occurs'. No negation removal.
    retained = set()
    for wording in (False, True):
        for index in range(12):
            pair = tuple(_transfer("{a}", "{b}", "{n}", active, index, wording)
                         for active in (1, 0))
            if pair not in retained:
                families.append((f"retention/{int(wording)}/{index}", *pair))
                retained.add(pair)
    return tuple(families)


def _polarity_lexical_fields():
    # Transfer templates only vary giving/receiving inflections. Include the
    # separately reported unseen lexicon so diagnostic coverage is explicit.
    result = []
    for lexicon in (INTERPRETATION_LEXICON, _INTERPRETATION_UNSEEN):
        for give, receive in itertools.product(lexicon["give"], lexicon["receive"]):
            result.append({"give": give[0], "give_base": give[1], "given": give[2],
                           "receive": receive[0], "receive_base": receive[1]})
    return result


@lru_cache(maxsize=3)
def _polarity_lookup(stage):
    """Canonical sentence lookup used only for annotated-data reconstruction."""
    lookup = {}
    formatter = string.Formatter()
    for family, positive, negative in _polarity_families(stage):
        used = {field for template in (positive, negative)
                for _, field, _, _ in formatter.parse(template) if field and field not in ("a", "b", "n")}
        for draw in _polarity_lexical_fields():
            fields = tuple(sorted((key, draw[key]) for key in used))
            rendered = tuple(template.format(a="SENDER", b="RECIPIENT", n="QUANTITY", **dict(fields))
                             for template in (negative, positive))
            record = (family, positive, negative, fields, rendered)
            for active, sentence in enumerate(rendered):
                lookup.setdefault((active, sentence), []).append(record)
    return lookup


_PROVENANCE_TOKENS = re.compile(r"(?<![A-Za-z0-9])-?\d+\b|[A-Za-z]+|\S")
_PROVENANCE_PREDICATES = frozenset(
    word for lexicon in (INTERPRETATION_LEXICON, _INTERPRETATION_UNSEEN)
    for category in ("give", "receive") for forms in lexicon[category] for word in forms
) | {"move", "moves", "movement"}


def transfer_provenance(example: Example, clause_index: int, stage="train") -> TransferProvenance:
    """Recover an exact supported clause and its paired polarity rendering.

    Oracle rows are used only to reconstruct labeled data. They never enter a
    learned parser. Unsupported text or ambiguous partners raise rather than
    dropping examples or guessing a new sentence family.
    """
    if example.task != "accounting" or " ".join(example.sentences) != example.text:
        raise ValueError("Polarity reconstruction requires an intact accounting example")
    if not 0 <= clause_index < len(example.rows) or len(example.rows) != len(example.sentences):
        raise ValueError("Polarity reconstruction requires a valid clause index")
    kind, a, b, amount, active = example.rows[clause_index]
    names = surface_names(example.text)
    if kind != TRANSFER or active not in (0, 1) or a == b or a not in range(len(names)) or b not in range(len(names)):
        raise ValueError("Polarity reconstruction requires a valid transfer row")
    original = example.sentences[clause_index]
    visible = NAME_PATTERN.findall(original)
    if sorted(visible) != sorted((names[a], names[b])):
        raise ValueError("A transfer must contain each of its two named participants once")
    numeric = list(re.finditer(r"(?<![A-Za-z0-9])-?\d+\b", original))
    if len(numeric) != 1 or int(numeric[0].group()) != amount:
        raise ValueError("A transfer must contain exactly its labeled integer amount")
    canonical = NAME_PATTERN.sub(lambda match: "SENDER" if match.group() == names[a] else "RECIPIENT", original)
    canonical = re.sub(r"(?<![A-Za-z0-9])-?\d+\b", "QUANTITY", canonical)
    matches = _polarity_lookup(stage).get((active, canonical), ())
    if not matches:
        raise ValueError("Unsupported exact transfer surface for polarity reconstruction")
    # Distinct template aliases or unused lexical draws are harmless only if
    # both complete polarity surfaces agree exactly.
    if len({record[4] for record in matches}) != 1:
        raise ValueError("Ambiguous exact transfer surface has conflicting polarity partners")
    family, positive, negative, fields, _ = matches[0]
    tokens = list(_PROVENANCE_TOKENS.finditer(original))
    provenance = TransferProvenance(
        interpretation_world_key(example.text), clause_index, family,
        tuple(dict.fromkeys(record[0] for record in matches)), positive, negative, fields,
        names[a], names[b], amount, active, original,
        tuple((match.group(), match.start(), match.end()) for match in NAME_PATTERN.finditer(original)),
        tuple((match.group(), index) for index, match in enumerate(tokens) if NAME_PATTERN.fullmatch(match.group())),
        tuple((match.group(), index) for index, match in enumerate(tokens) if match.group() in _PROVENANCE_PREDICATES))
    if provenance.render(active) != original:
        raise ValueError("Polarity provenance failed byte-exact original reconstruction")
    return provenance


def transfer_source_frame(example: Example, clause_index: int, stage="train") -> TransferSourceFrame:
    """Label each exact-view predicate with BOTH semantic event arguments.

    All supported predicates in a transfer clause describe the same event.
    Every predicate therefore receives sender and recipient links, including
    implied arguments with no direct grammatical attachment. This annotation
    is for training supervision or explicitly labeled oracle diagnostics;
    it must never be supplied to ordinary text inference. Training callers
    must use stage='train'; other stages only authorize diagnostic rendering.
    """
    provenance = transfer_provenance(example, clause_index, stage=stage)
    tokens = tuple(match.group() for match in _PROVENANCE_TOKENS.finditer(provenance.original_sentence))
    positions = {}
    for role, name in (("sender", provenance.sender), ("recipient", provenance.recipient)):
        matches = tuple(position for entity, position in provenance.entity_positions if entity == name)
        if len(matches) != 1 or not 0 <= matches[0] < len(tokens) or tokens[matches[0]] != name:
            raise ValueError(f"Source frame requires exactly one supported {role} mention")
        positions[role] = matches[0]
    predicates = tuple(position for _, position in provenance.predicate_positions)
    if (not predicates or tuple(sorted(set(predicates))) != predicates or
            any(not 0 <= position < len(tokens) or tokens[position] != word
                for word, position in provenance.predicate_positions)):
        raise ValueError("Source frame requires nonempty exact predicate positions")
    return TransferSourceFrame(clause_index, provenance.origin_key, provenance.source_family,
                               provenance.original_sentence, predicates,
                               positions["sender"], positions["recipient"])


def polarity_world_key(example: Example, stage="train") -> tuple[str, ...]:
    """Group worlds differing only in transfer polarity or their final query.

    Every transfer is rendered active using its exact original family/lexical
    choices. This is stronger than retaining an original parent's metadata key:
    independently generated counterpart worlds also receive the same key.
    Callers must audit split overlap before augmentation, never silently discard
    or regenerate existing held-out parents when a collision is discovered.
    """
    if stage not in BINDING_FAMILIES or " ".join(example.sentences) != example.text:
        raise ValueError("Polarity grouping requires an intact example and supported stage")
    if len(example.rows) != len(example.sentences):
        raise ValueError("Polarity grouping requires aligned rows and sentences")
    sentences = list(example.sentences)
    for index, row in enumerate(example.rows):
        if row[0] == TRANSFER:
            sentences[index] = transfer_provenance(example, index, stage).render(1)
    text = " ".join(sentences)
    if surface_names(text) != surface_names(example.text):
        raise ValueError("Polarity grouping changed visible identity order")
    return interpretation_world_key(text)


def polarity_pair(example: Example, clause_index: int, stage="train") -> tuple[Example, Example]:
    """Return inactive/active views, preserving the original question and facts.

    Zero amounts are valid here: role consistency remains meaningful even when
    activity cannot change an answer. Neither family nor lexical choices redraw.
    """
    provenance = transfer_provenance(example, clause_index, stage)
    if example.rows[-1][0] != QUERY_BALANCE:
        raise ValueError("Accounting polarity pairs require a final balance query")
    variants = []
    for active in (0, 1):
        sentences, rows = list(example.sentences), list(example.rows)
        sentences[clause_index] = provenance.render(active)
        rows[clause_index] = (*rows[clause_index][:-1], active)
        text, rows, sentences = " ".join(sentences), tuple(rows), tuple(sentences)
        if surface_names(text) != surface_names(example.text):
            raise ValueError("A polarity partner changed visible identity order")
        variants.append(Example(text, sentences, rows, _interpretation_answer(rows), example.task))
    return tuple(variants)


def direct_polarity_groups(examples, stage="train", include_zero=False):
    """Probe every original nonzero transfer at both participants.

    Returns (groups, coverage). Reconstruction is strict; unsupported/ambiguous
    examples raise instead of creating hidden drops. Zero amounts are counted
    and skipped by default because a balance-change probe would be uninformative.
    include_zero retains them for explicit role-fidelity checks with equal answers.
    All four views retain their original world's split key in separate metadata.
    """
    groups = []
    coverage = {"parents": 0, "accounting_parents": 0, "other_parents": 0,
                "accounting_without_transfers": 0, "accounting_without_nonzero_transfers": 0,
                "transfer_clauses": 0, "supported_clauses": 0, "zero_amount_skipped": 0,
                "zero_amount_groups": 0,
                "groups": 0, "views": 0, "unsupported_clauses": 0, "ambiguous_clauses": 0}
    for example in examples:
        coverage["parents"] += 1
        if example.task != "accounting":
            coverage["other_parents"] += 1
            continue
        coverage["accounting_parents"] += 1
        transfer_rows = [row for row in example.rows if row[0] == TRANSFER]
        coverage["accounting_without_transfers"] += int(not transfer_rows)
        coverage["accounting_without_nonzero_transfers"] += int(not any(row[3] for row in transfer_rows))
        query_names = surface_names(example.sentences[-1])
        if len(query_names) != 1 or example.rows[-1][0] != QUERY_BALANCE:
            raise ValueError("Direct polarity queries require one visible query participant")
        names = surface_names(example.text)
        for index, row in enumerate(example.rows):
            if row[0] != TRANSFER:
                continue
            coverage["transfer_clauses"] += 1
            provenance = transfer_provenance(example, index, stage)
            coverage["supported_clauses"] += 1
            if row[3] == 0:
                if not include_zero:
                    coverage["zero_amount_skipped"] += 1
                    continue
                coverage["zero_amount_groups"] += 1
            inactive, active = polarity_pair(example, index, stage)
            views = []
            for participant in row[1:3]:
                for variant in (inactive, active):
                    query = NAME_PATTERN.sub(lambda _: names[participant], example.sentences[-1])
                    sentences = variant.sentences[:-1] + (query,)
                    rows = variant.rows[:-1] + ((QUERY_BALANCE, participant, -1, 0, 1),)
                    views.append(Example(" ".join(sentences), sentences, rows, _interpretation_answer(rows), "accounting"))
                if row[3] != 0 and views[-2].answer == views[-1].answer:
                    raise ValueError("A nonzero polarity intervention must change its participant's balance")
                if row[3] == 0 and views[-2].answer != views[-1].answer:
                    raise ValueError("A zero-amount polarity intervention must preserve its participant's balance")
            groups.append(PolarityGroup(provenance.origin_key, index, provenance, tuple(views)))
    coverage["groups"], coverage["views"] = len(groups), 4 * len(groups)
    return groups, coverage


# Exact reference execution and reproducible split identity

def execute(rows: Iterable[Row]) -> str:
    """Exact semantic executor, a separately reported diagnostic reference.

    This is not a neural reasoning result and must not be called by learned
    answer heads. It applies balance updates or builds an ordering graph, then
    answers the one final query. Malformed rows raise instead of silently
    inventing semantics.
    """
    balances: dict[int, int] = {}
    edges: dict[int, set[int]] = {}
    query = None
    for kind, a, b, value, active in rows:
        # Validate each row before allowing it to change the reference state.
        if kind == PAD:
            continue
        if kind not in range(INITIAL, QUERY_BEFORE + 1) or a not in range(4):
            raise ValueError("Invalid event type or first entity")
        if active not in (0, 1):
            raise ValueError("Invalid activity flag")
        if kind in (TRANSFER, BEFORE, QUERY_BEFORE) and (b not in range(4) or b == a):
            raise ValueError("Invalid second entity")
        if kind == INITIAL:
            if a in balances:
                raise ValueError("Duplicate initial balance")
            balances[a] = int(value)
        elif kind == TRANSFER and active:
            # An active transfer debits the sender and credits the recipient.
            if a not in balances or b not in balances:
                raise ValueError("Transfer without initial balance")
            balances[a] -= int(value)
            balances[b] += int(value)
        elif kind == BEFORE and active:
            # Directed edges encode the visible "a is before b" facts.
            edges.setdefault(a, set()).add(b)
        elif kind in (QUERY_BALANCE, QUERY_BEFORE):
            if query is not None:
                raise ValueError("Multiple queries")
            query = (kind, a, b)
    if query is None:
        raise ValueError("Missing query")
    kind, a, b = query
    if kind == QUERY_BALANCE:
        if a not in balances:
            raise ValueError("Query without initial balance")
        return str(balances[a])
    # Follow ordering edges transitively; reaching b makes the answer "yes".
    visited, pending = set(), list(edges.get(a, ()))
    while pending:
        node = pending.pop()
        if node == b:
            return "yes"
        if node not in visited:
            visited.add(node)
            pending.extend(edges.get(node, ()))
    return "no"


def split_fingerprint(examples: Iterable[Example]) -> str:
    """Return a stable SHA-256 ID for an ordered set of prompts and answers.

    The separators make field and example boundaries unambiguous. Callers use
    this digest to confirm that a named split has not silently changed.
    """
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.answer.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
