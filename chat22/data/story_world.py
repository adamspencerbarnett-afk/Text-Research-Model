"""Generated story worlds (16 October, Adam: story training starts in v2.2). Every fact is known, so consistency is
checkable. A world has a story bible (3-6 characters: name, role, trait, home, possession, companion; places) and an
outline of scenes; each scene states some facts and moves the plot. Three kinds of training windows come from it:

  scene      "Continue the story." -> the next scene, with the bible facts that scene uses in the memory block
  question   a question about an earlier scene (who / where / what), answered from memory
  repair     a sentence that contradicts the bible -> the corrected sentence

Names use the training split of the skills generator (G3.setup(test=False) keeps the reserved test names out); the
held-out story test uses the reserved names."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import gen_skills as G  # noqa: E402

ROLES = ["baker", "sailor", "teacher", "detective", "farmer", "painter", "doctor", "carpenter", "musician", "pilot",
         "librarian", "gardener", "inventor", "fisher", "tailor"]
TRAITS = ["brave", "shy", "curious", "stubborn", "kind", "clever", "cheerful", "careful", "restless", "patient"]
THINGS = ["a brass compass", "a red umbrella", "an old map", "a silver key", "a wooden flute", "a green notebook",
          "a lantern", "a pocket watch", "a blue scarf", "a small telescope"]
PETS = ["a grey cat", "a loyal dog", "a talking parrot", "a lazy tortoise", "a white goat", "a clever fox"]
PLACES = ["the harbour", "the old mill", "the market square", "the lighthouse", "the forest path", "the bakery",
          "the town hall", "the river bridge", "the library", "the hill farm", "the train station", "the orchard"]
EVENTS = ["found a letter that had no name on it", "heard a bell ringing where no bell should be", "lost the way in thick fog",
          "met a stranger who knew their name", "saw a light moving on the water", "discovered a door behind the shelves",
          "followed footprints in the snow", "repaired a broken cart", "won a small contest", "argued about the map"]


class World:
    def __init__(self, rng, n_chars=None):
        self.rng = rng
        names = list(rng.choice(G.NAMES, int(n_chars or rng.integers(3, 7)), replace=False))
        homes = [G.pick(rng, G.CITIES) for _ in names]
        self.chars = [{"name": str(n), "role": G.pick(rng, ROLES), "trait": G.pick(rng, TRAITS), "home": h,
                       "thing": G.pick(rng, THINGS), "pet": G.pick(rng, PETS)} for n, h in zip(names, homes)]
        self.scenes = []                       # (text, facts used [(char index, field)], place, event)
        self.state = {}                        # (char, "place") -> where they were last

    def bible_facts(self, c: int) -> list[str]:
        ch = self.chars[c]
        return [f"{ch['name']} is a {ch['role']} from {ch['home']}.", f"{ch['name']} is {ch['trait']}.",
                f"{ch['name']} always carries {ch['thing']}.", f"{ch['name']} has {ch['pet']}."]

    def fact(self, c: int, field: str) -> str:
        ch = self.chars[c]
        return {"role": f"{ch['name']} is a {ch['role']}.", "home": f"{ch['name']} comes from {ch['home']}.",
                "trait": f"{ch['name']} is {ch['trait']}.", "thing": f"{ch['name']} always carries {ch['thing']}.",
                "pet": f"{ch['name']} has {ch['pet']}."}[field]

    def next_scene(self) -> tuple[str, list]:
        rng = self.rng
        k = int(rng.integers(1, min(3, len(self.chars)) + 1))
        cs = [int(c) for c in rng.choice(len(self.chars), k, replace=False)]
        place, event = G.pick(rng, PLACES), G.pick(rng, EVENTS)
        a = self.chars[cs[0]]
        used = [(cs[0], "trait"), (cs[0], "thing")]
        s = [f"{a['name']}, the {a['trait']} {a['role']}, went to {place} carrying {a['thing']}."]
        if len(cs) > 1:
            b = self.chars[cs[1]]
            s.append(f"There {a['name']} met {b['name']}, who had come all the way from {b['home']} with {b['pet']}.")
            used += [(cs[1], "home"), (cs[1], "pet")]
            s.append(f"Together they {event}.")
        else:
            s.append(f"At {place}, {a['name']} {event}.")
        if len(cs) > 2:
            c = self.chars[cs[2]]
            s.append(f"Later {c['name']} the {c['role']} joined them, as {c['trait']} as ever.")
            used += [(cs[2], "role"), (cs[2], "trait")]
        s.append(G.pick(rng, ["Nobody said much on the way back.", "It was late when they parted.", "The wind had changed by evening.",
                              "Something about the day stayed with them.", "They promised to meet again soon."]))
        for c in cs:
            self.state[(c, "place")] = place
        text = " ".join(s)
        self.scenes.append((text, used, place, event, cs))
        return text, used

    def question(self, j: int) -> tuple[str, str]:
        """A question about scene j, answered from what that scene says."""
        text, used, place, event, cs = self.scenes[j]
        a = self.chars[cs[0]]
        r = self.rng.random()
        if r < 0.4:
            return f"Where did {a['name']} go in the story?", f"{a['name']} went to {place}."
        if r < 0.7:
            return f"What does {a['name']} always carry?", f"{a['name']} always carries {a['thing']}."
        if len(cs) > 1:
            b = self.chars[cs[1]]
            return f"Where does {b['name']} come from?", f"{b['name']} comes from {b['home']}."
        return f"What kind of person is {a['name']}?", f"{a['name']} is {a['trait']}."

    def contradiction(self) -> tuple[str, str, int]:
        """A sentence that breaks the bible, and its correction."""
        c = int(self.rng.integers(len(self.chars))); ch = self.chars[c]
        wrong_home = G.pick(self.rng, [x for x in G.CITIES if x != ch["home"]])
        return (f"{ch['name']} comes from {wrong_home}.", f"{ch['name']} comes from {ch['home']}.", c)
