"""Download and prepare the public-domain book corpus used by the initial encoder tests.

    python compiler/experiments/prepare_books.py DATA_DIR

Clones GITenberg repositories (shallow), strips the Project Gutenberg header and footer,
normalises CRLF to LF, and writes DATA_DIR/train/*.txt, DATA_DIR/heldout/*.txt and
DATA_DIR/ood/*.txt (Markdown and Python source from this repository). Books that fail to
clone are skipped and listed. Text preparation is the only normalisation; the encoder
itself is lossless on any bytes.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

BOOKS = """A-Christmas-Carol_46 A-Study-in-Scarlet_244 A-Tale-of-Two-Cities_98 Adventures-of-Huckleberry-Finn_76
Alice-s-Adventures-in-Wonderland_11 Anna-Karenina_1399 Bleak-House_1023 Crime-and-Punishment_2554
David-Copperfield_766 Don-Quixote_996 Dracula_345 Dubliners_2814 Emma_158 Frankenstein_84 Great-Expectations_1400
Gulliver-s-Travels_829 Jane-Eyre_1260 Les-Mis-rables_135 Leviathan_3207 Little-Women_514 Mansfield-Park_141
Metamorphosis_5200 Middlemarch_145 Moby-Dick--Or-The-Whale_2701 Northanger-Abbey_121 Oliver-Twist_730 Persuasion_105
Peter-Pan_16 Pride-and-Prejudice_1342 Sense-and-Sensibility_161 Siddhartha_2500 The-Brothers-Karamazov_28054
The-Call-of-the-Wild_215 The-Count-of-Monte-Cristo_1184 The-Hound-of-the-Baskervilles_2852 The-Iliad_6130
The-Jungle-Book_236 The-Picture-of-Dorian-Gray_174 The-Republic_1497 The-Scarlet-Letter_33
The-Strange-Case-Of-Dr.-Jekyll-And-Mr.-Hyde_42 The-Time-Machine_35 The-War-of-the-Worlds_36
The-Wonderful-Wizard-of-Oz_55 Treasure-Island_120 Ulysses_4300 War-and-Peace_2600 Wuthering-Heights_768
The-Adventures-of-Sherlock-Holmes_1661 Heart-of-Darkness_219 The-Prince_1232""".split()
HELD_OUT = {"The-Adventures-of-Sherlock-Holmes_1661", "Heart-of-Darkness_219", "The-Prince_1232"}
REPO = Path(__file__).resolve().parents[2]


def clean(path: str) -> str:
    raw = open(path, "rb").read()
    text = raw.decode("latin-1") if path.endswith("-8.txt") else raw.decode("utf-8", errors="replace")
    text = text.lstrip("﻿").replace("\r\n", "\n")
    a = re.search(r"\*\*\* ?START OF[^\n]*\n", text)
    b = re.search(r"\n\*\*\* ?END OF", text)
    return text[a.end() if a else 0: b.start() if b else len(text)].strip("\n") + "\n"


def main(out: str) -> int:
    raw_dir = Path(out) / "raw"
    for split in ("train", "heldout", "ood", "raw"):
        (Path(out) / split).mkdir(parents=True, exist_ok=True)
    missing = []
    for name in BOOKS:
        target = raw_dir / name
        if not target.exists():
            r = subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/GITenberg/{name}", str(target)],
                               capture_output=True)
            if r.returncode:
                missing.append(name)
                continue
        texts = [f for f in glob.glob(str(target / "*.txt")) if os.path.getsize(f) > 20000]
        if not texts:
            missing.append(name)
            continue
        best = sorted(texts, key=lambda p: (not p.endswith("-0.txt"), not p.endswith("-8.txt"), -os.path.getsize(p)))[0]
        split = "heldout" if name in HELD_OUT else "train"
        (Path(out) / split / f"{name}.txt").write_text(clean(best), encoding="utf-8")
    md = sorted(glob.glob(str(REPO / "docs" / "*.md"))) + [str(REPO / n) for n in ("MASTER_PLAN.md", "MASTER_LOG.md", "README.md")]
    py = sorted(glob.glob(str(REPO / "scripts" / "*.py")))
    (Path(out) / "ood" / "repo_docs_markdown.txt").write_text("".join(Path(p).read_text(encoding="utf-8") + "\n" for p in md), encoding="utf-8")
    (Path(out) / "ood" / "repo_python_code.txt").write_text("".join(Path(p).read_text(encoding="utf-8") + "\n" for p in py), encoding="utf-8")
    print(f"prepared {len(BOOKS) - len(missing)} books; missing: {', '.join(missing) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
