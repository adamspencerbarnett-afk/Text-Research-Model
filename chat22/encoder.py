"""The code dictionary (V0Dict, our own byte-phrase codes, about 3.6 characters per code) and the fixed text pieces of
the chat layout. Wraps compiler/experiments/ladder_encodings.py unchanged."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ladder_encodings as LE  # noqa: E402

NATIVE = str(ROOT / "data" / "cv0.exe")
EOS = 12                     # end of a reply (the form-feed byte; also the document separator in core text)
SEP_TEXT = b"\n\n"           # between the memory block and the message (one code)
MEM_HEAD = b"Memory:\n"      # heading of the memory block; its last code is the 'nothing here' position


def load(dict_path: str, native: str | None = NATIVE):
    return LE.V0Dict(str(dict_path), native)


def render(q: bytes, a: bytes | None) -> bytes:
    """One exchange as the model sees it; a=None leaves the reply open."""
    return b"User: " + q + b"\nAssistant:" + (b" " + a + b"\n\n" if a is not None else b"")


def parse_exchanges(text: bytes) -> list[tuple[bytes, bytes]]:
    """Exchanges are separated by a blank line followed by a new 'User: ' turn (answers may have paragraphs)."""
    out = []
    for block in re.split(rb"\n\n(?=User: )", text):
        m = re.match(rb"User: (.*?)\nAssistant: (.*)\Z", block, re.S)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out
