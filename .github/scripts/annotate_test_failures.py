"""Turn unittest failures into GitHub Actions error annotations.

Reads the captured output of `python -m unittest -v` and prints one
`::error::` workflow command per failed or errored test, so failures are
visible on the run page and through the check-run annotations API without
downloading the full log.
"""

import re
import sys

MAX_ANNOTATIONS = 10  # GitHub shows at most 10 error annotations per step
MAX_LINES = 40        # Keep each annotation readable


def escape(text: str) -> str:
    # Workflow command data must escape %, CR and LF.
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text: str) -> str:
    # Property values (such as title=) must also escape ':' and ','.
    return escape(text).replace(":", "%3A").replace(",", "%2C")


def main(path: str) -> None:
    output = open(path, encoding="utf-8", errors="replace").read()
    summary = [line for line in output.splitlines()
               if re.match(r"^(Ran \d+ tests?|FAILED|OK)\b", line)]
    blocks = re.split(r"^={50,}$", output, flags=re.M)[1:]
    failures = []
    for block in blocks:
        lines = block.strip("\n").splitlines()
        if not lines or not re.match(r"^(FAIL|ERROR): ", lines[0]):
            continue
        body = [line for line in lines[1:] if not re.match(r"^-{50,}$", line)]
        # Drop the trailing run summary from the last block.
        while body and (not body[-1].strip() or re.match(r"^(Ran \d+|FAILED|OK)\b", body[-1])):
            body.pop()
        failures.append((lines[0], body[-MAX_LINES:]))

    if not failures:
        tail = output.splitlines()[-MAX_LINES:]
        print(f"::error title=Test run failed - no unittest failure blocks found::{escape(chr(10).join(tail))}")
        return

    title = " | ".join(summary) or f"{len(failures)} failing tests"
    shown = failures[:MAX_ANNOTATIONS - 1]
    for header, body in shown:
        print(f"::error title={escape_property(header)}::{escape(chr(10).join(body))}")
    names = "\n".join(header for header, _ in failures)
    print(f"::error title={escape_property('All failing tests (' + title + ')')}::{escape(names)}")


if __name__ == "__main__":
    main(sys.argv[1])
