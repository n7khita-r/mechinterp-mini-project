#!/usr/bin/env python3
"""Split a generated booktabs table into N side-by-side chunks.

A 52-row table does not fit on a slide: scaled to the frame height it becomes a
tall thin column, and scaled to the width the text is unreadable. Splitting the
body across two tabulars of equal height, each with its own header, keeps the
font legible and the frame full.

    python scripts/split_table.py presentation/tables/controls.tex 2

Writes <name>_1of2.tex, <name>_2of2.tex next to the original.
"""

from __future__ import annotations

import sys
from pathlib import Path


def split(path: Path, parts: int) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()

    try:
        mid = next(i for i, l in enumerate(lines) if l.strip() == r"\midrule")
        bot = next(i for i, l in enumerate(lines) if l.strip() == r"\bottomrule")
    except StopIteration:
        raise SystemExit(f"{path}: not a booktabs table (need \\midrule and \\bottomrule)")

    head, body, tail = lines[: mid + 1], lines[mid + 1 : bot], lines[bot:]
    # Keep only real data rows; the generator inserts \midrule between groups.
    rows = [r for r in body if r.strip() and r.strip() != r"\midrule"]

    per = -(-len(rows) // parts)          # ceiling, so the last chunk is the short one
    out = []
    for k in range(parts):
        chunk = rows[k * per : (k + 1) * per]
        if not chunk:
            continue
        dest = path.with_name(f"{path.stem}_{k + 1}of{parts}.tex")
        dest.write_text("\n".join(head + chunk + tail) + "\n", encoding="utf-8")
        out.append(dest)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    parts = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for p in split(path, parts):
        n = sum(1 for l in p.read_text().splitlines() if l.strip().endswith(r"\\"))
        print(f"wrote {p}  ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
