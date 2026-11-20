#!/usr/bin/env python3
"""Bundle the presentation as a self-contained LaTeX zip.

Everything needed to rebuild the deck on another machine: the source, the
vendored Telugu font (so it does not depend on system fonts), the figures it
actually includes, and the compiled PDF. Build artefacts are left out.

    python scripts/make_slides_zip.py

Writes presentation/slides.zip.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "presentation"
OUT = SRC / "slides.zip"

# Only these extensions are build inputs; everything else in the folder is
# output or scratch.
KEEP_SUFFIX = {".tex", ".ttf", ".otf", ".sty", ".bst", ".bib", ".pdf", ".png"}
SKIP_SUFFIX = {".aux", ".log", ".nav", ".out", ".snm", ".toc", ".vrb",
               ".fls", ".fdb_latexmk", ".synctex.gz", ".xdv"}


def included_images(tex: str) -> set[str]:
    """Image basenames actually referenced by \\includegraphics."""
    return {Path(m).name for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)}


def included_inputs(tex: str) -> set[str]:
    """Paths pulled in by \\input -- the generated tables live here.

    Without these the bundle builds on this machine (where the files exist) and
    fails everywhere else, which is the worst kind of broken.
    """
    out = set()
    for m in re.findall(r"\\input\{([^}]+)\}", tex):
        out.add(m if m.endswith(".tex") else f"{m}.tex")
    return out


def main() -> int:
    main_tex = SRC / "main.tex"
    if not main_tex.exists():
        print(f"missing {main_tex}")
        return 1
    tex = main_tex.read_text(encoding="utf-8")
    wanted = included_images(tex)

    members: list[tuple[Path, str]] = [(main_tex, "presentation/main.tex")]

    for f in sorted((SRC / "fonts").glob("*")):
        if f.suffix.lower() in KEEP_SUFFIX:
            members.append((f, f"presentation/fonts/{f.name}"))

    img_dir = SRC / "images"
    missing = []
    for name in sorted(wanted):
        # \includegraphics{foo.png} resolves via \graphicspath{{images/}}
        cand = img_dir / name
        if cand.exists():
            members.append((cand, f"presentation/images/{name}"))
        elif (img_dir / f"{name}.png").exists():
            members.append((img_dir / f"{name}.png", f"presentation/images/{name}.png"))
        else:
            missing.append(name)

    for rel in sorted(included_inputs(tex)):
        f = SRC / rel
        if f.exists():
            members.append((f, f"presentation/{rel}"))
        else:
            missing.append(rel)

    pdf = SRC / "main.pdf"
    if pdf.exists():
        members.append((pdf, "presentation/main.pdf"))

    readme = (
        "Build with LuaLaTeX, not pdfLaTeX -- the Telugu examples need a Unicode\n"
        "engine. The font ships in fonts/, so no system font is required.\n\n"
        "    cd presentation && lualatex main.tex\n"
    )

    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path, arc in members:
            z.write(path, arc)
        z.writestr("presentation/BUILD.txt", readme)

    size = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({size:.0f} KB, {len(members) + 1} files)")
    for _, arc in members:
        print(f"  {arc}")
    if missing:
        print(f"\nWARNING: referenced but not found: {missing}")
        print("The zip will not build until these exist.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
