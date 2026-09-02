"""Rebuild every README figure into ../img.

    python docs/figures/build.py

No dependencies beyond the standard library: each figure is emitted as plain SVG so it stays a few
kilobytes, renders inline on GitHub, and shows up as a readable diff when a number changes.

The numbers in the figures are transcribed from recorded runs against CMU TRACE and PSC Bridges-2
on 2026-09-02, and each module's docstring says which call it came from. When you re-record a run,
update the constants at the top of the relevant module rather than the drawing code.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = [
    "fig1_contrast",
    "fig2_placement",
    "fig3_lifecycle",
    "fig4_architecture",
    "fig5_capabilities",
]


def main() -> int:
    out = HERE.parent / "img"
    print(f"building {len(MODULES)} figure(s) into {out}")
    for name in MODULES:
        importlib.import_module(name).main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
