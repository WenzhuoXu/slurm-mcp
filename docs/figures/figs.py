"""Shared drawing primitives for the slurm-mcp README figures.

Plain SVG, no dependencies, no external fonts. Each figure paints its own light background so it
reads the same in GitHub's light and dark themes.
"""
from __future__ import annotations

import html
import pathlib

# ---- palette ---------------------------------------------------------------------------------
INK = "#1f2328"
MUTED = "#59636e"
FAINT = "#818b98"
LINE = "#d1d9e0"
SURFACE = "#f6f8fa"
WHITE = "#ffffff"
GREEN = "#1a7f37"
GREEN_BG = "#dafbe1"
BLUE = "#0969da"
BLUE_BG = "#ddf4ff"
PURPLE = "#8250df"
PURPLE_BG = "#fbefff"
AMBER = "#9a6700"
AMBER_BG = "#fff8c5"
RED = "#cf222e"
RED_BG = "#ffebe9"

SANS = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif")
MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace")

MONO_W = 0.601          # width of one monospace char at font-size 1
SANS_W = 0.512          # rough average for the sans stack


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def mono_w(text: str, size: float) -> float:
    return len(text) * MONO_W * size


def sans_w(text: str, size: float) -> float:
    return len(text) * SANS_W * size


class Fig:
    """An SVG canvas with a few helpers. Coordinates are plain user units."""

    def __init__(self, w: int, h: int, bg: str = WHITE) -> None:
        self.w, self.h = w, h
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{SANS}">',
            f'<rect width="{w}" height="{h}" fill="{bg}"/>',
        ]

    # -- primitives ---------------------------------------------------------------------------
    def rect(self, x, y, w, h, *, fill=WHITE, stroke=None, rx=8, sw=1, dash=None, opacity=None) -> None:
        a = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{sw}"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        self.parts.append(a + "/>")

    def text(self, x, y, s, *, size=13, fill=INK, weight=None, mono=False, anchor="start",
             opacity=None, style=None) -> None:
        a = (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}"'
             f'{f" font-family=\"{MONO}\"" if mono else ""}')
        if weight:
            a += f' font-weight="{weight}"'
        if anchor != "start":
            a += f' text-anchor="{anchor}"'
        if opacity is not None:
            a += f' opacity="{opacity}"'
        if style:
            a += f' font-style="{style}"'
        self.parts.append(a + f'>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=LINE, sw=1, dash=None, cap="round") -> None:
        a = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
             f'stroke-width="{sw}" stroke-linecap="{cap}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        self.parts.append(a + "/>")

    def path(self, d, *, stroke=LINE, fill="none", sw=1, dash=None, marker=None) -> None:
        a = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"'
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if marker:
            a += f' marker-end="url(#{marker})"'
        self.parts.append(a + "/>")

    def circle(self, cx, cy, r, *, fill=WHITE, stroke=None, sw=1) -> None:
        a = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{sw}"'
        self.parts.append(a + "/>")

    # -- composites ---------------------------------------------------------------------------
    def arrowhead(self, name: str, colour: str) -> None:
        self.parts.append(
            f'<defs><marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker></defs>')

    def pill(self, x, y, label, *, fg, bg, size=11, pad=8, h=20, weight="600") -> float:
        w = sans_w(label, size) + pad * 2
        self.rect(x, y, w, h, fill=bg, rx=h / 2)
        self.text(x + pad, y + h / 2 + size * 0.36, label, size=size, fill=fg, weight=weight)
        return w

    def title(self, x, y, main: str, sub: str | None = None) -> None:
        self.text(x, y, main, size=17, weight="600")
        if sub:
            self.text(x, y + 20, sub, size=12.5, fill=MUTED)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts) + "\n</svg>\n", encoding="utf-8")
        print(f"  {path.name:<26} {self.w}x{self.h}  {path.stat().st_size / 1024:.1f} KB")
