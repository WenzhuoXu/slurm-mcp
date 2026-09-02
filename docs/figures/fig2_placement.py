"""Figure 2: the placement decision, drawn.

Real output of one plan_job call on 2026-09-02 across CMU TRACE and PSC Bridges-2. The score the
placer reports is in hours and decomposes exactly into queue wait + the job's own wall clock +
the SU price converted to hours at the `balanced` objective (0.25 h per SU), so the bars are that
decomposition rather than a re-derivation:

    trace:cpuonly-debug  0.003 + 0.1667 + 0      = 0.167  (reported 0.167)
    bridges2:RM-small    0.004 + 0.1667 + 0.0825 = 0.253  (reported 0.249)
    trace:cpuonly        1.75  + 0.1667 + 0      = 1.917  (reported 1.917)
    bridges2:RM-512      1.75  + 0.1667 + 0.0825 = 1.999  (reported 1.999)
    bridges2:EM          2.0   + 0.1667 + 0.0825 = 2.249  (reported 2.249)
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figs import (AMBER, AMBER_BG, BLUE, FAINT, Fig, GREEN, GREEN_BG, INK, LINE, MUTED,  # noqa: E402
                  RED, RED_BG, SURFACE, WHITE, sans_w)

OUT = pathlib.Path(__file__).resolve().parent.parent / "img"

W, H = 1020, 556
BAR_X = 322                     # where bars start
BAR_MAX = 560                   # px for HOURS_MAX hours
HOURS_MAX = 2.4

WAIT_C, RUN_C, SU_C = "#0969da", "#8250df", "#bf8700"

# target, cluster, wait_h used in the score, runtime_h, su, score_h, note
ROWS = [
    ("trace:cpuonly-debug", "TRACE", 0.003, 0.1667, 0.0, 0.167, "queue ahead 0 \u00b7 free"),
    ("bridges2:RM-small", "Bridges-2", 0.004, 0.1667, 0.33, 0.249, "queue ahead 0 \u00b7 0.33 SU"),
    ("trace:cpuonly", "TRACE", 1.75, 0.1667, 0.0, 1.917, "queue ahead 7 \u00b7 free"),
    ("bridges2:RM-512", "Bridges-2", 1.75, 0.1667, 0.33, 1.999, "queue ahead 7 \u00b7 0.33 SU"),
    ("bridges2:EM", "Bridges-2", 2.0, 0.1667, 0.33, 2.249, "queue ahead 8 \u00b7 0.33 SU"),
]


def px(hours: float) -> float:
    return hours / HOURS_MAX * BAR_MAX


def main() -> None:
    f = Fig(W, H)

    f.title(30, 34,
            "It ranks every partition you are allowed to use, on every cluster you configured",
            "Score is in hours, and lower wins: time until it starts, plus how long it runs, plus what it "
            "costs priced back into hours.")

    # legend
    lx, ly = 30, 76
    for colour, label in ((WAIT_C, "queue wait"), (RUN_C, "your job's wall clock"),
                          (SU_C, "SU price at 0.25 h/SU")):
        f.rect(lx, ly, 10, 10, fill=colour, rx=2)
        f.text(lx + 16, ly + 9, label, size=11.5, fill=MUTED)
        lx += sans_w(label, 11.5) + 40

    top = 106
    row_h = 62

    for i, (target, cluster, wait, run, su, score, note) in enumerate(ROWS):
        y = top + i * row_h
        chosen = i == 0
        if chosen:
            f.rect(22, y - 8, W - 44, row_h - 4, fill=GREEN_BG, rx=8)

        # name + cluster tag
        f.text(38, y + 12, target, size=13, mono=True, weight="600" if chosen else None,
               fill=INK if chosen else "#32383f")
        tag_fg, tag_bg = (GREEN, "#c9f3d4") if cluster == "TRACE" else (BLUE, "#ddf4ff")
        f.pill(38, y + 20, cluster, fg=tag_fg, bg=tag_bg, size=10, h=17, pad=6)
        f.text(38 + sans_w(cluster, 10) + 22, y + 32, note, size=10.5, fill=FAINT)

        # stacked bar
        bx = BAR_X
        for value, colour in ((wait, WAIT_C), (run, RUN_C), (su * 0.25, SU_C)):
            w = px(value)
            if w > 0.6:
                f.rect(bx, y, w, 20, fill=colour, rx=3)
            bx += w

        f.text(bx + 12, y + 15, f"{score:.3f} h", size=12.5, weight="600",
               fill=INK if chosen else MUTED)

        if chosen:
            f.pill(bx + 78, y + 1, "chosen \u2014 submitted here", fg=GREEN, bg="#c9f3d4", size=10.5, h=18)

    # ---- the one it refused ------------------------------------------------------------
    y = top + len(ROWS) * row_h + 4
    f.line(30, y - 6, W - 30, y - 6, stroke=LINE, dash="3 3")
    f.text(38, y + 18, "bridges2:applications", size=13, mono=True, fill=FAINT)
    f.rect(BAR_X, y + 6, 190, 20, fill=RED_BG, rx=3)
    f.text(BAR_X + 12, y + 20, "Invalid qos specification", size=11, fill=RED, weight="600")
    f.text(BAR_X + 214, y + 20,
           "ruled out before you could waste a submission on it", size=11.5, fill=MUTED)

    # ---- the takeaway ------------------------------------------------------------------
    box_y = y + 48
    f.rect(30, box_y, W - 60, 46, fill=SURFACE, stroke=LINE, rx=8)
    f.text(46, box_y + 20, "Free beats charging on a tie, and a short queue beats a long one.",
           size=12, weight="600", fill=INK)
    f.text(46, box_y + 37,
           "Bridges-2 RM-small would have started just as fast, so the 0.33 SU is what lost it. "
           "Change that with objective=fastest or cheapest.", size=11.5, fill=MUTED)

    f.save(OUT / "02-placement.svg")


if __name__ == "__main__":
    main()
