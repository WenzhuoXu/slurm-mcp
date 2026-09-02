"""Figure 1: what the server actually replaces.

Left, the manual loop anyone with a cluster account already knows. Right, the same outcome asked
for in a sentence. Every number on the right is from the recorded run on TRACE on 2026-09-02.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figs import (AMBER, BLUE, FAINT, Fig, GREEN, GREEN_BG, INK, LINE, MONO, MUTED,  # noqa: E402
                  SURFACE, WHITE, mono_w, sans_w)

OUT = pathlib.Path(__file__).resolve().parent.parent / "img"

W, H = 1020, 560
COL_W = 470
LX, RX = 30, 520

MANUAL = [
    ("ssh you@trace.cmu.edu", "and again for the other cluster"),
    ("sinfo -o '%P %a %l %D %t'", "which partition is free right now?"),
    ("squeue -p cpuonly | wc -l", "…and how long is the line?"),
    ("sacctmgr show assoc user=$USER", "am I even allowed on it? which QOS?"),
    ("vim run.sh", "hand-write #SBATCH, get --qos wrong, retry"),
    ("sbatch run.sh", "615819. now what?"),
    ("watch -n30 squeue -u $USER", "sit there, or forget about it for a day"),
    ("sacct -j 615819 -X", "preempted? requeued? exit code?"),
    ("scp -r trace:...:/out ./", "find the files, guess the path"),
]

ASSISTANT = [
    ("ranked", "5 targets across both clusters, with a real "
               "sbatch --test-only start estimate"),
    ("chose", "trace:cpuonly-debug — free, ~0 h wait — over "
              "bridges2:RM-small at 0.33 SU"),
    ("submitted", "615819, with --qos, -A, --requeue and the "
                  "output paths filled in for you"),
    ("watched", "started on trace123 after 2 s, completed rc=0 "
                "after 3 s"),
    ("delivered", "2 files on this laptop, π = 3.141338"),
]


def wrap(text: str, size: float, width: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if sans_w(trial, size) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def main() -> None:
    f = Fig(W, H)
    f.arrowhead("gray", FAINT)

    f.title(LX, 34, "You already know how to do this. That is the problem.",
            "The right-hand column is one sentence to Claude Code. Everything under it is what the server did on its own.")

    top = 84

    # ---------------- left: the manual loop ----------------------------------------------
    f.rect(LX, top, COL_W, 430, fill=SURFACE, stroke=LINE)
    f.text(LX + 18, top + 28, "BY HAND", size=11, weight="700", fill=MUTED)
    f.text(LX + 18, top + 48, "every time, for every job, on every cluster", size=11.5, fill=FAINT)

    y = top + 76
    for cmd, note in MANUAL:
        f.text(LX + 18, y, "$", size=11.5, mono=True, fill=FAINT)
        f.text(LX + 32, y, cmd, size=11.5, mono=True, fill=INK)
        f.text(LX + 32, y + 15, note, size=11, fill=FAINT, style="italic")
        y += 39

    # ---------------- right: the one sentence --------------------------------------------
    f.rect(RX, top, COL_W, 430, fill=WHITE, stroke=GREEN, sw=1.5)
    f.text(RX + 18, top + 28, "WITH slurm-mcp", size=11, weight="700", fill=GREEN)

    # the ask, as a speech bubble
    ask = ("\u201cRun this Monte Carlo wherever it starts soonest. "
           "Don't spend SUs if you don't have to. Tell me when it's done.\u201d")
    ask_lines = wrap(ask, 13, COL_W - 76)
    bub_h = 20 + len(ask_lines) * 19
    f.rect(RX + 18, top + 42, COL_W - 36, bub_h, fill="#f0f6ff", stroke="#c8e1ff", rx=10)
    for i, line in enumerate(ask_lines):
        f.text(RX + 34, top + 66 + i * 19, line, size=13, fill="#0a3069")
    f.text(RX + 34, top + 42 + bub_h + 16, "you, in the chat", size=10.5, fill=FAINT)

    # what it did
    y = top + 42 + bub_h + 44
    f.line(RX + 18, y, RX + COL_W - 18, y, stroke=LINE)
    y += 26
    for verb, detail in ASSISTANT:
        f.circle(RX + 26, y - 4, 3.5, fill=GREEN)
        f.text(RX + 40, y, verb, size=12.5, weight="600", fill=INK)
        vw = sans_w(verb, 12.5) + 10
        lines = wrap(detail, 12, COL_W - 58 - vw)
        f.text(RX + 40 + vw, y, lines[0], size=12, fill=MUTED)
        for i, line in enumerate(lines[1:], 1):
            f.text(RX + 40, y + i * 16, line, size=12, fill=MUTED)
        y += 16 * len(lines) + 16

    # the payoff
    f.rect(RX + 18, top + 356, COL_W - 36, 56, fill=GREEN_BG, rx=8)
    f.text(RX + 34, top + 379, "One message. 107 s from ask to files on disk,",
           size=12.5, weight="600", fill=GREEN)
    f.text(RX + 34, top + 398, "3 s of which was the job itself. None of it was you.",
           size=12.5, weight="600", fill=GREEN)

    f.text(LX, H - 16,
           "Numbers recorded from a real run on CMU TRACE, 2026-09-02. Nothing here is a mock-up.",
           size=11, fill=FAINT)

    f.save(OUT / "01-what-it-replaces.svg")


if __name__ == "__main__":
    main()
