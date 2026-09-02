"""Figure 3: what happens after you submit, and why you can walk away.

The four cards are the real event sequence of job j7 (SLURM 615819) on TRACE, 2026-09-02, from the
ledger: submit 1788378173, start 1788378175, end 1788378178. The second row is the rest of the
event vocabulary the monitor emits, which is what makes a long job safe to leave alone.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figs import (AMBER, AMBER_BG, BLUE, BLUE_BG, FAINT, Fig, GREEN, GREEN_BG, INK,  # noqa: E402
                  LINE, MUTED, PURPLE, PURPLE_BG, RED, RED_BG, SURFACE, WHITE, sans_w)

OUT = pathlib.Path(__file__).resolve().parent.parent / "img"

W, H = 1020, 500
M = 30

CARDS = [
    ("submit_job", "you, once", "placement=\"auto\"\nhandle j7 comes back immediately", BLUE, BLUE_BG, False),
    ("submitted", "t+0 s", "615819 on trace:cpuonly-debug\nqos, account and paths filled in", BLUE, BLUE_BG, True),
    ("started", "t+2 s", "node trace123\ntwo seconds of queue", PURPLE, PURPLE_BG, True),
    ("completed", "t+5 s", "rc=0 after 3 s\nconfirmed against sacct", GREEN, GREEN_BG, True),
]

TROUBLE = [
    ("preempted", RED, RED_BG, "higher-priority job took the node"),
    ("requeued", AMBER, AMBER_BG, "put back in the queue, attempt 2"),
    ("timeout", AMBER, AMBER_BG, "hit the wall clock"),
    ("oom", RED, RED_BG, "killed for memory"),
    ("needs_attention", RED, RED_BG, "restart loop, quota, stale heartbeat"),
]


def main() -> None:
    f = Fig(W, H)
    f.arrowhead("gray", "#8c959f")

    f.title(M, 34, "It watches the job so you do not, and it cannot lose the answer",
            "One real job on CMU TRACE. Every box below is an event the server recorded and handed to the "
            "session.")

    # ---- the happy path -----------------------------------------------------------------
    top = 82
    n = len(CARDS)
    gap = 26
    cw = (W - 2 * M - gap * (n - 1)) / n
    ch = 96

    for i, (name, when, body, fg, bg, is_event) in enumerate(CARDS):
        x = M + i * (cw + gap)
        f.rect(x, top, cw, ch, fill=WHITE, stroke=fg if is_event else LINE,
               sw=1.5 if is_event else 1)
        f.rect(x, top, cw, 4, fill=fg, rx=2)

        f.text(x + 14, top + 30, name, size=13, weight="600", mono=not is_event, fill=INK)
        f.text(x + cw - 14, top + 30, when, size=10.5, fill=FAINT, anchor="end")
        for j, line in enumerate(body.split("\n")):
            f.text(x + 14, top + 54 + j * 17, line, size=11.5, fill=MUTED)

        if i < n - 1:
            ax = x + cw + 5
            f.path(f"M {ax} {top + ch / 2} L {ax + gap - 10} {top + ch / 2}",
                   stroke="#8c959f", sw=1.4, marker="gray")

    f.text(M, top + ch + 22,
           "You call submit_job once and wait_for_events once. The three event boxes arrive on their own.",
           size=11.5, fill=MUTED)

    # ---- when it does not go well -------------------------------------------------------
    ty = top + ch + 52
    f.rect(M, ty, W - 2 * M, 96, fill=SURFACE, stroke=LINE, rx=8)
    f.text(M + 18, ty + 26, "And when it does not go well, that is an event too",
           size=13, weight="600")
    f.text(M + 18, ty + 44, "Each one carries the exit code, the cause and the log path, so the "
                            "next thing you ask can act on it.", size=11.5, fill=MUTED)

    px_ = M + 18
    for label, fg, bg, note in TROUBLE:
        w = f.pill(px_, ty + 58, label, fg=fg, bg=bg, size=11, h=22)
        f.text(px_, ty + 92, note, size=10, fill=FAINT)
        px_ += max(w, sans_w(note, 10)) + 22

    # ---- durability ---------------------------------------------------------------------
    by = ty + 116
    f.rect(M, by, W - 2 * M, 86, fill=WHITE, stroke=GREEN, sw=1.5, rx=8)
    f.text(M + 18, by + 26, "Close the laptop. The answer waits for you.", size=13,
           weight="600", fill=GREEN)

    steps = [
        ("delivered", "handed to the session, but not yet marked seen"),
        ("acknowledged", "only when you come back and confirm the batch"),
        ("otherwise replayed", "a result lost to a dead session is re-sent, never dropped"),
    ]
    col_w = (W - 2 * M - 36) / 3
    for i, (head, body) in enumerate(steps):
        x = M + 18 + i * col_w
        f.circle(x + 4, by + 48, 4, fill=GREEN)
        f.text(x + 16, by + 52, head, size=11.5, weight="600", fill=INK)
        f.text(x + 16, by + 70, body, size=10.8, fill=MUTED)
        if i:
            f.line(x - 12, by + 40, x - 12, by + 74, stroke=LINE)

    f.save(OUT / "03-lifecycle.svg")


if __name__ == "__main__":
    main()
