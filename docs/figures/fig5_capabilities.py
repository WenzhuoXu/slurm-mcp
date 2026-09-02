"""Figure 5: the twenty tools, grouped by the thing you would actually ask for.

The point of the figure is that you never name a tool. You say the sentence on the card; Claude
picks from the row underneath it.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figs import (AMBER, BLUE, FAINT, Fig, GREEN, INK, LINE, MONO, MUTED, PURPLE,  # noqa: E402
                  RED, SURFACE, WHITE, mono_w, sans_w)

OUT = pathlib.Path(__file__).resolve().parent.parent / "img"

W, H = 1020, 560
M = 30

TEAL = "#0f766e"

GROUPS = [
    (BLUE, "See what you have",
     "\u201cwhat's free on either cluster right now?\u201d",
     ["clusters", "cluster_status", "list_jobs"]),
    (PURPLE, "Decide where it runs",
     "\u201cwhere would this start soonest without burning SUs?\u201d",
     ["plan_job", "rebalance"]),
    (GREEN, "Run it",
     "\u201crun this sweep, four at a time\u201d",
     ["submit_job", "allocate", "alloc_run", "run_command"]),
    (TEAL, "Watch it",
     "\u201ctell me when it finishes, and why if it doesn't\u201d",
     ["wait_for_events", "job_status", "job_logs"]),
    (AMBER, "Steer it",
     "\u201ckill the stuck one and move the rest somewhere faster\u201d",
     ["job_control", "configure"]),
    (RED, "Move the data",
     "\u201cbring the outputs back and show me the log\u201d",
     ["upload", "download", "collect_results", "remote_ls", "remote_read", "remote_write"]),
]


def main() -> None:
    f = Fig(W, H)

    f.title(M, 34, "Twenty tools, and you never have to name one of them",
            "You say the sentence. Claude picks the tools. This is the whole surface, grouped by what you "
            "are trying to get done.")

    top = 86
    cols, rows = 3, 2
    gap = 20
    cw = (W - 2 * M - gap * (cols - 1)) / cols
    ch = 196

    for i, (colour, head, ask, tools) in enumerate(GROUPS):
        cx = M + (i % cols) * (cw + gap)
        cy = top + (i // cols) * (ch + gap)

        f.rect(cx, cy, cw, ch, fill=WHITE, stroke=LINE)
        f.rect(cx, cy, cw, 4, fill=colour, rx=2)
        f.text(cx + 16, cy + 32, head, size=14, weight="600", fill=INK)

        # the plain-English ask
        f.rect(cx + 16, cy + 46, cw - 32, 44, fill=SURFACE, rx=6)
        words, lines, cur = ask.split(), [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if sans_w(trial, 11.5) > cw - 60 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        for j, line in enumerate(lines[:2]):
            f.text(cx + 28, cy + 66 + j * 15, line, size=11.5, fill=MUTED, style="italic")

        f.text(cx + 16, cy + 112, "reaches for", size=9.5, weight="700", fill=FAINT)

        # tool pills, wrapped
        px_, py = cx + 16, cy + 122
        for tool in tools:
            tw = mono_w(tool, 11) + 16
            if px_ + tw > cx + cw - 16:
                px_ = cx + 16
                py += 26
            f.rect(px_, py, tw, 21, fill="#f6f8fa", stroke=LINE, rx=5)
            f.text(px_ + 8, py + 15, tool, size=11, mono=True, fill=colour, weight="600")
            px_ += tw + 7

    f.text(M, H - 16,
           "Read-only tools (clusters, cluster_status, list_jobs, job_status, job_logs, plan_job, "
           "wait_for_events, remote_ls, remote_read) can be allow-listed so they never prompt.",
           size=11, fill=FAINT)

    f.save(OUT / "05-capabilities.svg")


if __name__ == "__main__":
    main()
