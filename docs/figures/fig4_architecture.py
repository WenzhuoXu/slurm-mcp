"""Figure 4: where each piece runs, and what you have to install where.

The short answer the diagram is built around: everything lives on your machine, the cluster gets
nothing permanent, and the password never leaves the OS keyring.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figs import (AMBER, AMBER_BG, BLUE, BLUE_BG, FAINT, Fig, GREEN, GREEN_BG, INK,  # noqa: E402
                  LINE, MUTED, PURPLE, PURPLE_BG, SURFACE, WHITE, sans_w)

OUT = pathlib.Path(__file__).resolve().parent.parent / "img"

W, H = 1020, 470
M = 30


def band(f: Fig, x, y, w, h, title, subtitle, accent):
    f.rect(x, y, w, h, fill=WHITE, stroke=LINE)
    f.rect(x, y, w, 4, fill=accent, rx=2)
    f.text(x + 16, y + 28, title, size=12.5, weight="700", fill=INK)
    f.text(x + 16, y + 45, subtitle, size=10.5, fill=FAINT)


def chip(f: Fig, x, y, w, label, detail, *, fg=INK, bg=SURFACE, mono=True):
    f.rect(x, y, w, 40, fill=bg, rx=6)
    f.text(x + 12, y + 18, label, size=11.5, mono=mono, weight="600", fill=fg)
    f.text(x + 12, y + 32, detail, size=10, fill=MUTED)


def main() -> None:
    f = Fig(W, H)
    f.arrowhead("blue", BLUE)
    f.arrowhead("faint", "#8c959f")

    f.title(M, 34, "Everything runs on your machine. The cluster gets nothing permanent.",
            "No daemon to install, no admin to ask, no scheduler plugin. Just your existing SSH login.")

    top = 82
    col_w = 292
    gap = 42
    x1 = M
    x2 = x1 + col_w + gap
    x3 = x2 + col_w + gap
    band_h = 236

    # ---------------- your machine ----------------
    band(f, x1, top, col_w, band_h, "YOUR MACHINE", "Windows, macOS or Linux", BLUE)
    chip(f, x1 + 16, top + 58, col_w - 32, "Claude Code", "talks MCP over stdio", mono=False)
    chip(f, x1 + 16, top + 106, col_w - 32, "slurm_mcp.server", "20 tools, one asyncssh connection per cluster")
    chip(f, x1 + 16, top + 154, col_w - 32, "~/.slurm-mcp/state.db",
         "SQLite ledger: jobs, attempts, events")
    chip(f, x1 + 16, top + 196, col_w - 32, "OS keyring", "the password, and only here",
         fg=GREEN, bg=GREEN_BG, mono=False)

    # ---------------- login node ----------------
    band(f, x2, top, col_w, band_h, "LOGIN NODE", "trace.cmu.edu \u00b7 bridges2.psc.edu", PURPLE)
    chip(f, x2 + 16, top + 58, col_w - 32, "sbatch  squeue  sacct", "the ordinary CLI, as you")
    chip(f, x2 + 16, top + 106, col_w - 32, "scontrol  sacctmgr  sinfo",
         "discovery: partitions, QOS, limits")
    chip(f, x2 + 16, top + 154, col_w - 32, "SFTP", "or a data-transfer node when the login node refuses")
    chip(f, x2 + 16, top + 196, col_w - 32, "~/.slurm-mcp/bin/<hash>/",
         "helpers, re-uploaded only when they change", fg=AMBER, bg=AMBER_BG)

    # ---------------- compute node ----------------
    band(f, x3, top, col_w, band_h, "COMPUTE NODE", "where your job actually runs", GREEN)
    chip(f, x3 + 16, top + 58, col_w - 32, "wrap.sh", "wraps your script, catches the signal")
    chip(f, x3 + 16, top + 106, col_w - 32, "your job", "unchanged, in your own workdir", mono=False)
    chip(f, x3 + 16, top + 154, col_w - 32, "status.json  heartbeat  rc",
         "so a lost sbatch reply is still recoverable")
    chip(f, x3 + 16, top + 196, col_w - 32, "alloc-agent.sh",
         "only for held allocations", fg=AMBER, bg=AMBER_BG)

    # ---------------- arrows ----------------
    ay = top + band_h / 2
    f.path(f"M {x1 + col_w + 6} {ay} L {x2 - 8} {ay}", stroke=BLUE, sw=2, marker="blue")
    f.text((x1 + col_w + x2) / 2, ay - 12, "SSH", size=11, weight="700", fill=BLUE, anchor="middle")
    f.text((x1 + col_w + x2) / 2, ay + 22, "password", size=9.5, fill=FAINT, anchor="middle")

    f.path(f"M {x2 + col_w + 6} {ay} L {x3 - 8} {ay}", stroke="#8c959f", sw=2, marker="faint")
    f.text((x2 + col_w + x3) / 2, ay - 12, "SLURM", size=11, weight="700", fill=MUTED, anchor="middle")

    # ---------------- the three claims ----------------
    by = top + band_h + 30
    claims = [
        ("Nothing to install on the cluster", GREEN,
         "The three helper scripts are uploaded per job, keyed by content hash, into your own home."),
        ("The password never leaves the keyring", GREEN,
         "Not in a config file, not in a log, not on a command line. auth set refuses a piped stdin."),
        ("Kill the server mid-job and nothing is lost", GREEN,
         "The ledger is on your disk; on restart the monitor reconciles it against sacct and squeue."),
    ]
    cw = (W - 2 * M - 24) / 3
    for i, (head, colour, body) in enumerate(claims):
        x = M + i * (cw + 12)
        f.rect(x, by, cw, 76, fill=SURFACE, stroke=LINE, rx=8)
        f.text(x + 14, by + 24, head, size=11.8, weight="600", fill=colour)
        words, lines, cur = body.split(), [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if sans_w(trial, 10.5) > cw - 28 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        for j, line in enumerate(lines[:3]):
            f.text(x + 14, by + 44 + j * 14, line, size=10.5, fill=MUTED)

    f.save(OUT / "04-architecture.svg")


if __name__ == "__main__":
    main()
