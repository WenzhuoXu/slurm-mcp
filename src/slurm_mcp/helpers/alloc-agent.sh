#!/bin/bash
# slurm-mcp alloc-agent.sh v2 — Usage: alloc-agent.sh <ctrl_dir> [idle_release_s]
# One event loop (1 s): heartbeat -> release? -> start new cmds -> poll all running cmds (fg and bg alike) for exit / .kill -> idle?
# Runs <ctrl>/cmds/NNN.sh one at a time (foreground: the next NNN.sh starts only when no fg command runs) and NNN.bg.sh detached
# (any number, started as soon as seen). Writes <base>.{out,pid,started,rc,done} where <base> = the file name without ".sh"
# (NNN or NNN.bg).  Control files: <ctrl>/release (exit), <ctrl>/cmds/<base>.kill (SIGTERM the command's process group, SIGKILL after 30 s):
# i.e. 002.kill for 002.sh and 003.bg.kill for 003.bg.sh — the server writes exactly these names.
set -u
CTRL="$1"; IDLE="${2:-0}"; Q="$CTRL/cmds"; mkdir -p "$Q"
export SLURM_MCP_CTRL="$CTRL"
HAVE_SETSID=0; command -v setsid >/dev/null 2>&1 && HAVE_SETSID=1
declare -A PID=()          # base -> pid of every running command (foreground and background)
FG=""                      # base of the running foreground command, if any
LAST=$(date +%s)
status() { printf '{"v":2,"phase":"%s","node":"%s","job_id":"%s","now":%s,"start":%s,"fg":"%s","running":%s}\n' "$1" "$(hostname -s)" "${SLURM_JOB_ID:-}" "$(date +%s)" "$T0" "${FG##*/}" "${#PID[@]}" > "$CTRL/status.json.tmp.$$" && mv -f "$CTRL/status.json.tmp.$$" "$CTRL/status.json"; }
killgrp() { if [ "$HAVE_SETSID" = 1 ]; then kill -TERM -- "-$1" 2>/dev/null; else kill -TERM "$1" 2>/dev/null; fi
            ( sleep 30; if [ "$HAVE_SETSID" = 1 ]; then kill -KILL -- "-$1" 2>/dev/null; else kill -KILL "$1" 2>/dev/null; fi ) & }
start() {  # $1 = script path; launched directly in this shell so `wait` works (never inside $(...))
  local s="$1" b="${1%.sh}"
  if [ "$HAVE_SETSID" = 1 ]; then setsid bash "$s" > "$b.out" 2>&1 & else bash "$s" > "$b.out" 2>&1 & fi
  PID[$b]=$!; echo "${PID[$b]}" > "$b.pid"; date +%s > "$b.started"
  case "$s" in *.bg.sh) ;; *) FG=$b ;; esac
}
finish() {  # $1 = base
  local b="$1" pid="${PID[$1]}" rc
  wait "$pid"; rc=$?
  echo "$rc" > "$b.rc.tmp.$$" && mv -f "$b.rc.tmp.$$" "$b.rc"; date +%s > "$b.done"
  unset "PID[$b]"; [ "$FG" = "$b" ] && FG=""; LAST=$(date +%s)
}
cleanup() { for b in "${!PID[@]}"; do killgrp "${PID[$b]}"; done; status exited; exit 0; }
trap cleanup TERM INT
T0=$(date +%s)
echo "=== slurm-mcp alloc-agent: job ${SLURM_JOB_ID:-?} node $(hostname -s) $(date -Is) ==="
status ready
while :; do
  date +%s > "$CTRL/heartbeat.tmp.$$" && mv -f "$CTRL/heartbeat.tmp.$$" "$CTRL/heartbeat"     # every second, fg command or not
  [ -f "$CTRL/release" ] && cleanup
  for s in "$Q"/*.sh; do                                # zero-padded names => submission order; ls-free
    [ -e "$s" ] || continue
    b="${s%.sh}"; [ -f "$b.pid" ] && continue           # already started (this or a previous agent incarnation)
    case "$s" in
      *.bg.sh) start "$s" ;;                            # detached: always start
      *)       [ -z "$FG" ] && start "$s" ;;            # foreground: only one at a time
    esac
  done
  for b in "${!PID[@]}"; do                             # poll everything that runs, fg and bg alike
    p=${PID[$b]}
    [ -f "$b.kill" ] && { killgrp "$p"; rm -f "$b.kill"; }
    kill -0 "$p" 2>/dev/null || finish "$b"
  done
  if [ "$IDLE" -gt 0 ] && [ "${#PID[@]}" -eq 0 ] && [ $(( $(date +%s) - LAST )) -ge "$IDLE" ]; then echo "slurm-mcp: idle release"; cleanup; fi
  status running
  sleep 1
done
