#!/bin/bash
# slurm-mcp wrap.sh v2 — runs inside the batch job.  Usage: wrap.sh <ctrl_dir> -- <command...>
# Env (from <ctrl_dir>/env.sh): SLURM_MCP_GRACE (s, 120) SLURM_MCP_ON_TIMEOUT (fail|requeue) SLURM_MCP_MAX_RESTARTS (3)
#   SLURM_MCP_CHILD_SIGNAL (unset = never forward the time-limit warning; "USR1" etc. = the payload handles it)
#   SLURM_MCP_HEARTBEAT (30) SLURM_MCP_TIMELIMIT_S (0 = unknown)
# Signals: USR1 = SLURM's --signal=B:USR1@GRACE time-limit warning (recorded; forwarded as $CSIG only when CSIG is set).
#          TERM = scancel / time limit / preemption (recorded; ALWAYS forwarded as TERM to the payload's process group).
# Files written (all atomic via .tmp + mv): status.json, heartbeat, and the program may append JSON lines to $SLURM_MCP_PROGRESS.
set -u
CTRL="$1"; shift; [ "${1:-}" = "--" ] && shift
mkdir -p "$CTRL"
TASK="${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"          # "_7" for array element 7, "" otherwise
STATUS="$CTRL/status$TASK.json"; HB="$CTRL/heartbeat$TASK"
export SLURM_MCP_CTRL="$CTRL" SLURM_MCP_PROGRESS="$CTRL/progress$TASK.json"
RESTART=${SLURM_RESTART_COUNT:-0}
GRACE=${SLURM_MCP_GRACE:-120}; ON_TIMEOUT=${SLURM_MCP_ON_TIMEOUT:-fail}; MAXR=${SLURM_MCP_MAX_RESTARTS:-3}
CSIG=${SLURM_MCP_CHILD_SIGNAL:-}; HBI=${SLURM_MCP_HEARTBEAT:-30}; LIMIT_S=${SLURM_MCP_TIMELIMIT_S:-0}
HAVE_SETSID=0; command -v setsid >/dev/null 2>&1 && HAVE_SETSID=1
CHILD=""; SIGNALED=""; CAUSE=""; FWD=""
now() { date +%s; }
signum() { kill -l "$1" 2>/dev/null || echo 0; }               # "USR1" -> 10, "TERM" -> 15 (bash builtin, no external dep)
T0=$(now)
write_status() {  # phase rc cause
  local tmp="$STATUS.tmp.$$"
  printf '{"v":2,"phase":"%s","rc":%s,"cause":"%s","restart":%s,"job_id":"%s","node":"%s","start":%s,"now":%s,"pid":%s,"signal":"%s","forwarded":"%s","limit_s":%s,"grace_s":%s}\n' \
    "$1" "${2:-null}" "${3:-}" "$RESTART" "${SLURM_JOB_ID:-}" "$(hostname -s)" "$T0" "$(now)" "${CHILD:-0}" "$SIGNALED" "$FWD" "$LIMIT_S" "$GRACE" > "$tmp" && mv -f "$tmp" "$STATUS"
}
[ -f "$CTRL/jobid" ] || { printf '%s\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" > "$CTRL/jobid.tmp.$$" && mv -f "$CTRL/jobid.tmp.$$" "$CTRL/jobid"; }
echo "=== slurm-mcp wrap: job ${SLURM_JOB_ID:-?} restart $RESTART node $(hostname -s) $(date -Is) ==="
if [ "$RESTART" -gt "$MAXR" ]; then
  write_status exited 75 max_restarts; echo "slurm-mcp: restart $RESTART exceeds max $MAXR; not running"; exit 75
fi
write_status running
( while sleep "$HBI"; do now > "$HB.tmp.$$" && mv -f "$HB.tmp.$$" "$HB"; done ) & HBPID=$!
killchild() {  # signal the whole process group of the payload (setsid) so torchrun/mpirun workers see it too
  [ -z "$CHILD" ] && return
  if [ "$HAVE_SETSID" = 1 ]; then kill -s "$1" -- "-$CHILD" 2>/dev/null; else kill -s "$1" "$CHILD" 2>/dev/null; fi
}
classify() {  # $1 = USR1|TERM -> sets CAUSE
  local el=$(( $(now) - T0 ))
  CAUSE=preempt
  if [ -f "$CTRL/cancel.requested" ]; then CAUSE=cancel
  elif [ "$1" = USR1 ]; then CAUSE=timeout                                   # only SLURM's --signal=B:USR1@GRACE sends us USR1
  elif [ "$LIMIT_S" -gt 0 ] && [ "$el" -ge $(( LIMIT_S - GRACE - 60 )) ]; then CAUSE=timeout
  fi
}
on_usr1() {  # time-limit warning: record it; forward ONLY when the payload declared it handles $CSIG
  INTR=1; SIGNALED=USR1; classify USR1; write_status signaled "" "$CAUSE"
  if [ -n "$CSIG" ]; then
    FWD=$CSIG; echo "slurm-mcp: time-limit warning (cause=$CAUSE); forwarding SIG$CSIG to child ${CHILD:-none}"; killchild "$CSIG"
  else
    echo "slurm-mcp: time-limit warning (cause=$CAUSE); no child_signal declared, payload keeps running"
  fi
  write_status signaled "" "$CAUSE"
}
on_term() {  # scancel / time limit / preemption: forward TERM as TERM, never $CSIG
  INTR=1; SIGNALED=TERM; classify TERM; FWD=TERM; write_status signaled "" "$CAUSE"
  echo "slurm-mcp: SIGTERM (cause=$CAUSE); forwarding SIGTERM to child ${CHILD:-none}"
  killchild TERM
}
trap on_usr1 USR1
trap on_term TERM
if [ "$HAVE_SETSID" = 1 ]; then setsid "$@" & else "$@" & fi
CHILD=$!; INTR=""
wait "$CHILD"; RC=$?
# A trapped signal makes `wait` return 128+signum before the trap runs; re-wait for the real status (a 127 means it was already collected)
while [ -n "$INTR" ] && [ "$RC" -gt 128 ]; do INTR=""; PREV=$RC; wait "$CHILD"; RC=$?; [ "$RC" -eq 127 ] && RC=$PREV; done
kill "$HBPID" 2>/dev/null
# A payload that died from the signal we forwarded near the limit is a timeout, not a failure (rc 128+signum, e.g. 138 for USR1, 143 for TERM)
if [ -n "$FWD" ] && [ "$RC" -eq $(( 128 + $(signum "$FWD") )) ] && [ "$CAUSE" != cancel ]; then
  if [ "$CAUSE" = preempt ] && [ "$LIMIT_S" -gt 0 ] && [ $(( $(now) - T0 )) -ge $(( LIMIT_S - GRACE - 60 )) ]; then CAUSE=timeout; fi
fi
if [ "$CAUSE" = timeout ] && [ "$ON_TIMEOUT" = requeue ] && [ "$RESTART" -lt "$MAXR" ] && [ -n "${SLURM_JOB_ID:-}" ] \
   && { [ "$RC" -ne 0 ] || [ -f "$CTRL/requeue.requested" ]; }; then
  rm -f "$CTRL/requeue.requested"
  write_status requeue "$RC" timeout
  echo "slurm-mcp: requeueing for restart $((RESTART + 1))"
  scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}" && exit 0
  echo "slurm-mcp: scontrol requeue failed; exiting $RC"
fi
write_status exited "$RC" "$CAUSE"
echo "=== slurm-mcp wrap: exit $RC ($CAUSE) $(date -Is) ==="
exit "$RC"
