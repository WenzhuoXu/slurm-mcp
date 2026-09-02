#!/bin/bash
# Bash tests for the remote helpers of design section 7 (wrap.sh, submit.sh, alloc-agent.sh).
#
# Usage: test_helpers.sh [--list] [--helpers DIR] [--smoke] [NAME...]
#   --list       print the test names, one per line, and exit
#   --helpers    directory holding wrap.sh/submit.sh/alloc-agent.sh (default: ../../src/slurm_mcp/helpers)
#   --smoke      force the smoke subset (skip tests that need setsid / process groups)
#   NAME...      run only these tests (default: all)
# Output: TAP-like lines "ok - NAME", "not ok - NAME: reason", "skip - NAME: reason"; exit 1 if anything failed.
# Runs under real Linux bash (full suite) and under Git Bash on Windows (smoke subset: no setsid, no
# `hostname -s` -> a shim `hostname` is put on PATH, process-group assertions are skipped).  See README.md.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HELPERS=${SLURM_MCP_HELPERS:-$HERE/../../src/slurm_mcp/helpers}
SMOKE=${SLURM_MCP_SMOKE:-}
LIST=0
SELECTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST=1 ;;
    --helpers) HELPERS="$2"; shift ;;
    --smoke) SMOKE=1 ;;
    *) SELECTED+=("$1") ;;
  esac
  shift
done
TESTS=(
  wrap_rc0 wrap_rc3 wrap_array_task wrap_usr1_not_forwarded wrap_usr1_forwarded wrap_term_cancel
  wrap_term_timeout_near_limit wrap_term_preempt wrap_max_restarts wrap_requeue_on_timeout
  wrap_requeue_requested_rc0 wrap_no_requeue_when_fail wrap_term_kills_group
  submit_ok submit_idempotent submit_err1 submit_retry_busy submit_err3_lock_held submit_err2_no_ctrl
  alloc_fg alloc_bg_concurrent alloc_kill_bg_while_fg alloc_heartbeat_during_fg alloc_release alloc_idle_release
  alloc_restart_skips_started
)
if [ "$LIST" = 1 ]; then printf '%s\n' "${TESTS[@]}"; exit 0; fi

# --- environment detection --------------------------------------------------------------------------
if [ -z "$SMOKE" ]; then
  if ! command -v setsid >/dev/null 2>&1; then SMOKE=1; fi
  case "$(uname -o 2>/dev/null || uname -s)" in Msys|Cygwin|MINGW*|MSYS*) SMOKE=1 ;; esac
fi
TMP=$(mktemp -d "${TMPDIR:-/tmp}/slurm-mcp-bash.XXXXXX") || exit 2
BGPIDS=()
cleanup() {
  for p in "${BGPIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  sleep 0.2
  rm -rf "$TMP" 2>/dev/null
}
trap cleanup EXIT
SHIM="$TMP/shim"; mkdir -p "$SHIM"
if ! hostname -s >/dev/null 2>&1; then
  printf '#!/bin/bash\necho testhost\n' > "$SHIM/hostname"; chmod +x "$SHIM/hostname"
fi
export FAKE_DIR="$TMP/fake"; mkdir -p "$FAKE_DIR"
cat > "$SHIM/sbatch" <<'EOF'
#!/bin/bash
# fake sbatch: records its args; behaviour from $FAKE_DIR/mode (ok | fail | busy)
printf '%s\n' "$*" >> "$FAKE_DIR/sbatch.calls"
mode=$(cat "$FAKE_DIR/mode" 2>/dev/null || echo ok)
case "$mode" in
  ok)   echo "12345;testcluster"; exit 0 ;;
  fail) echo "sbatch: error: Batch job submission failed: Invalid partition name specified" >&2; exit 1 ;;
  busy) echo ok > "$FAKE_DIR/mode"
        echo "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation" >&2; exit 1 ;;
  *)    echo "fake sbatch: bad mode $mode" >&2; exit 2 ;;
esac
EOF
cat > "$SHIM/scontrol" <<'EOF'
#!/bin/bash
# fake scontrol: records its args (wrap.sh calls `scontrol requeue <id>`)
printf '%s\n' "$*" >> "$FAKE_DIR/scontrol.calls"
exit "${FAKE_SCONTROL_RC:-0}"
EOF
chmod +x "$SHIM/sbatch" "$SHIM/scontrol"
export PATH="$SHIM:$PATH"

# --- assertion helpers ------------------------------------------------------------------------------
PASS=0; FAIL=0; SKIP=0; CUR=""; FAILED_MSG=""
jf() { sed -n "s/.*\"$2\":\"\{0,1\}\([^,\"}]*\).*/\1/p" "$1" 2>/dev/null | head -n 1; }   # json field (flat objects only)
expect_eq() {  # what actual expected
  if [ "$2" != "$3" ]; then FAILED_MSG="${FAILED_MSG}$1: got '$2' expected '$3'; "; fi
}
expect_file() { [ -e "$1" ] || FAILED_MSG="${FAILED_MSG}missing $1; "; }
expect_nofile() { [ -e "$1" ] && FAILED_MSG="${FAILED_MSG}unexpected $1; "; return 0; }
expect_grep() { grep -q -- "$2" "$1" 2>/dev/null || FAILED_MSG="${FAILED_MSG}'$2' not in $1; "; }
wait_for() {  # file [timeout_s]
  local n=0 max=$(( ${2:-10} * 5 ))
  until [ -e "$1" ]; do n=$((n+1)); [ "$n" -ge "$max" ] && return 1; sleep 0.2; done
  return 0
}
wait_json() {  # file key value [timeout_s]
  local n=0 max=$(( ${4:-10} * 5 ))
  until [ "$(jf "$1" "$2")" = "$3" ]; do n=$((n+1)); [ "$n" -ge "$max" ] && return 1; sleep 0.2; done
  return 0
}
wait_pid_gone() {  # pid [timeout_s]
  local n=0 max=$(( ${2:-10} * 5 ))
  while kill -0 "$1" 2>/dev/null; do n=$((n+1)); [ "$n" -ge "$max" ] && return 1; sleep 0.2; done
  return 0
}
skip() { FAILED_MSG="SKIP:$1"; }
new_ctrl() { local d="$TMP/$1"; rm -rf "$d"; mkdir -p "$d"; echo "$d"; }
clear_wrap_env() { unset SLURM_MCP_GRACE SLURM_MCP_ON_TIMEOUT SLURM_MCP_MAX_RESTARTS SLURM_MCP_CHILD_SIGNAL \
                   SLURM_MCP_TIMELIMIT_S SLURM_RESTART_COUNT SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID; }

# payload scripts
PAYLOAD_TRAP="$TMP/payload_trap.sh"      # exits 138/143 when it receives USR1/TERM, 0 after 30 s otherwise
cat > "$PAYLOAD_TRAP" <<'EOF'
#!/bin/bash
sleep 30 & SP=$!
trap 'kill $SP 2>/dev/null; echo got-USR1; exit 138' USR1
trap 'kill $SP 2>/dev/null; echo got-TERM; exit 143' TERM
wait $SP
echo done
exit 0
EOF
PAYLOAD_REQUEUE="$TMP/payload_requeue.sh"  # checkpoints on USR1, asks for a requeue and exits 0
cat > "$PAYLOAD_REQUEUE" <<'EOF'
#!/bin/bash
sleep 30 & SP=$!
trap 'kill $SP 2>/dev/null; touch "$SLURM_MCP_CTRL/requeue.requested"; echo checkpointed; exit 0' USR1
wait $SP
exit 0
EOF
PAYLOAD_GROUP="$TMP/payload_group.sh"      # spawns a grandchild that must die with the process group
cat > "$PAYLOAD_GROUP" <<'EOF'
#!/bin/bash
sleep 60 & echo $! > "$SLURM_MCP_CTRL/grandchild.pid"
wait
EOF

# run wrap.sh in the background; sets WPID and OUTLOG
start_wrap() {  # ctrl payload...
  local ctrl="$1"; shift
  OUTLOG="$ctrl/wrap.log"
  bash "$HELPERS/wrap.sh" "$ctrl" -- "$@" > "$OUTLOG" 2>&1 &
  WPID=$!; BGPIDS+=("$WPID")
}

# --- wrap.sh -----------------------------------------------------------------------------------------
t_wrap_rc0() {
  local ctrl; ctrl=$(new_ctrl wrap_rc0)
  SLURM_JOB_ID=4242 SLURM_MCP_HEARTBEAT=1 bash "$HELPERS/wrap.sh" "$ctrl" -- bash -c 'echo hello-payload; exit 0' > "$ctrl/wrap.log" 2>&1
  expect_eq rc "$?" 0
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 0
  expect_eq cause "$(jf "$ctrl/status.json" cause)" ""
  expect_eq job_id "$(jf "$ctrl/status.json" job_id)" 4242
  expect_eq v "$(jf "$ctrl/status.json" v)" 2
  expect_eq jobid_file "$(cat "$ctrl/jobid")" 4242
  expect_grep "$ctrl/wrap.log" "hello-payload"
  expect_grep "$ctrl/wrap.log" "=== slurm-mcp wrap: exit 0"
}
t_wrap_rc3() {
  local ctrl; ctrl=$(new_ctrl wrap_rc3)
  SLURM_JOB_ID=4243 bash "$HELPERS/wrap.sh" "$ctrl" -- bash -c 'exit 3' > "$ctrl/wrap.log" 2>&1
  expect_eq rc "$?" 3
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 3
}
t_wrap_array_task() {
  local ctrl; ctrl=$(new_ctrl wrap_array)
  SLURM_JOB_ID=4300 SLURM_ARRAY_JOB_ID=4299 SLURM_ARRAY_TASK_ID=7 bash "$HELPERS/wrap.sh" "$ctrl" -- true > "$ctrl/wrap.log" 2>&1
  expect_eq rc "$?" 0
  expect_file "$ctrl/status_7.json"
  expect_nofile "$ctrl/status.json"
  expect_eq jobid_file "$(cat "$ctrl/jobid")" 4299
}
t_wrap_usr1_not_forwarded() {
  local ctrl; ctrl=$(new_ctrl wrap_usr1_nf)
  SLURM_JOB_ID=1 SLURM_MCP_HEARTBEAT=1 start_wrap "$ctrl" bash -c 'trap "echo payload-got-usr1; exit 99" USR1; sleep 3; echo done; exit 0'
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.5; kill -USR1 "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 0
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 0
  expect_eq signal "$(jf "$ctrl/status.json" signal)" USR1
  expect_eq forwarded "$(jf "$ctrl/status.json" forwarded)" ""
  expect_eq cause "$(jf "$ctrl/status.json" cause)" timeout
  expect_grep "$OUTLOG" "^done"
  expect_grep "$OUTLOG" "no child_signal declared"
  if grep -q "payload-got-usr1" "$OUTLOG"; then FAILED_MSG="${FAILED_MSG}USR1 reached the payload; "; fi
}
t_wrap_usr1_forwarded() {
  local ctrl; ctrl=$(new_ctrl wrap_usr1_fwd)
  SLURM_JOB_ID=1 SLURM_MCP_CHILD_SIGNAL=USR1 SLURM_MCP_TIMELIMIT_S=3600 start_wrap "$ctrl" bash "$PAYLOAD_TRAP"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.5; kill -USR1 "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 138
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 138
  expect_eq signal "$(jf "$ctrl/status.json" signal)" USR1
  expect_eq forwarded "$(jf "$ctrl/status.json" forwarded)" USR1
  expect_eq cause "$(jf "$ctrl/status.json" cause)" timeout
  expect_grep "$OUTLOG" "got-USR1"
  expect_grep "$OUTLOG" "forwarding SIGUSR1"
}
t_wrap_term_cancel() {
  local ctrl; ctrl=$(new_ctrl wrap_term_cancel)
  SLURM_JOB_ID=1 SLURM_MCP_CHILD_SIGNAL=USR1 start_wrap "$ctrl" bash "$PAYLOAD_TRAP"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  touch "$ctrl/cancel.requested"
  sleep 0.3; kill -TERM "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 143
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_eq signal "$(jf "$ctrl/status.json" signal)" TERM
  expect_eq forwarded "$(jf "$ctrl/status.json" forwarded)" TERM       # TERM is forwarded as TERM, never $CSIG
  expect_eq cause "$(jf "$ctrl/status.json" cause)" cancel
  expect_grep "$OUTLOG" "got-TERM"
}
t_wrap_term_timeout_near_limit() {
  local ctrl; ctrl=$(new_ctrl wrap_term_timeout)
  SLURM_JOB_ID=1 SLURM_MCP_TIMELIMIT_S=30 SLURM_MCP_GRACE=0 start_wrap "$ctrl" bash "$PAYLOAD_TRAP"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.3; kill -TERM "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 143
  expect_eq cause "$(jf "$ctrl/status.json" cause)" timeout          # elapsed >= LIMIT - GRACE - 60
  expect_eq forwarded "$(jf "$ctrl/status.json" forwarded)" TERM
  expect_eq limit_s "$(jf "$ctrl/status.json" limit_s)" 30
}
t_wrap_term_preempt() {
  local ctrl; ctrl=$(new_ctrl wrap_term_preempt)
  SLURM_JOB_ID=1 SLURM_MCP_TIMELIMIT_S=0 start_wrap "$ctrl" bash "$PAYLOAD_TRAP"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.3; kill -TERM "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 143
  expect_eq cause "$(jf "$ctrl/status.json" cause)" preempt            # no limit known, no cancel file
}
t_wrap_max_restarts() {
  local ctrl; ctrl=$(new_ctrl wrap_maxr)
  SLURM_JOB_ID=1 SLURM_RESTART_COUNT=4 SLURM_MCP_MAX_RESTARTS=3 bash "$HELPERS/wrap.sh" "$ctrl" -- bash -c 'echo RAN; exit 0' > "$ctrl/wrap.log" 2>&1
  expect_eq rc "$?" 75
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 75
  expect_eq cause "$(jf "$ctrl/status.json" cause)" max_restarts
  expect_eq restart "$(jf "$ctrl/status.json" restart)" 4
  if grep -q "^RAN" "$ctrl/wrap.log"; then FAILED_MSG="${FAILED_MSG}payload ran past max_restarts; "; fi
}
t_wrap_requeue_on_timeout() {
  local ctrl; ctrl=$(new_ctrl wrap_requeue)
  rm -f "$FAKE_DIR/scontrol.calls"
  SLURM_JOB_ID=4242 SLURM_MCP_CHILD_SIGNAL=USR1 SLURM_MCP_ON_TIMEOUT=requeue SLURM_MCP_MAX_RESTARTS=3 start_wrap "$ctrl" bash "$PAYLOAD_TRAP"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.5; kill -USR1 "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 0                                                  # scontrol requeue succeeded => exit 0
  expect_eq phase "$(jf "$ctrl/status.json" phase)" requeue
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 138
  expect_eq cause "$(jf "$ctrl/status.json" cause)" timeout
  expect_eq scontrol "$(cat "$FAKE_DIR/scontrol.calls" 2>/dev/null)" "requeue 4242"
}
t_wrap_requeue_requested_rc0() {
  local ctrl; ctrl=$(new_ctrl wrap_requeue_req)
  rm -f "$FAKE_DIR/scontrol.calls"
  SLURM_JOB_ID=4242 SLURM_MCP_CHILD_SIGNAL=USR1 SLURM_MCP_ON_TIMEOUT=requeue start_wrap "$ctrl" bash "$PAYLOAD_REQUEUE"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.5; kill -USR1 "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 0
  expect_eq phase "$(jf "$ctrl/status.json" phase)" requeue
  expect_eq rc_field "$(jf "$ctrl/status.json" rc)" 0
  expect_nofile "$ctrl/requeue.requested"                               # consumed
  expect_eq scontrol "$(cat "$FAKE_DIR/scontrol.calls" 2>/dev/null)" "requeue 4242"
}
t_wrap_no_requeue_when_fail() {  # on_timeout=fail: a forwarded-signal death is reported, never requeued
  local ctrl; ctrl=$(new_ctrl wrap_norequeue)
  rm -f "$FAKE_DIR/scontrol.calls"
  SLURM_JOB_ID=4242 SLURM_MCP_CHILD_SIGNAL=USR1 SLURM_MCP_ON_TIMEOUT=fail start_wrap "$ctrl" bash "$PAYLOAD_TRAP"
  wait_json "$ctrl/status.json" phase running 10 || FAILED_MSG="never running; "
  sleep 0.5; kill -USR1 "$WPID"
  wait "$WPID"; local rc=$?
  expect_eq rc "$rc" 138
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
  expect_nofile "$FAKE_DIR/scontrol.calls"
}
t_wrap_term_kills_group() {  # needs setsid: the forwarded TERM reaches the payload's grandchildren
  if [ -n "$SMOKE" ]; then skip "needs setsid / process groups (not on Git Bash)"; return; fi
  local ctrl; ctrl=$(new_ctrl wrap_group)
  SLURM_JOB_ID=1 start_wrap "$ctrl" bash "$PAYLOAD_GROUP"
  wait_for "$ctrl/grandchild.pid" 10 || FAILED_MSG="grandchild never started; "
  sleep 0.3; kill -TERM "$WPID"
  wait "$WPID"
  local gc; gc=$(cat "$ctrl/grandchild.pid")
  wait_pid_gone "$gc" 5 || FAILED_MSG="${FAILED_MSG}grandchild $gc survived the forwarded TERM; "
  expect_eq forwarded "$(jf "$ctrl/status.json" forwarded)" TERM
}

# --- submit.sh ---------------------------------------------------------------------------------------
t_submit_ok() {
  local ctrl; ctrl=$(new_ctrl submit_ok); echo ok > "$FAKE_DIR/mode"; rm -f "$FAKE_DIR/sbatch.calls"
  local out; out=$(bash "$HELPERS/submit.sh" "$ctrl" t-abc -- -p batch --comment=slurm-mcp:j1:1:t-abc "$ctrl/job.sbatch")
  expect_eq rc "$?" 0
  expect_eq first "$(printf '%s\n' "$out" | head -n 1)" "JOBID 12345"
  expect_eq jobid "$(cat "$ctrl/jobid")" 12345                          # ";cluster" stripped
  expect_eq calls "$(cat "$FAKE_DIR/sbatch.calls")" "--parsable -p batch --comment=slurm-mcp:j1:1:t-abc $ctrl/job.sbatch"
  expect_nofile "$ctrl/.submit.lock"
}
t_submit_idempotent() {
  local ctrl; ctrl=$(new_ctrl submit_idem); echo ok > "$FAKE_DIR/mode"; rm -f "$FAKE_DIR/sbatch.calls"
  bash "$HELPERS/submit.sh" "$ctrl" t-abc -- "$ctrl/job.sbatch" > /dev/null
  local out; out=$(bash "$HELPERS/submit.sh" "$ctrl" t-abc -- "$ctrl/job.sbatch")
  expect_eq first "$(printf '%s\n' "$out" | head -n 1)" "JOBID 12345"
  expect_eq ncalls "$(wc -l < "$FAKE_DIR/sbatch.calls" | tr -d ' ')" 1   # sbatch not invoked again
}
t_submit_err1() {
  local ctrl; ctrl=$(new_ctrl submit_err1); echo fail > "$FAKE_DIR/mode"
  local out; out=$(bash "$HELPERS/submit.sh" "$ctrl" t-abc -- "$ctrl/job.sbatch")
  expect_eq rc "$?" 0                                                   # always exits 0; the caller parses line 1
  expect_eq first "$(printf '%s\n' "$out" | head -n 1)" "ERR 1"
  expect_eq second "$(printf '%s\n' "$out" | sed -n 2p)" "sbatch: error: Batch job submission failed: Invalid partition name specified"
  expect_nofile "$ctrl/jobid"
  expect_nofile "$ctrl/.submit.lock"
}
t_submit_retry_busy() {
  local ctrl; ctrl=$(new_ctrl submit_busy); echo busy > "$FAKE_DIR/mode"; rm -f "$FAKE_DIR/sbatch.calls"
  local t0; t0=$(date +%s)
  local out; out=$(bash "$HELPERS/submit.sh" "$ctrl" t-abc -- "$ctrl/job.sbatch")
  local el=$(( $(date +%s) - t0 ))
  expect_eq first "$(printf '%s\n' "$out" | head -n 1)" "JOBID 12345"
  expect_eq ncalls "$(wc -l < "$FAKE_DIR/sbatch.calls" | tr -d ' ')" 2
  [ "$el" -ge 9 ] || FAILED_MSG="${FAILED_MSG}retry came after ${el}s, expected >= 10; "
}
t_submit_err3_lock_held() {
  local ctrl; ctrl=$(new_ctrl submit_lock); echo ok > "$FAKE_DIR/mode"; rm -f "$FAKE_DIR/sbatch.calls"
  mkdir "$ctrl/.submit.lock"
  local out; out=$(bash "$HELPERS/submit.sh" "$ctrl" t-abc -- "$ctrl/job.sbatch")
  expect_eq first "$(printf '%s\n' "$out" | head -n 1)" "ERR 3"
  expect_eq second "$(printf '%s\n' "$out" | sed -n 2p)" "lock timeout"
  expect_file "$ctrl/.submit.lock"                                      # never removes a lock it did not create
  expect_nofile "$FAKE_DIR/sbatch.calls"
}
t_submit_err2_no_ctrl() {
  local out; out=$(bash "$HELPERS/submit.sh" "$TMP/does-not-exist" t-abc -- x.sbatch)
  expect_eq first "$(printf '%s\n' "$out" | head -n 1)" "ERR 2"
}

# --- alloc-agent.sh ----------------------------------------------------------------------------------
start_agent() {  # ctrl [idle]
  AGENTLOG="$1/agent.log"
  SLURM_JOB_ID=777 bash "$HELPERS/alloc-agent.sh" "$1" "${2:-0}" > "$AGENTLOG" 2>&1 &
  APID=$!; BGPIDS+=("$APID")
}
stop_agent() { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; }
write_cmd() {  # path body
  printf '#!/bin/bash\n# slurm-mcp cmd=test\nset -o pipefail\n%s\n' "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"
}
t_alloc_fg() {
  local ctrl; ctrl=$(new_ctrl alloc_fg); mkdir -p "$ctrl/cmds"
  start_agent "$ctrl"
  wait_file_ok=1; wait_for "$ctrl/status.json" 10 || wait_file_ok=0
  [ "$wait_file_ok" = 1 ] || FAILED_MSG="no status.json; "
  write_cmd "$ctrl/cmds/001.sh" 'echo hello-fg; exit 4'
  wait_for "$ctrl/cmds/001.done" 10 || FAILED_MSG="${FAILED_MSG}001.done never appeared; "
  expect_eq rc "$(cat "$ctrl/cmds/001.rc" 2>/dev/null)" 4
  expect_grep "$ctrl/cmds/001.out" hello-fg
  expect_file "$ctrl/cmds/001.pid"; expect_file "$ctrl/cmds/001.started"
  sleep 1.2
  expect_eq phase "$(jf "$ctrl/status.json" phase)" running
  expect_eq running "$(jf "$ctrl/status.json" running)" 0
  expect_eq job_id "$(jf "$ctrl/status.json" job_id)" 777
  stop_agent "$APID"
}
t_alloc_bg_concurrent() {
  local ctrl; ctrl=$(new_ctrl alloc_bg); mkdir -p "$ctrl/cmds"
  start_agent "$ctrl"
  wait_for "$ctrl/status.json" 10 || FAILED_MSG="no status.json; "
  write_cmd "$ctrl/cmds/002.bg.sh" 'sleep 4; echo bg-done'
  write_cmd "$ctrl/cmds/003.sh" 'echo fg-done'
  wait_for "$ctrl/cmds/003.done" 10 || FAILED_MSG="${FAILED_MSG}003.done never appeared; "
  expect_nofile "$ctrl/cmds/002.bg.rc"                                  # bg still running while fg finished
  expect_file "$ctrl/cmds/002.bg.pid"
  wait_for "$ctrl/cmds/002.bg.done" 10 || FAILED_MSG="${FAILED_MSG}002.bg.done never appeared; "
  expect_eq bg_rc "$(cat "$ctrl/cmds/002.bg.rc" 2>/dev/null)" 0
  expect_grep "$ctrl/cmds/002.bg.out" bg-done
  stop_agent "$APID"
}
t_alloc_kill_bg_while_fg() {
  local ctrl; ctrl=$(new_ctrl alloc_kill); mkdir -p "$ctrl/cmds"
  start_agent "$ctrl"
  wait_for "$ctrl/status.json" 10 || FAILED_MSG="no status.json; "
  write_cmd "$ctrl/cmds/001.sh" 'sleep 6; echo fg-done'                 # a long foreground command
  write_cmd "$ctrl/cmds/002.bg.sh" 'sleep 60; echo never'
  wait_for "$ctrl/cmds/002.bg.pid" 10 || FAILED_MSG="${FAILED_MSG}002.bg never started; "
  local t0; t0=$(date +%s)
  touch "$ctrl/cmds/002.bg.kill"
  wait_for "$ctrl/cmds/002.bg.rc" 5 || FAILED_MSG="${FAILED_MSG}002.bg.rc not written within 5 s of .kill; "
  local el=$(( $(date +%s) - t0 ))
  [ "$el" -le 3 ] || FAILED_MSG="${FAILED_MSG}kill honoured after ${el}s (> 2 s); "
  expect_nofile "$ctrl/cmds/002.bg.kill"                                # consumed
  local rc; rc=$(cat "$ctrl/cmds/002.bg.rc" 2>/dev/null)
  [ "$rc" != "0" ] || FAILED_MSG="${FAILED_MSG}killed command reported rc 0; "
  expect_nofile "$ctrl/cmds/001.rc"                                     # the fg command is still running
  wait_for "$ctrl/cmds/001.done" 12 || FAILED_MSG="${FAILED_MSG}001.done never appeared; "
  expect_eq fg_rc "$(cat "$ctrl/cmds/001.rc" 2>/dev/null)" 0
  stop_agent "$APID"
}
t_alloc_heartbeat_during_fg() {
  local ctrl; ctrl=$(new_ctrl alloc_hb); mkdir -p "$ctrl/cmds"
  start_agent "$ctrl"
  wait_for "$ctrl/heartbeat" 10 || FAILED_MSG="no heartbeat; "
  write_cmd "$ctrl/cmds/001.sh" 'sleep 4'
  wait_for "$ctrl/cmds/001.pid" 10 || FAILED_MSG="${FAILED_MSG}001 never started; "
  local h1; h1=$(cat "$ctrl/heartbeat"); sleep 2.5
  local h2; h2=$(cat "$ctrl/heartbeat")
  [ "$h2" -gt "$h1" ] || FAILED_MSG="${FAILED_MSG}heartbeat not refreshed during a fg command ($h1 -> $h2); "
  expect_eq fg "$(jf "$ctrl/status.json" fg)" 001
  stop_agent "$APID"
}
t_alloc_release() {
  local ctrl; ctrl=$(new_ctrl alloc_release); mkdir -p "$ctrl/cmds"
  start_agent "$ctrl"
  wait_for "$ctrl/status.json" 10 || FAILED_MSG="no status.json; "
  touch "$ctrl/release"
  wait_pid_gone "$APID" 5 || FAILED_MSG="${FAILED_MSG}agent did not exit on release; "
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
}
t_alloc_idle_release() {
  local ctrl; ctrl=$(new_ctrl alloc_idle); mkdir -p "$ctrl/cmds"
  start_agent "$ctrl" 2
  wait_pid_gone "$APID" 8 || FAILED_MSG="${FAILED_MSG}agent did not idle-release; "
  expect_grep "$AGENTLOG" "idle release"
  expect_eq phase "$(jf "$ctrl/status.json" phase)" exited
}
t_alloc_restart_skips_started() {  # a restarted agent never re-runs a command that has a .pid file
  local ctrl; ctrl=$(new_ctrl alloc_restart); mkdir -p "$ctrl/cmds"
  write_cmd "$ctrl/cmds/001.sh" 'echo ran-once'
  echo 99999999 > "$ctrl/cmds/001.pid"                                  # started by a previous incarnation
  write_cmd "$ctrl/cmds/002.sh" 'echo second'
  start_agent "$ctrl"
  wait_for "$ctrl/cmds/002.done" 10 || FAILED_MSG="${FAILED_MSG}002.done never appeared; "
  expect_nofile "$ctrl/cmds/001.out"
  expect_grep "$ctrl/cmds/002.out" second
  stop_agent "$APID"
}

# --- runner ------------------------------------------------------------------------------------------
run_one() {
  CUR="$1"; FAILED_MSG=""
  clear_wrap_env
  "t_$1"
  case "$FAILED_MSG" in
    "")     PASS=$((PASS+1)); echo "ok - $1" ;;
    SKIP:*) SKIP=$((SKIP+1)); echo "skip - $1: ${FAILED_MSG#SKIP:}" ;;
    *)      FAIL=$((FAIL+1)); echo "not ok - $1: $FAILED_MSG" ;;
  esac
}
if [ ${#SELECTED[@]} -gt 0 ]; then
  for t in "${SELECTED[@]}"; do run_one "$t"; done
else
  for t in "${TESTS[@]}"; do run_one "$t"; done
fi
echo "# passed=$PASS failed=$FAIL skipped=$SKIP smoke=${SMOKE:-0}"
[ "$FAIL" -eq 0 ]
