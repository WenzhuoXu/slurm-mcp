#!/bin/bash
# slurm-mcp submit.sh v1.  Usage: submit.sh <ctrl_dir> <token> -- <sbatch args...> <script>
# Output: first line "JOBID <id>" or "ERR <rc>", then sbatch's stderr.  Always exits 0 (the caller parses the first line).
CTRL="$1"; TOKEN="$2"; shift 2; [ "${1:-}" = "--" ] && shift
export SLURM_TIME_FORMAT=%s LC_ALL=C
[ -d "$CTRL" ] || { echo "ERR 2"; echo "no ctrl dir $CTRL"; exit 0; }
if [ -s "$CTRL/jobid" ]; then echo "JOBID $(cat "$CTRL/jobid")"; exit 0; fi       # an earlier (lost) call already submitted
n=0; until mkdir "$CTRL/.submit.lock" 2>/dev/null; do n=$((n+1)); [ "$n" -ge 30 ] && { echo "ERR 3"; echo "lock timeout"; exit 0; }; sleep 1; done
trap 'rmdir "$CTRL/.submit.lock" 2>/dev/null' EXIT
if [ -s "$CTRL/jobid" ]; then echo "JOBID $(cat "$CTRL/jobid")"; exit 0; fi
for try in 1 2 3; do
  OUT=$(sbatch --parsable "$@" 2>"$CTRL/submit.err"); RC=$?
  if [ $RC -eq 0 ] && [ -n "$OUT" ]; then
    printf '%s\n' "${OUT%%;*}" > "$CTRL/jobid.tmp.$$" && mv -f "$CTRL/jobid.tmp.$$" "$CTRL/jobid"
    echo "JOBID ${OUT%%;*}"; cat "$CTRL/submit.err"; exit 0                        # warnings, if any, follow
  fi
  if grep -qiE 'socket timed out|unable to contact|zero bytes' "$CTRL/submit.err"; then sleep 10; continue; fi
  break
done
echo "ERR $RC"; cat "$CTRL/submit.err"; exit 0
