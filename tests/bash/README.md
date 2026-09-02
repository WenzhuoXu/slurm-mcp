# Bash tests for the remote helpers (design section 7)

`test_helpers.sh` exercises `wrap.sh`, `submit.sh` and `alloc-agent.sh` from `src/slurm_mcp/helpers/`
with real bash, a fake `sbatch`/`scontrol` on `PATH` and short-lived payload scripts. It prints TAP-like
lines (`ok - NAME`, `not ok - NAME: reason`, `skip - NAME: reason`) and exits non-zero when anything failed.

```
bash tests/bash/test_helpers.sh                  # everything
bash tests/bash/test_helpers.sh --list           # test names
bash tests/bash/test_helpers.sh wrap_rc0 submit_ok
bash tests/bash/test_helpers.sh --helpers /path/to/helpers   # test a deployed copy (e.g. <control_root>/bin/<sha8>)
```

Options / environment: `--helpers DIR` or `SLURM_MCP_HELPERS`; `--smoke` or `SLURM_MCP_SMOKE=1` forces the
smoke subset; `TMPDIR` is where the temporary control directories go (default `/tmp`).

## Full suite (Linux)

Real Linux bash 4+ with GNU coreutils and `setsid` is the reference environment (design section 10:
"real Linux bash is mandatory"). Run it:

- locally under WSL: `wsl bash tests/bash/test_helpers.sh` (the repo is visible under `/mnt/c/...`);
- on a cluster login node: copy `tests/bash/test_helpers.sh` and the three helpers, then
  `TMPDIR=$HOME/tmp bash test_helpers.sh --helpers ./helpers` (the fake `sbatch`/`scontrol` shims are put
  first on `PATH`, so the real SLURM binaries are never called; nothing is submitted). Use `--helpers
  <control_root>/bin/<sha8>` to test exactly what the server deployed.
- on CI: a Linux job running `bash tests/bash/test_helpers.sh` plus `pytest tests/unit/test_helpers_bash.py`.

The full suite takes about 90 s: `submit_err3_lock_held` waits the scripted 30 s lock timeout and
`submit_retry_busy` the 10 s controller-busy retry.

## Smoke subset (Git Bash on Windows)

Git Bash (MSYS2) lacks `setsid`, cannot signal process groups (`kill -- -pgid`) and its `hostname` rejects
`-s`. The script detects this (`uname -o` = `Msys`, or no `setsid`) and:

- puts a tiny `hostname` shim first on `PATH` (only when `hostname -s` fails) so `status.json` gets a node name;
- skips the process-group assertions (`wrap_term_kills_group`), reported as `skip - ...`;
- keeps everything else: exit codes, `status.json` phases/causes, USR1 recorded-but-not-forwarded by default,
  USR1 forwarded with `SLURM_MCP_CHILD_SIGNAL`, TERM forwarded as TERM, `cancel.requested`, restart cap,
  requeue-on-timeout through the fake `scontrol`, `submit.sh` idempotency/lock/retry/error lines, and the
  allocation agent's foreground/background/kill/heartbeat/release/idle paths.

`tests/unit/test_helpers_bash.py` is the pytest wrapper: it locates Git's `bash.exe` on Windows (never the
WSL launcher in `System32`), parametrizes one pytest test per bash test name, and maps `skip -` lines to
`pytest.skip`. Set `SLURM_MCP_BASH=/path/to/bash` to choose the interpreter, `SLURM_MCP_SKIP_BASH_TESTS=1`
to skip the wrapper entirely.

## Platform notes

- MSYS numbers signals differently from Linux (`kill -l USR1` is 30, not 10), so a payload killed by a forwarded
  USR1 exits 158 on Git Bash and 138 on Linux; the tests only compare against what the payload itself reports.
- `wrap.sh` re-waits for the payload while a trap flag is set: bash's `wait` returns 128+signum *before* running
  the trap, and the design's `kill -0` re-wait is skipped once the child has been reaped, which leaked the
  interrupted-wait status as the payload's rc (visible on Git Bash as 158; on Linux only when the payload
  exits with a different code after handling the signal, e.g. `wrap_requeue_requested_rc0`).
- `submit_err3_lock_held` needs a writable `TMPDIR` on a filesystem where `mkdir` is atomic (any local disk or
  NFS/VAST is fine); `wsl -d <distro> -e bash tests/bash/test_helpers.sh` runs the full suite from Windows
  when a WSL distribution is installed.
