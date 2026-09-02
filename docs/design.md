# slurm-mcp — final design

Status: **authoritative implementation contract** (supersedes the three panel drafts). Every command string, field name, dataclass field, SQLite column, tool name and module path in this document is the contract that implementers working in parallel must follow. Where this document and a research note disagree, the captured fixtures in `tests/fixtures/{trace,bridges2}/` win, then `research_6_slurm_cli.md`, then this document's stated rationale.

Conventions: `RO` = read-only tool (`_mcp.read_only()`), `W` = mutating non-destructive (`_mcp.mutating()`), `D` = destructive (`_mcp.destructive()`). Handles: jobs `j17`, allocations `a3`, allocation commands `a3.c2`, plans `p9`, transfers `t4`, array elements `j18[7]`, attempts `j17/a2`. All remote paths are absolute. All timestamps stored for SLURM events are **cluster epoch seconds** (see §6.0); local timestamps are labelled `*_local`.

## 1. Overview and principles

slurm-mcp is one local Python process (`mcp>=2.1,<3` `MCPServer` via `slurm_mcp._mcp`, asyncio, asyncssh 2.24) that Claude Code talks to over stdio. It keeps one SSH connection per cluster host (login host, plus the transfer host when the profile has one), a SQLite ledger (`~/.slurm-mcp/state.db`, WAL), and a background Monitor that reconciles the ledger against SLURM once per cluster per 30–120 s with **one composite command per tick**. Every user-visible object has a short stable handle that survives rebalancing (SLURM ids change; handles do not), server restarts and cluster moves.

| Requirement | Calls the agent makes |
|---|---|
| 1 Provision | `submit_job(...)` for batch; `allocate(...)` → `alloc_run("a3", "nvidia-smi")` for a reusable node |
| 2 Send jobs | `submit_job(job=spec)` (uploads `inputs`, renders, submits, auto-places); arrays and dependencies are spec fields |
| 3 Monitor | `list_jobs()`, `job_status(["j17"])`, `job_logs("j17")`, `cluster_status("trace")` |
| 4 Balance | `plan_job(spec)` → ranked table; `submit_job(plan_id="p9")`; `rebalance()` proposes, `rebalance(dry_run=False)` applies; `configure(placement=...)` |
| 5 Notify | `wait_for_events(timeout_s=300)` long-poll (Claude Code backgrounds it after 2 min and delivers the result as a task notification); deliver-then-ack cursor per client so a lost result is replayed; toast/webhook/SLURM mail for the human; every response carries `unread_events` |
| 6 Collect | `collect_results(["j17"])` (spec.outputs, incremental) or `download(cluster, globs, local_dir)` |
| 7 General | profiles in `~/.slurm-mcp/config.json`; capabilities discovered at first connect and cached 24 h; no cluster name appears in code |

Design rules (each is enforced by a test in §10):

1. **Structured output only.** Claude Code forwards only `structuredContent`, so every tool returns a Pydantic model with `summary: str` first, `unread_events: int`, and `next: str | None` (what to call next). No tool returns a bare `list`/`str`. Failures are `ToolError` with text `E_CODE: what happened — fix: what to do`; never a success payload carrying an error.
2. **Bounded outputs.** Defaults: 50 rows per list, 80 log lines / 12 KB per read, 50 events per wait, 6 placement options. Large tools declare `meta={"anthropic/maxResultSizeChars": 60000}`. Paging is explicit (`offset`/`next_offset`, `cursor`).
3. **Write the intent before the side effect; make every side effect idempotent or recoverable.** A submit is an `attempts` row with a unique token *before* `sbatch`; the row moves to `UNCONFIRMED` *before* `submit.sh` is invoked (so `INTENT` always means "sbatch was never run"); `sbatch` runs through a remote `submit.sh` that records the job id atomically so a retry returns the same id; the per-attempt script path (`%o`/`Command`), the comment (`%k`) and `SubmitLine` identify the attempt so a lost reply is recovered from the live queue first and from `sacct` later. `sbatch` is never blindly retried by the transport. Anything ambiguous (lock timeout, missing first line, dropped channel) is `UNCONFIRMED`, never `FAILED`.
4. **The cluster is the source of truth for state; the ledger is the source of truth for identity.** Terminal state comes from `sacct` (authoritative), liveness from `squeue -t all` (fresh; valid `MinJobAge` after end), restart counts from `squeue -O RestartCnt` while in memory and from `sacct -D` row counts afterwards, exit codes from the wrapper's `status.json` when slurmdbd lags. Precedence rules are explicit (§5.2), never "last write wins".
5. **No laptop-clock comparisons.** Every remote command exports `SLURM_TIME_FORMAT=%s LC_ALL=C` and prints the cluster's own `date +%s`; all SLURM timestamps are parsed as cluster epoch; if a cluster ignores `SLURM_TIME_FORMAT` (probed at bootstrap), ISO strings are converted with the discovered `date +%z` offset. Local time is used only for local things (tool timeouts, backoff).
6. **Login nodes are shared and tiny.** One composite command per tick, remote aggregation with `sort | uniq -c`, no per-job process, no long-lived `tail -f`/`srun` channels, ≤ 6 concurrent exec channels per connection, at most 4 `sbatch --test-only` per placement decision, adaptive backoff when nothing changes. A probe whose output lacks the `::END` sentinel is discarded whole.
7. **Long operations outlive the tool call.** Submits (upload → render → place → sbatch), transfers, allocation commands and rebalances are server-side tasks with handles; the tool waits up to `wait_s` with progress and otherwise returns the handle with a non-terminal `state`; completion arrives as a durable event that `wait_for_events` long-polls. A client abort never cancels the server-side task. Restart is a normal event: everything needed to resume is in SQLite and the first tick after start is a full reconcile.
9. **One writer per ledger, fenced.** Exactly one process runs the Monitor at a time: a SQLite lease row with a fencing token, renewed inside `BEGIN IMMEDIATE`; every state-writing transaction re-checks the token and a Monitor that lost it stops. Handles and sequence numbers are allocated inside `BEGIN IMMEDIATE`, never under an in-process lock alone.
10. **Never kill the payload with our own signal.** The time-limit warning (`--signal=B:USR1@grace`) is delivered to `wrap.sh` only; it is forwarded to the payload only when the spec declares `child_signal` ("my program handles SIG<x>"). `scancel`'s TERM is forwarded as TERM. A job that ends from a forwarded signal near the limit is classified `TIMEOUT`, never `FAILED`.
8. **Human parity.** Every tool is also `slurm-mcp <tool> [args]` through the same `Service` facade and SQLite file, so a human can inspect and continue what the agent did.

Deviations from the brief's proposed architecture: 20 tools instead of ~25 (`bootstrap_cluster`, `list_events`, `ack_events`, `sync_project`, `submit_script`, `hold/release/requeue_job`, `run_in_allocation`/`release_allocation`, `set_policy`, `configure_notifications`, `transfer_status` are folded into neighbours, §4); allocations are a file-queue agent inside a sleeper batch job rather than `srun --jobid --overlap` (§5.7, §7); `probe.sh`+`jq` are replaced by a Python-built composite command with `::SECTION`/`::END` framing (§6.2); racing duplicates are excluded from v1 (§11b); the remote helper bundle is three POSIX-bash files deployed to a content-addressed directory (§7).

## 2. Module layout

```
src/slurm_mcp/
  __init__.py
  _mcp.py             existing; ADD: `from mcp.server.mcpserver.exceptions import ToolError` re-export (fallback for 1.x: mcp.server.fastmcp.exceptions)
  config.py           existing ClusterProfile + new fields (below); load/save unchanged
  credentials.py      existing (unchanged)
  transport.py        existing SSHTransport, extended (§2.2): password callable off-thread, client_keys=None, idempotent flag on run(),
                      run_with_stdin_file(), run_to_file(), host-key store with per-pool acceptance, tcp_probe(), banner_probe(), per-host instances
  errors.py           SlurmMcpError(code, message, fix) -> ToolError text "E_CODE: message — fix: fix"; the E_* catalogue (§9.1)
  clock.py            ClusterClock: remote_now(), to_epoch(str) using caps.epoch_format / tz_offset_s, age_s()
  models.py           Pydantic: Resources, JobSpec, InputSpec, Target, PlacementPolicy, NotifyPolicy, RebalancePolicy, all tool result models (§3, §4)
  store.py            SQLite schema (PRAGMA user_version ladder), DAO, fenced monitor lease (§3.3/§5.8); every write is a BEGIN IMMEDIATE
                      transaction run via asyncio.to_thread (the in-process asyncio.Lock only serialises this process; cross-process
                      safety comes from SQLite's writer lock + the lease token); handle allocation inside the same transaction
  events.py           EventBus: append(), read(since_seq, filters), per-client deliver-then-ack cursors (§5.6), asyncio.Condition wake-ups
  submitter.py        SubmitTask: the background submit pipeline (§5.1) with per-handle asyncio tasks, progress, INTENT/UNCONFIRMED bookkeeping
  service.py          Service facade: every business operation (used by server tools AND cli); owns ClusterRegistry, Monitor, TransferManager
  server.py           MCPServer instance, 20 tool functions (validate -> Service -> model), prompts, resources, INSTRUCTIONS, lifespan
  slurm/commands.py   pure builders returning exact command strings (§6); unit-tested against the strings in this document
  slurm/parse.py      pure parsers for squeue/sacct/sinfo/scontrol/sacctmgr/sshare/sbatch/df/tools output; golden-tested on tests/fixtures
  slurm/states.py     JobState enum, SLURM state maps, terminal sets, transition(old, new) validation, reason classification
  slurm/client.py     SlurmClient(cluster): typed ops (submit, test_only, tick, snapshot, cancel, hold, release, requeue, enrich) over SSHTransport
  slurm/discovery.py  bootstrap: capability discovery (§6.1), helper deploy/verify, caps cache in kv, 30-day wait-history back-fill
  render.py           JobSpec -> job.sbatch/user_body.sh/env.sh; `#SBATCH` parser that *strips* every directive the server re-emits (§6.3);
                      Target -> sbatch CLI args; QOS selection; text normalisation (CRLF/BOM/NUL, §5.5); -o/-e pattern expansion
  textio.py           normalize_text(), read_local_text() (utf-8-sig, CRLF -> LF, refuse NUL), posix_rel(), local_safe_name() (NTFS rules), long_path()
  monitor.py          Monitor: per-cluster tick loop, reconciliation state machine (§5.2), event emission, cancel-grace, rebalance scheduling
  placer.py           candidates -> feasibility -> estimates -> score -> ranked options; rebalance proposals; circuit breaker; maintenance windows
  transfer.py         TransferManager: manifests, ignore rules, tar-stream vs SFTP, quota check, resumable per-file rows, background tasks
  alloc.py            allocation protocol: cmd files, rc/out polling, kill, release
  notify.py           toast (win11toast in a thread), webhook (urllib in a thread), SLURM mail option rendering, quiet hours
  helpers/wrap.sh, helpers/submit.sh, helpers/alloc-agent.sh    package data (§7), deployed to <control_root>/bin/<sha8>/
  testing/fake_slurm.py, testing/fake_transport.py               fake cluster + transport (§10); `slurm-mcp serve --fake`
  testing/stubs/{squeue,sacct,sinfo,scontrol,sbatch,scancel,sacctmgr,sshare,date}   stub executables for real-bash tests (§10)
tests/
  fixtures/{trace,bridges2}/   real captured outputs (index.json maps file -> command); parser goldens
  unit/ (parse, commands, render, placer, transfer, states, textio)   scenario/ (monitor on FakeSlurm)   contract/ (in-memory mcp.Client)
  bash/ (helpers + composite commands under real Linux bash with the stub binaries)   ssh/ (in-process asyncssh server: host-key pool)
```

Dependency direction: `server -> service -> {submitter, monitor, placer, transfer, alloc, notify} -> slurm/client -> transport`; `slurm/{commands,parse,states}`, `models`, `errors`, `clock`, `textio` import nothing from the package except each other. `monitor.py` is the only writer of `jobs.state` for live/terminal SLURM states; `submitter.py` writes only the pre-SLURM states (`QUEUED`, `UPLOADING`, `SUBMITTING`, `SUBMITTED`, and `FAILED` for definite submit errors); tools write intents only (`cancel_requested_ts`, new `attempts` rows in state `INTENT`, `hold_reason`). Every writer runs inside a fenced transaction (§3.3 `lease`).

### 2.1 `ClusterProfile` additions (config.py)

```python
@dataclass
class ClusterProfile:                    # existing fields kept: name, host, user, port, auth, key_path, data_host, remote_root,
                                         # default_account, default_partition, extra
    transfer_host: str | None = None     # alias of data_host (data_host kept for backward compat; transfer_host wins when both set)
    transfer_port: int | None = None     # None -> discovered by banner probe (22, then 2222 if 22 lacks "hpn"); stored in caps
    control_root: str | None = None      # default: f"{remote_root}/.slurm-mcp" if remote_root else "$HOME/.slurm-mcp"
    partition_groups: list[list[str]] = field(default_factory=list)   # partitions that may be comma-joined in one -p (same charge model & limits)
    qos_map: dict[str, str] = field(default_factory=dict)             # partition -> qos override; else discovered (§6.1 QOS selection)
    no_mem_flag: list[str] = field(default_factory=list)              # partitions where --mem must not be sent (e.g. ["RM-shared"])
    su_rates: dict[str, float] = field(default_factory=dict)          # {"gpu:h100-80": 2, "gpu:*": 1, "cpu": 1} SU per unit-hour; used when no TRESBillingWeights
    balance_command: str | None = None   # e.g. "projects"; balance_regex extracts SU numbers when sshare has no GrpTRESMins
    balance_regex: str | None = None     # e.g. r"(?P<left>[\d,]+)\s*/\s*(?P<total>[\d,]+)\s*SU"
    quota_command: str | None = None     # e.g. "my_quotas"; else df -Pk on every quota_path
    quota_paths: list[str] = field(default_factory=list)   # extra paths to df (upload roots, group volumes); remote_root, control_root, $HOME,
                                                           # $PROJECT and $GROUP (when set) are always included (§6.1 ::DF)
    poll: dict = field(default_factory=lambda: {"base_s": 60, "min_s": 30, "max_s": 120})
    requires_vpn_hint: str | None = None # e.g. "Connect Cisco Secure Client to vpn.cmu.edu (group Campus VPN)"
    target_overrides: dict[str, dict] = field(default_factory=dict)   # target_key glob -> {"enabled", "max_pending", "max_running", "soft_cap",
                                                                      #   "preference", "penalty_h", "allow_self_preempt"}
    ssh_max_exec: int = 6                # exec channels per connection (MaxSessions 10 minus SFTP + headroom)
    cmd_timeout_s: int | None = None     # per-command timeout; None = discovered: max(120, MessageTimeout + 60) (260 s on Bridges-2, 120 s on TRACE)
```

Profile examples (documented in README, not code): TRACE `remote_root=/trace/group/biosimmlab/wxu2`, `transfer_host=data.trace.cmu.edu`, `requires_vpn_hint=...`, `su_rates={}` (free), `quota_paths=["/trace/group/biosimmlab"]` (the 3 TB group volume; `$HOME` sits on the 932 TB volume and says nothing about the group quota, fixture `trace/df_home.out`), `target_overrides={"trace:biosimmlab*": {"enabled": false}}` — **the lab partition is opt-in**: it preempts `batch` jobs on the same 29 nodes, including the user's own (§8 Feasibility); flip `enabled` to true and set `max_running` (e.g. 1) to use it; Bridges-2 `remote_root=/ocean/projects/mch250030p/wxu7`, `transfer_host=data.bridges2.psc.edu`, `default_account=mch250030p`, `partition_groups=[["GPU-small","GPU-shared"]]`, `no_mem_flag=["RM-shared"]`, `su_rates={"gpu:h100-80":2,"gpu:*":1,"cpu":1}`, `balance_command="projects"`, `quota_command="my_quotas"`.

### 2.2 Transport contract (transport.py, extended not rewritten)

- `SSHTransport(profile, host=None, port=None, role="login"|"transfer")`; the registry holds one instance per (cluster, role). Connect options: `username`, `password=lambda: asyncio.to_thread(credentials.get_password, profile)` (single-use callable; asyncssh awaits it), `client_keys=None`, `agent_path=None`, `gss_host=None`, `preferred_auth=['keyboard-interactive','password']` (auth=password) — key/agent profiles unchanged, `config=None`, `known_hosts=<callable, below>`, `connect_timeout=45`, `login_timeout=90`, `keepalive_interval=30`, `keepalive_count_max=4`, `compression_algs=None`, `encoding='utf-8'`, `errors='replace'`.
- Host keys: store `~/.slurm-mcp/known_hosts` (OpenSSH format, **multiple lines per alias allowed**) plus `kv` `hostkeys.<alias>` = `{ip: [fingerprint,...]}`. `known_hosts` callable returns the stored keys for the alias; `HpcClient.validate_host_public_key(host, addr, port, key)` is called only for unknown keys: if `addr` was never seen for this alias (round-robin pool member) → accept, append line, log `new host key for <alias> from <addr> <fp>`; if `addr` was seen with a different key of the same type → return False ⇒ `HostKeyChanged` (E_HOSTKEY). `slurm-mcp hostkeys forget <cluster>` clears both.
- `run(command, *, timeout=None, input=None, idempotent=True, login_shell=True) -> CommandResult`. `timeout=None` resolves to the cluster's `caps.cmd_timeout_s` (= `profile.cmd_timeout_s` or `max(120, MessageTimeout + 60)`; `MessageTimeout` is 200 s on Bridges-2 and 30 s on TRACE, fixtures `scontrol_config.out`) so a legitimately slow `sbatch`/`sacct` under `max_rpc_cnt` pressure is not reported as lost; the tick, `--test-only` and `submit.sh` always use that value, `run_command` uses its own `timeout_s`. On `ConnectionLost`/`ChannelOpenError('connection closed')`/`exit_status is None`: reconnect and retry **once only when `idempotent=True`**; `sbatch`, `submit.sh`, `scancel`-with-signal and `tar -x` are called with `idempotent=False` and the caller decides (§5.1 step 7). A timeout on a non-idempotent command is reported as `CommandTimeout` (ambiguous), never as failure. `PermissionDenied` sets `auth_failed=True` on the transport; no reconnect is attempted until `reset_auth()` (CLI `slurm-mcp auth set` + `clusters(refresh=True)`).
- `run_with_stdin_file(command, path, timeout)` (binary stdin from a local file, `encoding=None`), `run_to_file(command, path, timeout)` (binary stdout to a local file). `sftp()` returns a cached `SFTPClient`, recreated after any reconnect. `tcp_probe(host, port, timeout=3) -> bool` (no auth; used before every connect when `requires_vpn_hint` is set and after connect failures), `banner_probe(host, port) -> str` (reads the `SSH-2.0-…` line; used once to choose `transfer_port`).
- Concurrency: `asyncio.Semaphore(profile.ssh_max_exec)` around channel opens; the persistent SFTP client holds one slot outside the semaphore.

### 2.3 File I/O without SFTP (verified 2026-09-02 on Bridges-2)

Not every host offers both channels. Measured: the **Bridges-2 login node refuses the SFTP subsystem** (`ChannelOpenError: Session request failed`, consistently) while exec channels work, and the **Bridges-2 DTN (`data.bridges2.psc.edu`) refuses every exec command** (`Login denied: <cmd> is not an allowed command`) while SFTP works. TRACE's login node offers both. Therefore:

- Discovery records two more capabilities per cluster: `caps.login_sftp_ok` (one `start_sftp_client()` attempt on the login transport at bootstrap; a `ChannelOpenError` ⇒ False, cached with the caps) and `caps.transfer_exec_ok` / `caps.transfer_sftp_ok` (already in §6.1 "Transfer-host capabilities").
- `SlurmClient` small-file operations — `write_file`, `mkdirs`, `ls`, `stat`, `read_file`, `deploy_helpers`, the ctrl-dir writes of §5.1 step 6 and `remote_write` — use **exec-channel primitives when `login_sftp_ok` is False**: write = `mkdir -p <dir> && cat > <path>.tmp-<8hex> && mv -f <path>.tmp-<8hex> <path>` with the text on stdin (`run(..., input=text, idempotent=True)`; append = `cat >> <path>`), executable = `chmod 755`, mkdirs = `mkdir -p`, ls = `find <path> -maxdepth 1 -mindepth 1 -printf '%y|%s|%T@|%f\n'` (fallback `ls -la --time-style=+%s`), stat = `stat -c '%F|%s|%Y' <path>`, read = the existing `tail`/`head`/`grep` commands (already exec-based). When `login_sftp_ok` is True the SFTP path is preferred (fewer round trips for many small files); both paths are golden-tested to produce identical remote content.
- Bulk transfers (§5.5) never assume exec on the transfer host: tar streams are extracted **on the login host** after an SFTP `put` to the transfer host whenever `transfer_exec_ok` is False (the PSC case); when the login host has no SFTP either, per-file SFTP through the transfer host is the only bulk path (Bridges-2: DTN SFTP, same `/ocean` and `/jet` filesystems).
- The fake harness must be able to emulate both restrictions (`fake_cluster(..., login_sftp=False)` and a transfer host with `exec=False`) so the fallbacks are tested without the real clusters; the live smoke test (§10 item 9) runs the whole submit → wait → collect cycle on Bridges-2 with `login_sftp_ok=False`.
## 3. Data model

### 3.1 Enums (slurm/states.py)

```python
class JobState(str, Enum):      # ours; a superset of SLURM base states
    QUEUED="QUEUED"; UPLOADING="UPLOADING"; SUBMITTING="SUBMITTING"; SUBMITTED="SUBMITTED"; RUNNING="RUNNING"; COMPLETING="COMPLETING"
    COMPLETED="COMPLETED"; FAILED="FAILED"; TIMEOUT="TIMEOUT"; OOM="OOM"; CANCELLED="CANCELLED"; PREEMPTED="PREEMPTED"
    NODE_FAIL="NODE_FAIL"; LOST="LOST"
TERMINAL = {COMPLETED, FAILED, TIMEOUT, OOM, CANCELLED, PREEMPTED, NODE_FAIL, LOST}
LIVE     = {QUEUED, UPLOADING, SUBMITTING, SUBMITTED, RUNNING, COMPLETING}
```
`QUEUED` = the intent is in the ledger (inputs may already be staged) but the job is **held locally** because its target has no free slot under `policy.max_pending_per_target` / the discovered `bf_max_job_user[_part]` cap (§8); the Monitor submits it when a slot frees. `SUBMITTED` covers SLURM PENDING/REQUEUED/held; `held` is `SUBMITTED` with `reason in {JobHeldUser, JobHeldAdmin}`. A requeue is not a state: it is an event, `restarts += 1`, and state back to `SUBMITTED`. `PREEMPTED`/`NODE_FAIL` are terminal only when SLURM did not requeue the job (§5.3).

SLURM long state → ours (`SLURM_STATE_MAP`, used for both `squeue %T` and `sacct State` first token): `PENDING, REQUEUED, REQUEUE_HOLD, SPECIAL_EXIT, RESV_DEL_HOLD, REQUEUE_FED → SUBMITTED`; `RUNNING, SUSPENDED, RESIZING, CONFIGURING, SIGNALING, STAGE_OUT, STOPPED → RUNNING`; `COMPLETING → COMPLETING`; `COMPLETED → COMPLETED`; `FAILED, BOOT_FAIL, DEADLINE, LAUNCH_FAILED, REVOKED → FAILED`; `TIMEOUT → TIMEOUT`; `OUT_OF_MEMORY → OOM`; `CANCELLED → CANCELLED`; `PREEMPTED → PREEMPTED`; `NODE_FAIL → NODE_FAIL`. Unknown token → keep previous state, log once.

`AttemptState = INTENT | UNCONFIRMED | ACTIVE | SUPERSEDED | FAILED | DONE`. `INTENT` = row written, `submit.sh` **not yet invoked** (safe to fail without looking at the cluster); `UNCONFIRMED` = `submit.sh` was (or may have been) invoked and no `JOBID` line was parsed — only the recovery sweep (§5.2 step 9) may leave this state. `TransferState = planned | running | done | failed | cancelled`. `CmdState = queued | running | done | killed | aborted`.

Reason classification (`classify_reason(reason) -> normal | limit | held | dependency | reservation | unknown`): `limit` if it matches `^(QOS|Assoc|Association)(Grp|Max|Job|Resource|Time|Usage)` or `PartitionNodeLimit|PartitionTimeLimit|JobArrayTaskLimit`; `held` if `JobHeldUser|JobHeldAdmin|JobHoldMaxRequeue`; `dependency` if `Dependency|DependencyNeverSatisfied`; `reservation` if `ReqNodeNotAvail|Reservation|Reserved for maintenance`; `normal` if `None|Priority|Resources|BeginTime|Cleaning|WaitingForScheduling|SchedDefer`.

### 3.2 Pydantic models (models.py)

```python
class Resources(BaseModel):
    time: str                              # "HH:MM:SS", "D-HH:MM:SS", "MM:SS" or minutes; required (both clusters default to 30 min)
    gpus: int = 0                          # GPUs per node (always rendered as a *typed* --gres=gpu:<type>:<gpus>, §8 candidates)
    gpu_types: list[str] | None = None     # acceptable gres types e.g. ["h100-80","l40s-48"]; None = any type present in the partition
                                           # (expanded into one candidate target per type; an untyped --gres is never sent, §8 Cost)
    cpus: int | None = None                # --cpus-per-task
    tasks: int | None = None               # --ntasks-per-node
    mem: str | None = None                 # "64G"; dropped on profile.no_mem_flag partitions with a warning
    nodes: int = 1
    exclusive: bool = False
    constraint: str | None = None          # -C

class InputSpec(BaseModel):
    local: str                             # local file or directory
    remote: str | None = None              # default <workdir>/<basename(local)>
    ignore: list[str] = []                 # extra gitignore-style patterns

class JobSpec(BaseModel):
    name: str                              # SLURM -J; [A-Za-z0-9_.-]{1,64}
    command: str | None = None             # bash body (may be multi-line); exactly one of command/script/script_path
    script: str | None = None              # full user sbatch script text; #SBATCH lines are *parsed into spec fields and stripped* (§6.3)
    script_path: str | None = None         # "local:<path>" (read locally as utf-8-sig, CRLF normalised) or an absolute remote path (cat'ed at submit)
    workdir: str | None = None             # remote cwd; default: the script's --chdir, else <remote_root>/<name>; per attempt after a cross-cluster move
    cluster: str | None = None; partition: str | None = None; qos: str | None = None; account: str | None = None
    resources: Resources
    inputs: list[InputSpec] = []
    outputs: list[str] = []                # globs relative to workdir for collect_results
    modules: list[str] = []; setup: str | None = None; env: dict[str, str] = {}
    array: str | None = None; array_parallel: int | None = None      # "0-99", %parallel appended
    depends_on: list[str] = []             # "j12" (afterok) | "afterany:j12" | "afternotok:j12" | "after:j12" | "singleton"
    wrap: bool = True                      # run through wrap.sh (status/heartbeat/exit code/timeout classification)
    requeue: bool | None = None            # None = auto (§5.3): True when the chosen partition's PreemptMode contains REQUEUE or
                                           # on_timeout=="requeue"; otherwise --no-requeue is rendered on clusters that charge SUs and have
                                           # JobRequeue=1 (Bridges-2) so the worst-case cost is one run; free clusters keep the site default
    on_timeout: Literal["fail","requeue"] = "fail"   # "requeue" is accepted only with child_signal set AND checkpoint_interval_h set
                                                     # (E_INVALID_SPEC otherwise): without a checkpointing payload a requeue reruns from scratch
    grace_s: int = 120                     # --signal=B:USR1@<grace_s> is sent to wrap.sh as a time-limit *warning*; also the graceful-cancel grace
    child_signal: str | None = None        # None (default): the warning is NOT forwarded to the payload (an untrapped USR1 would kill it, rc 138).
                                           # "USR1"/"TERM"/…: "my payload handles SIG<x>" — wrap.sh forwards it to the payload's process group
    max_restarts: int = 3                  # bounds requeues (wrap.sh refuses to run past it) and the worst-case cost (1 + max_restarts runs)
    checkpoint_interval_h: float | None = None   # declared periodic checkpointing (preemption risk §8; required for on_timeout="requeue")
    stdout: str | None = None; stderr: str | None = None   # sbatch patterns (%j %x %A %a …); relative paths resolve against workdir; always
                                                           # rendered absolute and the directory is created before submit (§6.3)
    extra_sbatch: list[str] = []           # verbatim "#SBATCH ..." lines (without the prefix), e.g. ["--time-min=01:00:00"]
    tags: dict[str, str] = {}

class Target(BaseModel):                   # string form "<cluster>:<partition[,partition]>[:<gres-type>][@<qos>]"
    cluster: str; partitions: list[str]; gres_type: str | None = None; qos: str | None = None; account: str | None = None
    @property
    def key(self) -> str: ...              # "bridges2:GPU-small,GPU-shared:v100-32@gpu"

class PlacementPolicy(BaseModel):
    objective: Literal["balanced","fastest","cheapest"] = "balanced"
    su_to_hours: float | None = None       # None = preset by objective (0.25 / 0.02 / 2.0)
    su_reserve: float = 50.0
    max_pending_per_target: int | None = None         # None = discovered per-cluster cap (bf_max_job_user − 2 on TRACE = 8; bf_max_job_user_part − 2
                                                      # on Bridges-2 = 18 per partition). Jobs above the cap are *held locally* (state QUEUED) and
                                                      # submitted when a slot frees — never escalated to another target because of the cap (§8)
    max_running_per_target: dict[str, int] = {}       # target_key glob -> cap (etiquette), e.g. {"trace:biosimmlab*": 1}
    allow_self_preempt: bool = False                  # False: a higher-tier partition that would preempt my own running jobs is infeasible (§8)
    soft_caps: dict[str, int] = {}                    # target_key glob -> running count above which etiquette_h applies
    etiquette_h: float = 2.0
    targets_allow: list[str] = []; targets_deny: list[str] = []     # globs on target_key
    prefer_cluster: str | None = None
    unknown_wait_h: float = 12.0
    rebalance: RebalancePolicy = RebalancePolicy()

class RebalancePolicy(BaseModel):
    enabled: bool = True; interval_min: int = 10; min_gain_h: float = 1.0; max_moves_per_job: int = 3
    max_extra_su: float = 0.0; min_age_min: int = 5; max_moves_per_hour: int = 6; hysteresis_h: float = 0.5

class NotifyPolicy(BaseModel):
    toast: bool = True
    toast_kinds: list[str] = ["completed","failed","timeout","oom","cancelled","preempted","node_fail","lost","needs_attention",
                              "alloc_ready","alloc_expiring","transfer_failed","cluster_unreachable"]
    webhook_url: str | None = None; webhook_kinds: list[str] = []      # [] = same as toast_kinds
    email: str | None = None                                           # adds --mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90 --mail-user
    quiet_hours: tuple[int, int] | None = None                         # local hours [start, end) during which toasts are suppressed
```

Validation (`JobSpec.model_validator`): exactly one of `command|script|script_path`; `resources.time` parses; `name` matches the regex; `array` matches `^\d+(-\d+)?(:\d+)?(,\d+(-\d+)?)*$`; `depends_on` entries match `^(after|afterok|afterany|afternotok|aftercorr)?:?j\d+$|^singleton$`; `child_signal` is a signal name without `SIG` and not `KILL|STOP`; `on_timeout == "requeue"` requires `child_signal` and `checkpoint_interval_h` (`E_INVALID_SPEC: on_timeout=requeue would rerun the job from scratch up to max_restarts times — fix: declare child_signal (the signal your program checkpoints on) and checkpoint_interval_h, or use on_timeout="fail" with a longer time`); `command`/`script` text is normalised by `textio.normalize_text()` (CRLF → LF, BOM stripped, NUL refused with `E_INVALID_SPEC`) and a `warnings` entry is produced when CRLF was stripped.

### 3.3 SQLite schema (store.py; `PRAGMA journal_mode=WAL; synchronous=NORMAL; foreign_keys=ON; busy_timeout=5000`; `PRAGMA user_version=1`)

```sql
CREATE TABLE jobs(                    -- cluster-independent identity and progress; everything cluster-relative lives in attempts
  handle TEXT PRIMARY KEY,            -- "j17" | "a3"
  kind TEXT NOT NULL,                 -- job | alloc
  name TEXT NOT NULL,
  state TEXT NOT NULL, slurm_state TEXT, reason TEXT,
  spec_json TEXT NOT NULL, placement_mode TEXT NOT NULL,   -- auto | explicit | plan
  attempt_no INTEGER NOT NULL DEFAULT 1,                   -- the current attempt; (handle, attempt_no) -> attempts row
  submit_ts INTEGER, start_ts INTEGER, end_ts INTEGER, est_start_ts INTEGER,   -- cluster epoch (of the current attempt's cluster)
  exit_code INTEGER, exit_signal INTEGER, restarts INTEGER NOT NULL DEFAULT 0, moves INTEGER NOT NULL DEFAULT 0,
  cost_est_su REAL, cost_worst_su REAL, cost_actual_su REAL,
  last_seen_ts INTEGER, stale_ticks INTEGER NOT NULL DEFAULT 0, terminal_ts INTEGER, enriched INTEGER NOT NULL DEFAULT 0,
  collected_ts INTEGER, cancel_requested_ts INTEGER, cancel_hard_ts INTEGER, hold_reason TEXT,
  alloc_ready INTEGER NOT NULL DEFAULT 0, alloc_end_ts INTEGER, array_size INTEGER, depends_on_json TEXT,   -- resolved deps: [{handle, type}]
  heartbeat_ts INTEGER, progress_json TEXT, last_line TEXT,
  created_local REAL NOT NULL, updated_local REAL NOT NULL);
CREATE TABLE attempts(                -- one row per (handle, attempt); every path/node/id below is valid only on attempts.cluster
  id INTEGER PRIMARY KEY AUTOINCREMENT, handle TEXT NOT NULL REFERENCES jobs(handle), attempt_no INTEGER NOT NULL,
  cluster TEXT NOT NULL, token TEXT NOT NULL UNIQUE,        -- "t-" + 12 hex
  slurm_id TEXT,                                            -- base id for arrays
  ctrl_root TEXT NOT NULL, ctrl_dir TEXT NOT NULL,          -- <control_root(cluster)>/jobs/<handle>, <ctrl_root>/a<attempt_no>
  workdir TEXT NOT NULL,                                    -- derived per cluster (§5.1 step 3, §5.4)
  stdout_pattern TEXT, stderr_pattern TEXT,                 -- absolute sbatch patterns as rendered (%j etc.)
  stdout_path TEXT, stderr_path TEXT,                       -- expanded once slurm_id is known (§6.3); arrays: pattern kept, elements in array_tasks
  node TEXT,
  target_json TEXT NOT NULL, submit_line TEXT, state TEXT NOT NULL,   -- AttemptState
  cause TEXT NOT NULL,                                      -- initial | rebalanced | preempted | timeout | nodefail | user | queued
  intent_local REAL NOT NULL, invoked_local REAL, confirmed_local REAL, submit_ts INTEGER, end_ts INTEGER,
  final_state TEXT, exit_code INTEGER, reason TEXT, excluded_nodes TEXT,   -- excluded_nodes: same-cluster nodes only
  UNIQUE(handle, attempt_no));
CREATE VIEW jobs_current AS           -- what every tool and the Monitor read: job fields + the current attempt's cluster-relative fields
  SELECT j.*, a.cluster, a.slurm_id, a.ctrl_root, a.ctrl_dir, a.workdir, a.stdout_path, a.stderr_path, a.stdout_pattern, a.stderr_pattern,
         a.node, a.token, a.target_json, a.submit_line, a.state AS attempt_state, a.excluded_nodes
  FROM jobs j JOIN attempts a ON a.handle = j.handle AND a.attempt_no = j.attempt_no;
CREATE TABLE lease(name TEXT PRIMARY KEY,                    -- "monitor"
  owner_pid INTEGER NOT NULL, owner_host TEXT NOT NULL, token INTEGER NOT NULL,   -- fencing token, +1 on every acquisition
  acquired_local REAL NOT NULL, renewed_local REAL NOT NULL);
CREATE TABLE event_acks(client_id TEXT NOT NULL, seq INTEGER NOT NULL, acked_local REAL NOT NULL, PRIMARY KEY(client_id, seq));
CREATE TABLE deliveries(client_id TEXT PRIMARY KEY, next_seq INTEGER NOT NULL, seqs_json TEXT NOT NULL, delivered_local REAL NOT NULL);
CREATE TABLE array_tasks(handle TEXT NOT NULL, task_id INTEGER NOT NULL, slurm_id TEXT, state TEXT NOT NULL, exit_code INTEGER,
  start_ts INTEGER, end_ts INTEGER, node TEXT, PRIMARY KEY(handle, task_id));
CREATE TABLE events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts_local REAL NOT NULL, ts INTEGER,           -- ts = cluster epoch when known
  kind TEXT NOT NULL, handle TEXT, cluster TEXT, slurm_id TEXT, summary TEXT NOT NULL, payload_json TEXT NOT NULL,
  notified INTEGER NOT NULL DEFAULT 0);
CREATE INDEX events_handle ON events(handle, seq);
CREATE TABLE transfers(
  id INTEGER PRIMARY KEY AUTOINCREMENT,                     -- exposed as "t<id>"
  kind TEXT NOT NULL,                                       -- upload | download | collect | inputs
  cluster TEXT NOT NULL, host_role TEXT NOT NULL,           -- login | transfer
  local TEXT NOT NULL, remote TEXT NOT NULL, state TEXT NOT NULL, mode TEXT,   -- tar | sftp
  files_total INTEGER, files_done INTEGER NOT NULL DEFAULT 0, bytes_total INTEGER, bytes_done INTEGER NOT NULL DEFAULT 0,
  error TEXT, handle TEXT, started_local REAL NOT NULL, finished_local REAL, seconds REAL);
CREATE TABLE transfer_files(transfer_id INTEGER NOT NULL REFERENCES transfers(id), rel_path TEXT NOT NULL,   -- always POSIX ("a/b/c")
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha1 TEXT, state TEXT NOT NULL, bytes_done INTEGER NOT NULL DEFAULT 0,
  local_name TEXT,                                          -- downloads: the NTFS-safe local rel_path when it differs from rel_path (§5.5)
  PRIMARY KEY(transfer_id, rel_path));
CREATE TABLE manifests(scope TEXT NOT NULL, rel_path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha1 TEXT,
  updated_local REAL NOT NULL, PRIMARY KEY(scope, rel_path));                 -- scope = "up:<cluster>:<remote_root>" | "down:<local_dir>"
CREATE TABLE alloc_cmds(id TEXT PRIMARY KEY,                                  -- "a3.c2"
  handle TEXT NOT NULL REFERENCES jobs(handle), n INTEGER NOT NULL, command TEXT NOT NULL, cwd TEXT, mode TEXT NOT NULL,   -- fg | bg
  state TEXT NOT NULL, submitted_local REAL NOT NULL, started_ts INTEGER, done_ts INTEGER, rc INTEGER, out_path TEXT NOT NULL,
  kill_path TEXT NOT NULL,                                                    -- <ctrl>/cmds/<base>.kill (002.kill | 003.bg.kill, §7.3)
  kill_requested_local REAL);
CREATE TABLE plans(plan_id TEXT PRIMARY KEY, created_local REAL NOT NULL, expires_local REAL NOT NULL, spec_json TEXT NOT NULL,
  options_json TEXT NOT NULL, recommended TEXT);
CREATE TABLE wait_history(id INTEGER PRIMARY KEY AUTOINCREMENT, cluster TEXT NOT NULL, target_key TEXT NOT NULL,
  submit_ts INTEGER NOT NULL, start_ts INTEGER NOT NULL, wait_s INTEGER NOT NULL, gpus INTEGER, hours REAL, source TEXT NOT NULL);  -- observed | backfill
CREATE INDEX wait_history_key ON wait_history(cluster, target_key, start_ts);
CREATE TABLE target_stats(cluster TEXT NOT NULL, target_key TEXT NOT NULL, consecutive_failures INTEGER NOT NULL DEFAULT 0,
  breaker_open_until_local REAL, last_error TEXT, infeasible_until_local REAL, infeasible_reason TEXT,
  last_node_fail_node TEXT, last_node_fail_local REAL, PRIMARY KEY(cluster, target_key));
CREATE TABLE kv(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
```

`kv` keys: `policy.placement` (PlacementPolicy), `policy.notify` (NotifyPolicy), `cursor.<client_id>` (int: the ack floor — every event `<= floor` is acknowledged by that client; rows of `event_acks` below the floor are pruned), `caps.<cluster>` (discovery cache, §6.1, with `fetched_local`), `hostkeys.<alias>`, `untracked.<cluster>` (last tick's squeue rows not in the ledger), `snapshot.<cluster>` (cluster load snapshot + ts), `moves.<cluster>` (timestamps of the last hour's rebalance moves), `counter.plan`, `counter.handle`.

Transactions: every write goes through `store.write(fn)` = `BEGIN IMMEDIATE; fn(conn); COMMIT` on a worker thread (`busy_timeout=5000` makes a second process wait, not fail). Writes that belong to the Monitor (`state`, events emitted by ticks, `scancel`/`submit` side effects of rebalancing) go through `store.write_fenced(token, fn)`, which first runs `SELECT token FROM lease WHERE name='monitor'` inside the same transaction and raises `LeaseLost` when it differs — the caller stops its loop (§5.8). Handle allocation: inside `BEGIN IMMEDIATE`, `counter.handle += 1` → `j<n>`/`a<n>` (no `max()` scan, no in-process-lock reliance); transfers use the autoincrement id. `plans.plan_id` = `p` + `counter.plan`. `events.seq` is the autoincrement, allocated in the same transaction as the state change it describes.

### 3.4 Event kinds and payloads (events.py)

Every payload includes `handle, cluster, slurm_id, state`. Kinds: `queued{target, why}` (held locally by a cap), `submitted{target, attempt_no, est_start_ts, stdout_path, workdir, ctrl_dir, injected, warnings}`, `submit_failed{error_code, stderr, hint}`, `started{node, wait_s}`, `completed|failed|timeout|oom|cancelled|preempted|node_fail|lost{exit_code, exit_signal, elapsed_s, time_limit_s, cost_su, stdout_path, last_line, cause, source ("sacct"|"helper"|"helper+sacct"|"squeue"), restarts, by}` (`preempted`/`node_fail` are emitted **non-terminal** with `requeued: true` when SLURM requeues; terminal otherwise), `requeued{cause, restarts, attempt_no}`, `rebalanced{from_target, to_target, old_slurm_id, new_slurm_id, gain_h, new_workdir}`, `held{reason}`, `dependency_updated{dependent, new_dependency}`, `needs_attention{why, hint}` (`why` ∈ `max_restarts, requeue_loop, submit_unconfirmed, submit_stuck, duplicate_cancelled, quota, heartbeat_stale, restart_cost, dependency_unsatisfiable, lease_lost, db_corrupt`), `alloc_ready{node, end_ts}`, `alloc_expiring{minutes_left}`, `alloc_ended{}`, `cmd_done{cmd_id, rc, out_path}`, `transfer_done{transfer_id, files, bytes, seconds, renamed: [{remote, local}]}`, `transfer_failed{transfer_id, error}`, `cluster_unreachable{error, hint}`, `cluster_recovered{}`, `quota_warning{path, used_pct}`. `payload.observed_late = true` when a transition is detected by the first tick after a server restart.
## 4. Tool surface (the contract)

Common base: `class Result(BaseModel): summary: str; unread_events: int = 0; next: str | None = None`. All 20 tools are `async def`, take `ctx: Context` where noted, and use `_mcp.read_only()/mutating()/destructive()` annotations. Tool names are short because Claude Code exposes them as `mcp__slurm__<name>`. Errors: `raise ToolError(str(SlurmMcpError))` → `E_CODE: message — fix: hint` (§9.1). `_meta` is static per tool (declared in `tools/list`); no per-call meta.

Server: `MCPServer("slurm", instructions=INSTRUCTIONS, lifespan=app_lifespan, log_level="INFO")`. INSTRUCTIONS (≤ 2 KB):

> Tools for running SLURM jobs on SSH clusters configured with `slurm-mcp cluster add`. Job handles look like `j17`, allocations `a3` (commands `a3.c2`), transfers `t4`, plans `p9`. Typical flow: `plan_job` (optional) → `submit_job` → `wait_for_events(timeout_s=300)` or `job_status` → `job_logs` → `collect_results`. Every response includes `unread_events`; when > 0 call `wait_for_events(timeout_s=0)`. `wait_for_events` with a long timeout is the right way to wait for a job: Claude Code moves the call to the background after 2 minutes and returns the result as a task notification. Events are delivered until you acknowledge them: pass the previous result's `next_seq` as `ack_seq` on your next call (if you never saw a result, call again without it and the same events are replayed). Prefer `placement="auto"` so the server balances partitions/clusters by wait time, SU cost and policy. `submit_job` returns a handle immediately and finishes in the background; the `submitted`/`submit_failed` event closes it. Use `job_control`/`rebalance` rather than `run_command`. Never poll faster than every 30 s; the server already does.

### Clusters

**`clusters(refresh: bool = False) -> ClustersResult`** RO. `clusters: [{name, host, transfer_host, connected, auth_failed, reachable, last_tick_age_s, tracked_jobs: {queued, pending, running}, su_balance, quota: [{path, used_pct, free_gb, role: "home"|"remote_root"|"control_root"|"project"|"group"|"upload_root"}], monitor: "self"|"held by pid N"|"lost to pid N"|"none", warnings: [str]}]`, plus `session_id` (this server process's default `client_id` for `wait_for_events`). `refresh=True` re-runs discovery (read-only commands) on every cluster; helper deployment is **not** done here (it happens on first `submit_job`/`allocate`, which are W). ≈150 tokens.

**`cluster_status(cluster: str, refresh: bool = False, detail: Literal["summary","partitions","queue","targets","full"] = "partitions") -> ClusterStatusResult`** RO. From the snapshot (cached 60 s) and discovery cache: `partitions: [{name, avail, preempt_mode, priority_tier, grace_time_s, max_wall_s, default_time_s, nodes: {idle, mix, alloc, other, total}, gres_types: [str], pending_by_gres: [{gres, count, mine}], running_by_gres: [...], my_jobs: {pending, running}, limits: {max_wall_s, max_jobs_pu, max_submit_pu, max_tres_pj}, qos: str|None, charge: "free"|{unit, su_per_unit_h}}]`, `su_balance`, `quota: [{path, used_pct, free_gb}]`, `reservations_upcoming: [{name, start_ts, end_ts, partitions, maint}]`, `slurm_version`, `helper_version`, `caps_age_s`. `queue` adds my pending rows (≤ 50) with `est_start_ts`/`reason`; `targets` lists candidate target keys with `enabled/max_pending/max_running`; `full` adds config keys. Bounded to 20 partitions.

**`run_command(cluster: str, command: str, timeout_s: int = 60, cwd: str | None = None, max_chars: int = 8000) -> CommandResult`** D (`meta={"anthropic/maxResultSizeChars": 60000}`). Escape hatch under `bash -lc`. Returns `rc, stdout_tail, stderr_tail, truncated, seconds`. Refuses heredocs and commands > 4000 chars (`E_CMD_TOO_LONG — fix: remote_write the script, then run it`). Not the way to submit (no lineage); the description says so.

### Files

**`upload(cluster: str, local: str, remote: str, ignore: list[str] | None = None, mode: Literal["auto","tar","sftp"] = "auto", dry_run: bool = False, wait_s: int = 600, ctx) -> TransferResult`** W idempotent. File or directory; incremental against `manifests` scope `up:<cluster>:<remote>` (§5.5). Default ignore `.git/ __pycache__/ *.pyc .venv/ node_modules/ .slurm-mcp/ wandb/ *.ckpt` + `.slurm-mcpignore` in `local` + `ignore`. Checks quota headroom first. Returns `transfer_id ("t4"), state, files_sent, files_skipped, bytes, seconds, mode, host_role, quota_after_pct, remote`; if not finished within `wait_s`, `state="running"` and `next: "job_status(['t4']) or wait_for_events(kinds=['transfer_done'])"`. Progress via `ctx.report_progress` every 5 % / 10 s. `dry_run` lists ≤ 100 paths that would be sent.

**`download(cluster: str, remote_globs: list[str], local_dir: str, incremental: bool = True, max_files: int = 2000, max_bytes: int = 2_000_000_000, wait_s: int = 600, ctx) -> TransferResult`** W idempotent (writes locally only). Globs absolute or relative to `remote_root`; `**` allowed. Refuses with `E_TOO_MANY_FILES`/`E_TOO_MANY_BYTES` (counts and the 5 largest paths) rather than truncating. Files modified in the last 15 s (cluster time) are skipped and listed in `skipped_in_progress`. Remote names that are not valid on the local filesystem (Windows: `: ? * " < > |`, control chars, trailing dots/spaces, reserved `CON NUL PRN AUX COM1-9 LPT1-9`) are saved under a safe name (`textio.local_safe_name`, e.g. `run_2026-09-01T12:00:00` → `run_2026-09-01T12_00_00`) and listed in `renamed: [{remote, local}]`; paths longer than 260 chars are opened with the `\\?\` prefix (§5.5).

**`remote_ls(cluster: str, path: str, glob: str | None = None, max_entries: int = 200, sort: Literal["name","mtime","size"] = "name") -> ListingResult`** RO. `entries: [{name, type, size, mtime_ts}]`, `truncated`.

**`remote_read(cluster: str, path: str, tail_lines: int | None = 100, head_lines: int | None = None, grep: str | None = None, offset: int | None = None, max_chars: int = 12000) -> ReadResult`** RO (`maxResultSizeChars` 60000). One `tail`/`head`/`grep -n -E -m 200`/`tail -c +N | head -c M` command; returns `text, size, next_offset, truncated`.

**`remote_write(cluster: str, path: str, text: str, mode: Literal["overwrite","append"] = "overwrite", mkdirs: bool = True, executable: bool = False) -> WriteResult`** W idempotent. SFTP write to `<path>.tmp-<8hex>` + `posix_rename` (append: SFTP open `a`). Returns `bytes, path`. ≤ 1 MB.

### Jobs

**`plan_job(job: JobSpec, placement: str | list[str] = "auto", max_options: int = 6) -> PlanResult`** RO (`sbatch --test-only` creates no job). Returns `plan_id` (valid 15 min), `options: [{target, feasible, est_wait_h, est_wait_src ("test_only"|"history"|"depth"|"none"), est_start_ts, queue_ahead, queue_ahead_untyped, cost_su, cost_worst_su, requeueable, charge, risk_pct, etiquette_h, score_h, why}]` sorted by score with infeasible rows last (≤ 3, with `why`), `recommended` (target key or null), `rendered_preview` (first 25 lines of `job.sbatch` plus the CLI args for the recommended target), `warnings` (incl. `stripped_directives: [str]` for user scripts and `crlf_normalized`). `cost_su` is one run; `cost_worst_su = cost_su × (1 + max_restarts)` when the job will be requeueable (§8 Cost) — feasibility uses the worst case. `placement` may be a target string (`"trace:batch:a40"`) or a list restricting auto placement. Each `--test-only` runs in its own exec (§6.3) with the cluster's `cmd_timeout_s`; a target whose pass times out is returned with `est_wait_src="none"`, never sinking the plan.

**`submit_job(job: JobSpec | None = None, plan_id: str | None = None, placement: str | list[str] = "auto", target: str | None = None, hold: bool = False, wait_s: int = 90, ctx) -> SubmitResult`** W. Exactly one of `job`/`plan_id`; `target` overrides the plan's recommendation. The tool validates, resolves dependencies, writes the `jobs`/`attempts` intent rows and **returns the handle as soon as they are committed**; the rest of §5.1 (helper deploy, inputs, render, placement, `submit.sh`) runs as a server-side `SubmitTask` that the tool merely awaits for up to `wait_s` (progress via `ctx.report_progress` every 5 s: `"uploading 3/12 files"`, `"test-only 2/4"`, `"submitting"`). A client abort/timeout never cancels the task. Returns `handle, cluster, slurm_id, attempt_no, target, state, est_start_ts, cost_est_su, cost_worst_su, submit_line, workdir, ctrl_dir, stdout_path, stderr_path, injected: [str], stripped_directives: [str], dependencies_resolved: [str], uploads: {transfer_ids, files_sent, bytes}, array_size, warnings` and `next`. When finished within `wait_s`: `state="SUBMITTED"` (or `QUEUED` when held by a cap) and `next: "wait_for_events(job_ids=['j17'], timeout_s=600) or job_status(['j17'])"`. Otherwise `state` is `UPLOADING`/`SUBMITTING`, `slurm_id=null`, and `next: "wait_for_events(kinds=['submitted','submit_failed','queued'], job_ids=['j17'])"`; the durable `submitted`/`submit_failed` event (§3.4) carries the same fields. Definite submit errors surface as `submit_failed` + `ToolError` when they happen inside `wait_s`, otherwise only as the event. If the reply to `submit.sh` was lost or ambiguous the state stays `SUBMITTING` with `summary` "being confirmed" (§5.1 step 7); the server never re-runs `sbatch` itself.

**`list_jobs(cluster: str | None = None, state: Literal["active","pending","running","terminal","all"] = "active", since_h: float | None = None, name: str | None = None, kind: Literal["job","alloc","all"] = "all", include_untracked: bool = False, limit: int = 50) -> JobListResult`** RO. Rows `{handle, kind, name, cluster, slurm_id, state, reason, target, elapsed_s, time_limit_s, restarts, moves, est_start_ts}`; `counts_by_state`; `truncated`. `include_untracked=True` appends this user's squeue rows not in the ledger as `handle=null`. ≈15 tokens per row.

**`job_status(ids: list[str], detail: Literal["normal","full"] = "normal") -> JobStatusResult`** RO. Accepts `j17`, `j18[7]`, `a3`, `a3.c2`, `t4`, or `<cluster>:<slurm_id>` (adopts an untracked job of mine: creates a handle with `wrap=False`, `placement_mode="explicit"`). Per job: list row fields plus `submit_ts, start_ts, end_ts, exit: {rc, signal}, node, progress (JSON ≤ 1 KB from progress.json), heartbeat_age_s, last_log_line, cost_su, cost_worst_su, attempts_count, paths: {cluster, workdir, ctrl_dir, stdout, stderr}` (all from `jobs_current`, i.e. the **current attempt's cluster**), `dependencies: [{handle, type, status}]`, `dependents: [handle]`, `alloc: {ready, end_ts, cmds_outstanding}` for allocations, `transfer: {state, files_done, files_total, bytes_done, bytes_total, error}` for `tN`, `cmd: {state, rc, started_ts, done_ts}` for `a3.c2`, `next_action` (e.g. `"FAILED rc=1: call job_logs('j17', stream='err')"`). `full` adds attempt history, raw squeue/sacct fields and `seff`-style efficiency when finished. Triggers one tick if the last one is > 20 s old.

**`job_logs(id: str, stream: Literal["out","err","both"] = "out", tail_lines: int = 80, grep: str | None = None, offset: int | None = None, max_chars: int = 12000) -> LogResult`** RO (`maxResultSizeChars` 60000). `j18[7]` reads the element's file; `a3.c2` reads the command's `.out`. Returns `out: {text, size, next_offset, path, truncated}` and `err` likewise (null when not requested). `E_NO_LOG_YET` (with state and `next`) when the file does not exist.

**`job_control(ids: list[str], action: Literal["cancel","hold","release","requeue","signal"], signal: str | None = None, graceful: bool = True, reason: str | None = None, confirm: bool = False) -> ControlResult`** D. Per-id outcomes `[{id, accepted, outcome, message, hard_kill_ts}]`; partial failures are not errors. Cancelling more than 10 ids requires `confirm=True` (`E_CONFIRM_REQUIRED` otherwise). `cancel` decides **per id from the ledger's last observed state** (a tick is run first if the last one is > 20 s old):
- `QUEUED`/`UPLOADING`/`SUBMITTING` (not yet in SLURM): mark the attempt `FAILED(cause=user)`, job `CANCELLED`, stop the SubmitTask; if the attempt is `UNCONFIRMED` the cancel is deferred until recovery names an id (`outcome="cancel_pending_confirmation"`).
- `SUBMITTED` (PENDING/held/requeued — nothing to signal): plain `scancel <id>` immediately (idempotent; exit 0 even when already gone, fixture `scancel_bad_job`), `outcome="cancelled"`.
- `RUNNING`/`COMPLETING` with `graceful=True`: write `<ctrl_dir>/cancel.requested` (so `wrap.sh` classifies the TERM as `cancel`), `scancel --signal=TERM --full <id>` (wrap.sh forwards TERM to the payload's process group, §7.1), set `cancel_requested_ts` and `cancel_hard_ts = now + spec.grace_s` (**`spec.grace_s` is the single source of truth**, default 120 s), `kick(cluster)`; the Monitor issues the plain `scancel` at `cancel_hard_ts` (§5.2 step 5). `outcome="terminating"`, `message="hard kill at <ts>"`.
- `RUNNING` with `graceful=False`: plain `scancel <id>` now.
- Arrays: `j18` → `scancel <base>`, `j18[7]` → `scancel <base>_7`.
On `a3`: writes `<ctrl>/release`, then scancel. On `a3.c2`: writes `cmds/<basename>.kill` (`002.kill` or `002.bg.kill`, §7.3). On `t4`: cancels the transfer task. `hold` also sets `placement_mode="explicit"` so the rebalancer leaves it alone; `release` restores `auto` if it was auto before (`hold_reason`). Cancelling a job that has live dependents adds `message="dependents: j13 (afterok) will be re-evaluated"` (§5.1 step 2).

**`rebalance(ids: list[str] | None = None, dry_run: bool = True, min_gain_h: float | None = None, wait_s: int = 90) -> RebalanceResult`** W (D when `dry_run=False`). Re-evaluates pending auto-placed jobs with a fresh plan; `proposals: [{handle, from_target, to_target, est_wait_now_h, est_wait_new_h, gain_h, cost_delta_su, will_move, why}]`, `skipped: [{handle, why}]`, `moved: [handle]`, `moving: [handle]` (moves still in flight after `wait_s`; their `rebalanced` event follows). With `dry_run=False` applies moves as server-side tasks in the order of §5.4 (submit new → re-check old → cancel old); each move holds the Monitor's fencing token.

### Allocations

**`allocate(cluster: str, resources: Resources, hours: float, partition: str | None = None, qos: str | None = None, name: str = "alloc", placement: str = "auto", workdir: str | None = None, setup: str | None = None, modules: list[str] = [], idle_release_min: int = 0, wait_s: int = 90) -> SubmitResult`** W. Submits (through the same `SubmitTask` pipeline as `submit_job`) a sleeper batch job running `alloc-agent.sh`; returns handle `a3` (kind `alloc`) with `next: "wait_for_events(kinds=['alloc_ready'], job_ids=['a3'])"`. `hours` is capped at the partition's effective max wall; Monitor emits `alloc_expiring` 10 min before the end. The summary states the SU cost per hour when the cluster charges.

**`alloc_run(alloc_id: str, command: str, wait_s: int = 55, detach: bool = False, cwd: str | None = None, ctx) -> AllocRunResult`** W. Writes `cmds/NNN.sh` (`NNN.bg.sh` when `detach`), polls `NNN.rc` every 3 s up to `wait_s` (progress every 10 s; nothing runs on the login node between polls). Returns `cmd_id ("a3.c2"), state, rc, out_tail (≤ 4 KB), started_ts, seconds`, and when not done `next: "job_logs('a3.c2') or wait_for_events(kinds=['cmd_done'], job_ids=['a3.c2'])"`. `E_ALLOC_NOT_READY` (with `est_start_ts`) while the allocation is pending; `E_ALLOC_ENDED` after it ended.

### Events

**`wait_for_events(timeout_s: int = 300, kinds: list[str] | None = None, job_ids: list[str] | None = None, since_seq: int | None = None, ack_seq: int | None = None, max_events: int = 50, client_id: str | None = None, ctx) -> EventsResult`** RO. Long-poll: returns at once when matching **unacknowledged** events exist, else blocks until one arrives or `timeout_s` (server cap 600). Sends `ctx.report_progress(i, message="waiting: 2 pending, 1 running (j17 01:12/04:00)")` every 30 s (keeps Claude Code's 30-min stdio idle timer alive and drives auto-backgrounding).

Deliver-then-ack semantics (§5.6): events are never consumed by being returned; they are consumed only when the client acknowledges them on a **later** call. `client_id` defaults to `ctx.client_id` when the client sends one, else this server process's `session_id` (one stdio server = one Claude session; a second Claude session has its own server and its own cursor, so two sessions never eat each other's events). Per client the server keeps `cursor.<client_id>` (ack floor) and `event_acks` above it. Arguments: `ack_seq` = the `next_seq` of the previous result → every seq that result listed in `delivered_seqs` becomes acknowledged (idempotent; an unknown `ack_seq` is ignored with `warnings`); `since_seq` = explicit read position (defaults to the ack floor + 1); when `since_seq` is passed **without** `ack_seq` nothing is acknowledged (pure re-read, `timeout_s=0` = "list events"). Filters (`kinds`/`job_ids`) never advance anything: unmatched events stay unacknowledged and are counted in `unread_events` (= unacknowledged events for this client, filter-independent), so a filtered wait for `j17` cannot lose `alloc_ready(a3)`. Calling again with the same `since_seq` returns the same events. Returns `events: [{seq, ts, kind, handle, cluster, slurm_id, summary, payload}]`, `delivered_seqs: [int]`, `next_seq` (pass back as `ack_seq`), `acked: int`, `unread_events`, `unread_unmatched: int` (unacknowledged events hidden by this call's filters), `timed_out`, `snapshot: {queued, pending, running, alloc_ready, transfers_running, submits_running}`, `next` (`"wait_for_events(ack_seq=<next_seq>, …)"`).

### Results

**`collect_results(ids: list[str], local_dir: str | None = None, patterns: list[str] | None = None, include_logs: bool = True, wait_s: int = 600, ctx) -> CollectResult`** W idempotent. `spec.outputs` (or `patterns`) relative to the job's workdir plus stdout/stderr, into `<local_dir or ./results>/<name>-<handle>/`; incremental via `manifests` scope `down:<dir>`; sets `collected_ts`. Per job `{handle, state, exit_code, transfer_id, files, bytes, skipped, local_path}`; summary states the final state so one call closes the loop. Running jobs are collected too (partial results) but `collected_ts` is set only when terminal.

### Configuration

**`configure(placement: dict | None = None, notify: dict | None = None) -> ConfigResult`** W idempotent. Merges patches into `kv.policy.placement`/`kv.policy.notify` (validated as `PlacementPolicy`/`NotifyPolicy`) and returns the effective policies; no args = read.

### Prompts, resources, permissions

Prompts: `/slurm:status` (calls `list_jobs` + `clusters`, asks Claude to summarise), `/slurm:run <cluster> <script_path>` (submits a user script with auto placement). Resource template `slurm://jobs/{handle}/log` → last 200 lines of stdout (for `@slurm:` mentions).

README guidance: `permissions.allow`: `mcp__slurm__clusters, cluster_status, list_jobs, job_status, job_logs, remote_ls, remote_read, wait_for_events, plan_job`; `.mcp.json` entry `{"type":"stdio","command":"uv","args":["--directory","<path>","run","slurm-mcp","serve"],"timeout":900000}` — 900 s is above the 600 s wait cap and also raises the idle floor for this server. Register from PowerShell, not Git Bash (`/c` path mangling).

Token budget (typical): `submit_job` ≈ 250, `wait_for_events` with one terminal event ≈ 200, `job_logs` default ≈ 1.5 k, `collect_results` ≈ 150, `plan_job` ≈ 500. A submit → wait → inspect → collect cycle is four calls under 3 k tokens.
## 5. Key flows

### 5.1 Submit with auto-placement

Steps 1–3 run inside the tool call; steps 4–8 run in the `SubmitTask` (one asyncio task per handle, registered in `service.submits`) that the tool awaits for ≤ `wait_s`.

1. **Validate** the spec (§3.2). `script_path` `local:<p>` is read with `textio.read_local_text()` (utf-8-sig, CRLF → LF, NUL refused); a remote path is `cat`'ed (`idempotent=True`) and normalised the same way. User scripts (`script`/`script_path`) go through `render.parse_sbatch()` (§6.3): every leading `#SBATCH` directive that the server itself renders or passes on the command line is **converted into spec fields and removed from the script** — `-p/--partition`, `-q/--qos`, `-A/--account`, `-t/--time`, `-N/--nodes`, `-n/--ntasks`, `--ntasks-per-node`, `-c/--cpus-per-task`, `--gres`, `--gpus`, `--gpus-per-node`, `--mem`, `--mem-per-cpu`, `--mem-per-gpu`, `-C/--constraint`, `--exclusive`, `-J/--job-name`, `-o/--output`, `-e/--error`, `--open-mode`, `--requeue`, `--no-requeue`, `--signal`, `-a/--array`, `-d/--dependency`, `--kill-on-invalid-dep`, `-D/--chdir`, `--mail-type`, `--mail-user`, `--comment`, `-x/--exclude`, `-H/--hold`, `--export`. Mapping: `--gpus=[type:]N` (per job, fixture `scontrol_show_job_615411` `TresPerJob=gres:gpu:a40`) → `resources.gpus = N / nodes` (N defaults to 1; `E_INVALID_SPEC` if not divisible) and `gpu_types=[type]`; `--gpus-per-node`/`--gres=gpu[:type]:N` → `gpus=N`, `gpu_types`; `--mem*` → `resources.mem` (dropped with a warning on `no_mem_flag` partitions like every spec `mem`); `-o/-e` → `spec.stdout/stderr`; `-D` → `workdir` default; `-d` → `depends_on` only when it names our handles (`j12`); a raw `-d afterok:615408` is refused (`E_DEPENDENCY: dependency on raw SLURM id 615408 — fix: use the tracked handle or adopt it with job_status(['trace:615408'])`); spec fields given explicitly win over script directives (`warnings` lists each override). Directives not in the list (`--time-min`, `--nice`, `--licenses`, `--tmp`, …) are kept verbatim in `extra_sbatch`. The stripped directives are echoed in `stripped_directives`. The remainder (everything after the last leading `#SBATCH`/comment/blank line, shebang dropped) becomes `user_body.sh`. Golden test: the two real scripts in `docs/clusters.md` (TRACE `train_wobl.job`, Bridges-2 `grpo_h100.sbatch`) must render to a `job.sbatch` with **no** `-p/--gpus/--gres/--mem/-o/-e/--qos/--time` lines and to the expected CLI args per target.
2. **Resolve dependencies semantically in the ledger** (each entry `j12` or `<type>:j12`, default `afterok`): the dependency must be a tracked handle (`E_UNKNOWN_ID` otherwise). If its `last_seen_ts` is older than `MinJobAge/2`, run one tick first so the ledger is fresh. Then per entry: dependency **terminal** → evaluate now and emit nothing: `afterok` on `COMPLETED` ⇒ satisfied (omit); `afterok` on any other terminal state ⇒ `E_DEPENDENCY: j12 ended FAILED; afterok can never be satisfied — fix: resubmit j12 or use afterany/afternotok`; `afternotok` on `COMPLETED` ⇒ refuse likewise, on other terminal ⇒ satisfied; `afterany` on any terminal ⇒ satisfied; `after` on anything that started or ended ⇒ satisfied; `aftercorr` on a terminal array ⇒ satisfied when every element `COMPLETED`, else refused. Dependency **live** (in SLURM, or still `QUEUED`/`SUBMITTING` on our side) → pin `cluster` to the dependency's current attempt cluster (`E_DEP_CROSS_CLUSTER` if an explicit target disagrees) and emit `--dependency=<type>:<slurm_id>`; a dependency without a slurm id yet (`QUEUED`/`SUBMITTING`) makes this job `QUEUED(why="waiting for j12's id")` until the tick sees it. `--kill-on-invalid-dep=yes` accompanies every `--dependency` (Bridges-2's `kill_invalid_depend` does the same; harmless twice). `jobs.depends_on_json` records `[{handle, type, resolved_slurm_id}]`; the Monitor keeps the SLURM-side list current when a dependency's id changes (§5.2 step 11). Never emit a dependency on an id that is not in controller memory (it fails with `Job dependency problem`, research_6 §5).
3. **Allocate handle and paths** inside one `BEGIN IMMEDIATE`: `handle=j17` (`counter.handle`), `attempt_no=1`, `token=t-<12hex>`; for the *initial* cluster (explicit target/plan/pinned dependency cluster, else the cluster holding the inputs if any, else the first enabled cluster — auto placement may change it in step 6, in which case paths are re-derived) `ctrl_root=<control_root(cluster)>/jobs/j17`, `ctrl_dir=<ctrl_root>/a1`, `workdir = spec.workdir | script --chdir | <remote_root(cluster)>/<name>`; `stdout_pattern/stderr_pattern` = the spec/script patterns resolved to absolute paths against `workdir` (default `<ctrl_root>/out/slurm-%j.out`, arrays `%A_%a`). Insert `jobs` (state `UPLOADING` if inputs else `SUBMITTING`; `QUEUED` when the cap check of §8 already says no slot on an explicit target) and `attempts` (state `INTENT`, `cause=initial`). Commit and **return** `SubmitResult(state=…)` to the caller's `wait_s` loop; the following steps run in the task.
4. **Helper deploy** if `caps.<cluster>.helper_sha8` differs from the packaged bundle (§6.1 last block). **Inputs**: a transfer of kind `inputs` per `InputSpec` (§5.5) into `workdir`; the job waits in `UPLOADING` (progress forwarded to the waiting tool call); failure → attempt `FAILED`, job `FAILED` with `reason="inputs: <error>"`, event `submit_failed{error_code:"E_UPLOAD"}`.
5. **Placement**: explicit target string → skip. `plan_id` → reuse its options if < 15 min old (re-run `--test-only` only for the chosen target). Else `placer.rank(spec, candidates)` (§8): feasibility from the discovery cache and snapshot, ≤ 4 `sbatch --test-only` (one exec per target, concurrently under the channel semaphore, each with `cmd_timeout_s`), score, choose the best feasible option. No feasible option → attempt `FAILED`, job `FAILED`, `submit_failed{error_code:"E_NO_TARGET"}` listing each candidate's reason. If the chosen cluster differs from the one paths were derived for in step 3, re-derive `ctrl_root/ctrl_dir/workdir/patterns` for the new cluster (still attempt 1) and stage inputs there. A cap (§8 "hold locally") turns the job into `QUEUED{target, why}` with the target fixed; the Monitor resumes at step 6 when a slot frees. Attempt `target_json` is set now.
6. **Render** `job.sbatch`, `user_body.sh` (script mode), `env.sh`, `spec.json` (§6.3) **without** target-specific directives — partition/qos/account/gres/time/signal/requeue/mail/-o/-e are always passed on the `sbatch` command line so one script serves every candidate. `mkdir -p` the directories of `stdout_pattern`/`stderr_pattern` and `workdir` via SFTP (a missing `-o` directory makes the job fail at launch with its output lost, research_6 §5). Write the files to `ctrl_dir` via SFTP (`makedirs`, `.tmp` + `posix_rename`).
7. **Submit** (`idempotent=False`): set `attempts.state=UNCONFIRMED`, `invoked_local=now` **before** the call; then `cd <workdir> && bash <bin>/submit.sh <ctrl_dir> <token> -- <sbatch args> <ctrl_dir>/job.sbatch` (§6.3) with `cmd_timeout_s`. First stdout line `JOBID <id>` → `attempts.slurm_id`, `state=ACTIVE`, `confirmed_local`, `stdout_path/stderr_path` = patterns expanded (§6.3); `jobs.state=SUBMITTED`, `submit_line`; event `submitted`; `Monitor.kick(cluster)`. First line `ERR <rc>` with `rc ∈ {1, 2}` (sbatch rejected the job / no ctrl dir) → stderr through the error map (§6.3); auto placement retries the **next** option once on `E_QOS*`, `E_PARTITION*`, `E_NODE_CONFIG`, `E_SUBMIT_LIMIT` (the failing target gets `target_stats.infeasible_until_local = now+30min`; a new attempt row, the old one `FAILED`) and reports the fallback in `summary`; otherwise attempt `FAILED`, job `FAILED`, event `submit_failed`. **Ambiguous** — `ERR 3` (lock timeout: another `submit.sh` for this ctrl_dir may still be running), a first line that is neither `JOBID` nor `ERR`, empty output, `ConnectionLost`, `CommandTimeout`, `exit_status None` — → attempt stays `UNCONFIRMED`, job stays `SUBMITTING`, the tool (if still waiting) returns `state="SUBMITTING"` honestly, and the Monitor recovers (§5.2 step 9). `submit.sh` itself retries only controller-timeout errors, guarded by the `jobid` file.
8. Finish: the awaiting tool call (if any) gets the final `SubmitResult`; the `submitted` event carries the same fields for a caller that already returned. Arrays: one handle, `array_size`, elements addressed `j18[7]`; `array_tasks` rows are created lazily from squeue `-r` output. A `SubmitTask` that dies with an unexpected exception leaves the attempt in `INTENT` (never invoked) or `UNCONFIRMED` (invoked); the Monitor's sweep (§5.2 step 10) turns stale `INTENT` rows into `FAILED + needs_attention{why:"submit_stuck"}` after 10 min and never touches `UNCONFIRMED` rows except through recovery.

### 5.2 Monitor tick and reconciliation

One asyncio task per cluster. Cadence (`profile.poll`): `base_s` (60); `min_s` (30) while any job is `COMPLETING`, has `cancel_requested_ts`, is `SUBMITTING`/`UNCONFIRMED`/`QUEUED`, started < 5 min ago, has a running-job time-left < 10 min, an allocation has a foreground command outstanding with a waiter, or a rebalance move happened < 5 min ago; `max_s` (120) when nothing is live (then only the cluster snapshot refreshes, every 10 min). Jitter ±10 %. `kick(cluster)` forces a tick within 5 s. On SSH failure: after 3 consecutive failed ticks emit `cluster_unreachable` once (with `requires_vpn_hint` when the TCP probe fails), backoff 30 s → 5 min, no state changes; `cluster_recovered` on the next success.

Before every tick the Monitor **renews the lease** (`UPDATE lease SET renewed_local=now WHERE name='monitor' AND token=<mine>` inside `BEGIN IMMEDIATE`; 0 rows updated ⇒ `LeaseLost`: stop all Monitor tasks, `clusters().monitor="lost to pid N"`, event `needs_attention{why:"lease_lost"}`, and the process serves tools from the ledger only). If the local monotonic clock or the `ClusterClock` jumped by > 120 s since the previous tick (laptop sleep), the lease is **re-acquired** (§5.8 rules) before anything else. All writes of a tick happen through `store.write_fenced(token, …)`; a tick whose writes fail with `LeaseLost` discards its observations.

Tick = one composite command (§6.2) framed by `::NOW … ::END`; output without `::END` or with a non-zero `::RC` after `::SQUEUE` or `::SACCT` is discarded (counts as a failed tick). `ClusterClock` is updated from `::NOW`. Per tracked live job (and per array task) build the observation `O = {squeue_row?, restart_cnt?, sacct_rows[] (with -D duplicates), files: {jobid?, status.json?, heartbeat?, progress.json?}}` and apply, in this order:

1. **sacct terminal truth.** If the current incarnation's sacct row (rule: the row with a non-terminal state if any, else the row with the latest `End`; `incarnations = len(rows)`) has a terminal state **and** `End != Unknown/None`: finalise (§5.3 mapping), unless the state is `PREEMPTED`/`NODE_FAIL`/`REQUEUED` **and** the job is still in squeue (or a newer incarnation row exists) — that is a requeue, not termination (step 4).
2. **squeue live state** (`%T` long names → `SLURM_STATE_MAP`). `PENDING`: update `reason`, `est_start_ts` (`%S`, `N/A` → null), emit `held` once when the reason class becomes `held`. `RUNNING` first seen: `start_ts` (`%S`), `node` (`%N`), event `started{wait_s}`, insert `wait_history(source="observed")`. Terminal names inside squeue (jobs within `MinJobAge`: 300 s TRACE, 200 s Bridges-2): set `COMPLETING` with the provisional state and wait for sacct to agree for ≤ 2 ticks, then finalise from squeue.
3. **In neither.** If `status.json.phase == "exited"` → finalise from it (`COMPLETED` if rc 0, `TIMEOUT` when `cause == "timeout"`, `CANCELLED` when `cause == "cancel"`, `FAILED` otherwise), `payload.source="helper"`; sacct confirmation later fills exit code/cost and may **upgrade** `FAILED` to `TIMEOUT` per §5.3 (never the reverse). Else if the attempt is < 120 s old (cluster time) → keep (dbd/ctld lag). Else `stale_ticks += 1`; at 3 → run `scontrol -o show job <id>` once: `Invalid job id specified` → `LOST` with hint `run_command('sacct -j <id>')`; otherwise parse `JobState`/`Restarts`/`Reason` and continue.
4. **Requeue detection** (any of): `RUNNING → PENDING` transition; `restart_cnt` (from `-O RestartCnt`) > `jobs.restarts`; `incarnations − 1 > jobs.restarts`; `status.json.restart` > `jobs.restarts`. Cause: previous incarnation row `PREEMPTED` → `preempted`; `NODE_FAIL` → `node_fail`; `status.json.cause == "timeout"` → `timeout`; else `requeued` with `cause="requeue"`. Set `restarts = max(signals)`, state `SUBMITTED`, emit the cause event (non-terminal, `requeued: true`) then `requeued{cause, restarts}`. Guards: `restarts > spec.max_restarts` → `scontrol hold` + `needs_attention{why:"max_restarts"}`; ≥ 3 requeues within 10 min each ending rc≠0 → hold + `needs_attention{why:"requeue_loop"}`.
5. **Cancel grace**: `cancel_hard_ts` set, `now ≥ cancel_hard_ts` and the job still live in squeue → `scancel <id>` (plain), `cancel_hard_ts = null`. A job that already left squeue needs nothing.
6. **Allocations**: `status.json.phase ∈ {"ready", "running"}` first seen → `alloc_ready=1`, event `alloc_ready{node, end_ts}` (the agent writes `ready` once, immediately before its event loop, then `running` every second, so a ~30 s tick would essentially never catch `ready` alone; any live phase means the node is ours, only `exited` does not) (`end_ts = start_ts + hours`); `end_ts − now < 600` → `alloc_expiring` once; terminal → `alloc_ended`, outstanding `alloc_cmds` → `aborted`. `::CMDS` rc files → `cmd_done` (state `done`, or `killed` when a kill was requested).
7. **Files**: `heartbeat` → `heartbeat_ts`; `progress.json` (last line) → `progress_json`; `heartbeat_age > 900 s` while RUNNING → `needs_attention{why:"heartbeat_stale"}` once (the allocation agent refreshes its heartbeat every second even while a foreground command runs, §7.3, so this fires only for a wedged agent/node).
8. **Untracked** squeue rows (mine, not in `attempts.slurm_id`): first match them against `UNCONFIRMED` attempts — a row whose `%o` (Command) equals `<ctrl_dir>/job.sbatch` (unique per attempt) or whose `%k` (Comment) starts with `slurm-mcp:<handle>:<attempt>:<token>` **confirms that attempt immediately** (`slurm_id`, `ACTIVE`, `submitted` event), long before slurmdbd catches up. The rest → `kv.untracked.<cluster>`.
9. **Unconfirmed submits** (`attempts.state == UNCONFIRMED`, not confirmed by step 8): `::FILES` shows `<ctrl_dir>/jobid` → confirm. Else the tick appends `sacct -n -P -X -S now-2hours -o JobIDRaw,Submit,State,WorkDir,SubmitLine` (`::RECOVER` section) and adopts the row whose `SubmitLine` contains `<ctrl_dir>/job.sbatch` or `slurm-mcp:<token>`; two matches (or a squeue match plus a different sacct match) → cancel the newer, event `needs_attention{why:"duplicate_cancelled"}`. The failure deadline counts **healthy observation time only**: 15 min of ticks in which `::SQUEUE` and `::SACCT`/`::RECOVER` all returned rc 0 and the connection was up (ticks with `Socket timed out`, non-zero `::RC`, or a failed connection do not count, so a lagging slurmdbd/busy slurmctld extends the window). Only then, and only if no untracked row with `%Z` (WorkDir) == `workdir` and a `Command` under `<ctrl_root>` exists, → attempt `FAILED(reason="submit_unconfirmed")`, job `FAILED`, `needs_attention{why:"submit_unconfirmed"}` with the hint to check `squeue --me`. Never resubmitted automatically; a stale `<ctrl_dir>/.submit.lock` is removed at that point.
10. **Submit bookkeeping**: `INTENT` attempts older than 10 min whose handle has no live `SubmitTask` in this process (always the case after a restart) → `FAILED(reason="submit_stuck")`, job `FAILED`, `needs_attention{why:"submit_stuck"}` (safe: `INTENT` means `submit.sh` was never invoked). `QUEUED` jobs whose target now has a free slot (§8 caps, evaluated against this tick's `::SQUEUE`) → resume their `SubmitTask` at §5.1 step 6, oldest first, at most `(free slots)` per tick.
11. **Dependencies**: for every live job with `depends_on_json`, if a dependency's current `slurm_id` differs from `resolved_slurm_id` (rebalanced) → `scontrol update JobId=<dependent id> Dependency=<rebuilt list>` (user-settable, research_6 §4.2), record the new id, event `dependency_updated`; if a dependency became terminal in a way that can never satisfy the type (`afterok` + not `COMPLETED`, `afternotok` + `COMPLETED`) → `scontrol hold` + `needs_attention{why:"dependency_unsatisfiable"}` **before** `kill_invalid_depend` cancels it (the hold keeps the job; the hint proposes `job_control(release)` after fixing, or cancel). Jobs with live dependents are never rebalanced without this repointing (§5.4).
12. **Enrichment** for newly terminal jobs (batched ≤ 20, `::ENRICH` on the next tick): steps' `MaxRSS`, `ReqMem`, last stdout line; OOM heuristics (§5.3); `cost_actual_su`; `cost_actual_su > cost_est_su × 1.1` after a restart → `needs_attention{why:"restart_cost"}` once per job. Then rebalance scheduling (§5.4), human notifications (§5.6), and `EventBus.notify_all()`.

Events are emitted once per transition, never per tick (`transition(old, new)` in `slurm/states.py` rejects illegal moves and duplicates).

### 5.3 Preemption / requeue / timeout / OOM / cancel classification

| Signal | Source | Result |
|---|---|---|
| Preemption with requeue (TRACE `batch`, `PreemptMode=REQUEUE`, job has `--requeue`) | previous `-D` row `PREEMPTED`; job back in squeue PENDING; `RestartCnt` up | `preempted{requeued:true}` + `requeued{cause:"preempted"}`; state `SUBMITTED`; `restarts += 1`. **`GraceTime=0` on both clusters' partitions and QOS ⇒ SLURM sends SIGCONT+SIGTERM and SIGKILL essentially at once; no checkpoint hook can run.** The wrapper's `cause=preempt` is best-effort only; classification relies on sacct/squeue. After `max_restarts` the Monitor holds the job and suggests another target. |
| Preemption without requeue | sacct `PREEMPTED`, `End` set, not in squeue | terminal `preempted` |
| Timeout | (a) sacct `TIMEOUT` on the allocation row (fixture `bridges2/sacct_job_44809480.out`: allocation `TIMEOUT 0:0`, `.batch` step `CANCELLED 0:15`, `ElapsedRaw 28805` vs `TimelimitRaw 480` min) — `source="sacct"`; (b) **helper-classified timeout**: sacct `FAILED` (or `CANCELLED`) whose `ExitCode` is `128+signum(child_signal):0` or `0:signum(child_signal)` (e.g. `138:0`/`0:10` for USR1, `143:0`/`0:15` for TERM), **and** `status.json.cause == "timeout"`, **and** `ElapsedRaw ≥ TimelimitRaw × 60 − grace_s − 60` — `source="helper+sacct"`; (c) `status.json.cause == "timeout"` alone while sacct has no row yet — `source="helper"`, upgraded when sacct arrives. A wrapper-forwarded signal is therefore never reported as `FAILED rc=138`. | terminal `timeout{elapsed_s, time_limit_s, source}` with hint (declare `child_signal` + `checkpoint_interval_h` and `on_timeout="requeue"`, longer `time`, or a partition with a longer MaxWall); when `on_timeout="requeue"` the wrapper requeues itself first, seen as `requeued{cause:"timeout"}` |
| OOM | sacct `OUT_OF_MEMORY` on the allocation or any step; or `FAILED`/`CANCELLED` with `ExitCode 0:9`/`137` and step `MaxRSS ≥ 0.95 × ReqMem` ⇒ `oom_suspected` | terminal `oom{max_rss, req_mem}` with hint `raise resources.mem to ≥ 1.3 × max_rss` |
| NODE_FAIL | sacct/squeue `NODE_FAIL`; requeued when `--requeue`/`JobRequeue=1` (Bridges-2) | `node_fail` (+ `requeued`); failed node recorded in `target_stats.last_node_fail_node` and passed as `--exclude` on resubmits of that job for 2 h |
| Cancelled | sacct `CANCELLED by <uid>` | `cancelled{by: "agent"}` when `cancel_requested_ts` is set within 15 min, else `"user/admin"`; `DependencyNeverSatisfied` + cancelled → `by: "scheduler"` |
| Node/time-limit kill order | `KillWait` (300 s TRACE / 400 s Bridges-2) applies to time-limit and `scancel` termination: SIGTERM, then SIGKILL after `KillWait`; preemption uses `GraceTime` | documented in tool descriptions |

Injected for wrapped jobs: `--signal=B:USR1@<grace_s>` always — this is a **warning to `wrap.sh` only** (`B:` = batch shell, not the steps; SLURM may deliver it up to 60 s early); `wrap.sh` records it in `status.json` and forwards `SIG<child_signal>` to the payload's process group **only when `child_signal` is set**; with the default `child_signal=None` the payload keeps running until SLURM's own TERM at the limit (an untrapped USR1 would otherwise kill a python/torch payload with rc 138, §7.1). On TERM (scancel, time limit, preemption) `wrap.sh` forwards **TERM**, never `child_signal`. Requeue flag (`render.requeue_flag(spec, partition, caps)`): `--requeue --open-mode=append` when `spec.requeue is True`, or `spec.requeue is None` and (the chosen partition's `PreemptMode` contains `REQUEUE` or `on_timeout == "requeue"`); `--no-requeue` when `spec.requeue is False`, or `spec.requeue is None` on a cluster that **charges SUs and has `JobRequeue=1`** (Bridges-2) so a NODE_FAIL cannot silently multiply the bill; otherwise nothing (site default; TRACE `JobRequeue=0`). The resulting requeueability (`requeueable = --requeue rendered, or nothing rendered and JobRequeue=1`) is stored on the attempt and drives the worst-case cost (§8). `SLURM_MCP_TIMELIMIT_S` is exported by the server from the resolved `-t`, so the wrapper needs no RPC; `SLURM_MCP_CHILD_SIGNAL` is exported only when set.

### 5.4 Rebalancing

Runs every `policy.rebalance.interval_min` (10) per cluster tick and on `rebalance()`. Eligible: `kind=job`, `placement_mode=auto`, state `SUBMITTED`, reason class `normal` or `reservation`, not held, no *dependencies* (a job that depends on others stays where its dependency pins it), `moves < max_moves_per_job`, age since last submit ≥ `min_age_min`, cluster-wide moves in the last hour < `max_moves_per_hour`. Jobs **with** live dependents are eligible only for same-cluster moves, and the move repoints the dependents (§5.2 step 11) in the same transaction that switches the attempt. `est_wait_now_h` = squeue `%S` (backfill's own estimate) if not `N/A`, else the last `--test-only`, else `unknown_wait_h`. Alternatives = `placer.rank` excluding the current target (test-only on the top 2 only). Move when `gain_h ≥ min_gain_h + hysteresis_h`, `cost_delta_su ≤ max_extra_su` (worst-case costs, §8), and caps hold. Cross-cluster moves require portable inputs (`spec.inputs` with local sources, or `command`-mode jobs with no inputs; jobs with remote `script_path`, an explicit `workdir`, or a `--chdir` are marked `infeasible: workdir not portable`).

Every move is a `MoveTask` holding the Monitor's fencing token (`write_fenced` for each step; `LeaseLost` aborts the move before the next side effect). Order (never lose the job): (1) insert attempt `n+1` (`cause=rebalanced`, new token) with **paths derived for the target cluster** — `ctrl_root=<control_root(target)>/jobs/<handle>`, `ctrl_dir=<ctrl_root>/a<n+1>`, `workdir=<remote_root(target)>/<name>` (same as the old one only when the cluster is unchanged), `stdout/stderr_pattern` re-resolved against the new workdir, `excluded_nodes` = the **target cluster's** `target_stats.last_node_fail_node` only (a TRACE node name is never sent to Bridges-2); stage inputs into the new workdir if another cluster (`inputs` transfer, incremental against the new scope); deploy helpers there if needed; render + `submit.sh` — **submit first**; (2) if the submit fails or is unconfirmed, leave the old attempt alone (the job's current attempt is still the old one, so `job_logs`/`collect_results`/`cancel` keep working there) and mark the new one `FAILED`/`UNCONFIRMED` (recovery §5.2.9 may still confirm it later; then step 3 runs at that time); (3) re-check the old job with `squeue --me -h -o '%A|%T'` filtered client-side for `<old>` (`idempotent=True`; `-j` exits 1 for a purged id, research_6 §1.1): **rc ≠ 0, missing `::END`, or `Socket timed out` ⇒ unknown: abort the move, keep both attempts, retry the re-check on the next tick** (neither side is cancelled while the state is unknown; a `needs_attention{why:"duplicate_cancelled"}` follows only when the recovery sweep later proves both ran); row `PENDING` ⇒ `scancel <old>`, old attempt `SUPERSEDED`, `jobs.attempt_no ← n+1` (which flips every path/cluster field of `jobs_current`), `moves += 1`, event `rebalanced{new_workdir}`, dependents repointed; row `RUNNING`/`COMPLETING`/terminal or no row at all (started and finished, or purged) ⇒ cancel the **new** job instead (`scancel <new>`, attempt `SUPERSEDED`), report "started meanwhile". Age-priority loss is negligible on both clusters (`PriorityWeightAge=1e4` vs `FairShare=1e6`). Etiquette caps: TRACE `bf_max_job_user=10` is per user cluster-wide ⇒ keep own pending jobs per cluster ≤ `bf_max_job_user − 2`; Bridges-2 `bf_max_job_user_part=20` ⇒ per partition ≤ 18; both discovered from `SchedulerParameters`; a job that would exceed the cap at the destination is not moved (it is never re-placed *because* of a cap either, §8). Scenario test (§10): move `j17` TRACE → Bridges-2, then `job_logs('j17')`, `collect_results(['j17'])` and `job_control(['j17'],'cancel')` must all address the Bridges-2 paths/connection and `cancel.requested` must land in the Bridges-2 `ctrl_dir`.

### 5.5 Upload / download / collect (transfer.py)

All transfers are `TransferManager` tasks with a `transfers` row and per-file `transfer_files` rows; the tool awaits up to `wait_s`, then returns `state="running"`. Host role: `transfer` when `profile.transfer_host` is set (Bridges-2 DTN, TRACE `data.trace.cmu.edu`), else `login`. Capabilities per transfer host (bootstrap): `exec_ok` (`echo ok` over an exec channel; TRACE's DTN is a restricted shell → False; Bridges-2's HPN-SSH DTN → probe), `sftp_ok`, `port` (banner probe), `mb_per_s` (4 MB SFTP put, for ETAs only).

Upload plan: walk the local tree in a thread applying ignore rules → `(rel_path, size, mtime_ns, sha1 for files ≤ 64 MB)` → diff against `manifests` scope `up:<cluster>:<remote>` → delta. **Windows client rules** (`textio`, unit-tested with fabricated CRLF/backslash inputs and on a Windows CI job): every `rel_path` (manifests, `transfer_files`, tar arcnames, remote targets) is `str(PurePosixPath(*Path(rel).parts))` — never an `os.sep` string; local scope keys use `os.path.normcase(os.path.realpath(local))` so `D:\x`, `D:/x` and `d:\X` share one manifest; `manifests.scope` for uploads is `up:<cluster>:<remote>` with the remote path as given (POSIX). Text files are transferred byte-exact (no CRLF rewriting — only rendered scripts and `remote_write` are normalised); a `warnings` entry lists `*.sh|*.sbatch|*.py` files that contain CRLF (they will fail with `$'\r': command not found` / `bad interpreter` on the cluster). Quota check on the login host: `profile.quota_command` if set else `df -Pk` of the **destination** path (not `$HOME`); refuse when `free < 1.2 × delta_bytes` (`E_QUOTA` with used % and the 5 largest local entries). The governing row is chosen with `parse.df_row_for_path(caps.df, destination)` — the longest queried-path prefix, never the mount — because a filesystem with per-directory quotas reports several different totals for one mount (TRACE: 932 TB at 1 % for `$HOME` vs 3 TB at 74 % for the group volume, both on `/trace`). Mode `auto`: ≥ 16 changed files, or delta < 256 MB with any file < 1 MB ⇒ **tar**; else **sftp** (`block_size=65536`, `max_requests=64`, files ≥ 64 MB written to `<name>.part-<tid>` with `seek` resume and `posix_rename`). Tar path: local `tarfile` (uid/gid 0, `+x` for `*.sh|*.py`), then if `exec_ok` on the transfer host: `run_with_stdin_file("mkdir -p R && tar -xf - -C R", tar)` there; else SFTP `put` the tar to `R/.slurm-mcp-upload-<tid>.tar` via the transfer host and `tar -xf R/.slurm-mcp-upload-<tid>.tar -C R && rm -f …` on the **login** host (extracting an already-present file is not a transfer and is within PSC's login-node policy; capped 2 GB/2000 files per call). `manifests` rows are written per file as they complete; a killed call or restart re-plans from `transfer_files` and skips `done` rows.

Download: expand globs with SFTP `glob()` (supports `**`; `SFTPNoSuchFile` → empty) → `[{path, size, mtime}]` → skip files with `mtime > now−15 s` → diff against `manifests` scope `down:<normcase(realpath(local_dir))>` → ≥ 16 files ⇒ `tar -cf - -C <root> -T -` (file list on stdin) on the exec-capable host (transfer host if `exec_ok`, else login host for ≤ 256 MB, else per-file SFTP) streamed to a local temp file, then extracted **member by member** (`tar.extract(member, path=stage, filter="data")` after rewriting `member.name` through `textio.local_safe_name()`; `extractall` is not used because one NTFS-invalid name would abort the whole archive) into `<local_dir>/.stage-<tid>` then moved per file; else SFTP `get` (4 files concurrently, `.part` resume) to the safe local name. Local paths are opened through `textio.long_path()` (`\\?\` prefix on Windows) so > 260-char results work. Every rename is recorded in `transfer_files.local_name` and returned as `renamed`. Collect = download of `spec.outputs` + stdout/stderr (+ `progress.json`) **from the current attempt's cluster and workdir** (`jobs_current`) with `kind=collect`, `handle` set, followed by the enrichment summary.

### 5.6 Events, notifications, `wait_for_events`

`EventBus.append()` inserts inside the caller's fenced transaction and calls `Condition.notify_all()` after commit. Per-client state: `cursor.<client_id>` (ack floor `F`: every `seq ≤ F` acknowledged), `event_acks(client_id, seq)` for `seq > F`, and `deliveries(client_id)` = the last delivery `{next_seq, seqs}`. `wait_for_events` algorithm: (1) if `ack_seq` is given and equals `deliveries.next_seq` → insert every seq of that delivery into `event_acks`, then raise `F` through the contiguous acknowledged prefix and prune (`acked = n`); an `ack_seq` that does not match is ignored with a warning (never acks by range, so a stale number cannot swallow events); (2) `start = since_seq or F + 1`; read events `seq ≥ start` **not in `event_acks`**, apply `kinds`/`job_ids` filters, take ≤ `max_events` → if any: record `deliveries = {next_seq: max(seq)+1, seqs}` and return them with `delivered_seqs`, `next_seq`, `unread_events` (= all unacknowledged events for this client), `unread_unmatched` (= unacknowledged events excluded by the filters); else `await cond.wait_for(..., timeout=30)`, `report_progress`, repeat until `timeout_s`. Properties: returning never consumes; the same `since_seq` re-delivers the same events; a dropped/compacted task-notification result is replayed on the next unacknowledged call; filters never hide an event permanently (`unread_events` stays accurate and an unfiltered call returns it). Clients: `client_id` defaults to `ctx.client_id` (when the client sends one) or the server's `session_id` (random 8 hex generated at lifespan start and shown by `clusters()`); a new `client_id` starts with `F = max(seq)` at first use minus nothing — i.e. it sees only events appended after its first call — **except** the `session_id` client, whose `F` is initialised to `kv.cursor.last_session` so events emitted while no server was running (`observed_late`) are delivered to the next session; the CLI uses `client_id="cli"`. `notify.py` runs after each tick: for kinds in `notify.toast_kinds` a Windows toast (`win11toast.notify` in a thread; title `slurm j17 COMPLETED (bridges2)`, body = summary; coalesced to one toast per 10 s: "3 jobs finished"), webhook POST (JSON event, 5 s timeout, 3 retries, thread) for `webhook_kinds`; `events.notified=1` on success so restarts never re-toast. Missed events at startup (< 24 h, `notified=0`) are delivered as one coalesced toast. Quiet hours suppress toasts, never ledger events. `--mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90 --mail-user=<email>` is injected when `notify.email` is set (`MailProg=/bin/mail` exists on both clusters).

### 5.7 Allocation reuse

`allocate` submits `alloc.sbatch` (§6.3) whose body is `exec <bin>/alloc-agent.sh <ctrl_dir>`; the agent (§7.3) runs numbered command files on the allocated node inside the batch step (all allocated CPUs/GPUs, `CUDA_VISIBLE_DEVICES`, `$LOCAL`). `alloc_run` writes `cmds/NNN.sh` atomically (`.tmp` + `posix_rename`; `NNN` zero-padded, monotonic per allocation from `alloc_cmds.n`), then polls `cat NNN.rc 2>/dev/null; tail -c 4096 NNN.out` on the login node every 3 s while waiting. Detached commands (`NNN.bg.sh`) return immediately; their completion is a `cmd_done` event from the tick. `job_control(["a3.c3"], "cancel")` creates `cmds/003.kill` for a foreground command and `cmds/003.bg.kill` for a detached one (`alloc_cmds.kill_path`, the `<base>.kill` rule of §7.3); the agent honours it within its 1 s loop even while another foreground command runs. Release = `job_control(["a3"], "cancel")` → `<ctrl>/release` then `scancel`. NFS attribute caching on TRACE (VAST) may delay visibility of a new `.sh` by up to `acdirmax` (typically ≤ 60 s); Lustre on Bridges-2 is coherent — `alloc_run`'s summary mentions this when `seconds > 15`. `idle_release_min > 0` makes the agent exit after that many idle minutes (the job ends, `alloc_ended`). `srun --jobid=<id> --overlap …` stays available through `run_command` for multi-node steps.

### 5.8 Server restart recovery

Lifespan start (must answer `initialize` within 30 s): open SQLite, migrate, load policies, create transports lazily (no SSH in the lifespan), generate `session_id`, then **acquire the lease**: inside `BEGIN IMMEDIATE`, read `lease WHERE name='monitor'`; acquire (`token += 1`, `owner_pid/host = me`) when there is no row, or `renewed_local` is older than 5 min **and** the owner pid is not alive (`psutil.pid_exists`, same host) — a lease whose pid is alive is never taken over automatically (the owner may be a suspended laptop process that will resume; `slurm-mcp monitor takeover --force` exists for a human); otherwise this instance runs no Monitor and serves tools from the ledger, `clusters().monitor = "held by pid N"`, and re-checks every 60 s so it takes over when the holder exits. The holder renews every tick (§5.2) and at least every 60 s when idle. Fencing: every Monitor write carries the token; a holder that wakes from sleep sees its next renew return 0 rows (if another process took over after 5 min because it had died — or, with `--force`, while alive), stops, and never writes again, so two Monitors never reconcile the same ledger (no duplicated terminal events, hard cancels or double moves). A holder that lost the lease while a `SubmitTask`/`MoveTask` was mid-flight: the task's next fenced write raises `LeaseLost`; the attempt stays `INTENT`/`UNCONFIRMED` for the new holder's sweep/recovery to settle (never two `sbatch` for one attempt: `submit.sh`'s `jobid` file is the cross-process guard). Start Monitor tasks with `startup_sweep=True`. First tick per cluster: TCP probe → connect → verify helper sha8 → normal reconciliation for every live job and attempt (transitions missed while down are emitted with `payload.observed_late=true`; terminal truth comes from `sacct -D`) → step 8/9 confirmation for `UNCONFIRMED` attempts → `INTENT` sweep → re-issue hard cancels whose `cancel_hard_ts` elapsed → resume `QUEUED` jobs and `running` transfers from `transfer_files` → re-poll outstanding `alloc_cmds` → coalesced missed notifications. Discovery cache older than 24 h refreshes lazily. The CLI (`slurm-mcp jobs|status|wait`) never takes a lease held by a live pid: with a live holder it reads the ledger (fresh within one poll interval); with no live holder it acquires the lease for one synchronous tick and releases it (`DELETE … WHERE token=<mine>`), so a human never sees stale data and never races the server.
## 6. SLURM command contracts (22.05.11, verified against `tests/fixtures`)

### 6.0 Conventions

Every remote command runs via `bash -lc` (so `module`/`sbatch` are on PATH) and begins with `export SLURM_TIME_FORMAT=%s LC_ALL=C;`. Composite commands are framed with `echo '::<SECTION>'` lines and end with `echo '::END'`; each SLURM command is followed by `echo "::RC $?"`. Field separator is `|` (`-P`/`-o` with explicit `|`), rows split on `\n`, empty output is valid. Free-text fields that may contain spaces or commas (`Reason`, `WorkDir`, `SubmitLine`) are always the **last** fields of a format so a stray comma cannot shift columns (`|` never appears in SLURM values). Timestamps: with `SLURM_TIME_FORMAT=%s` honoured (capability `epoch_format`, probed at bootstrap via `BOOT_TIME`), `%S/%V/%e`, `Submit/Start/End`, `StartTime` and the `--test-only` estimate are epoch integers; otherwise ISO `YYYY-MM-DDTHH:MM:SS` in cluster-local time, converted by `ClusterClock.to_epoch()` using the discovered `tz_offset_s`. Sentinels to handle everywhere: `N/A`, `Unknown`, `None`, `None assigned`, `(null)`, `UNLIMITED`, `Partition_Limit`, `INVALID`.

### 6.1 Discovery (bootstrap; cached 24 h in `kv.caps.<cluster>`; forced by `clusters(refresh=True)`)

```
export SLURM_TIME_FORMAT=%s LC_ALL=C
echo '::ENV'; echo "$HOME|$USER|$(hostname -f)|${PROJECT:-}|${SCRATCH:-}|${LOCAL:-}|$(date +%s)|$(date +%z)|$(id -gn)"
echo '::VERSION'; sinfo --version
echo '::CONFIG'; scontrol show config | grep -E '^(ClusterName|SLURM_VERSION|MinJobAge|MessageTimeout|PreemptMode|PreemptType|PreemptExemptTime|PreemptParameters|GraceTime|JobRequeue|KillWait|MaxArraySize|MaxJobCount|SchedulerParameters|DefMemPerCPU|DefMemPerNode|MaxMemPerCPU|MailProg|AccountingStorageEnforce|AccountingStoreFlags|EnforcePartLimits|PriorityWeight(Age|FairShare|QOS|Partition|JobSize)|BOOT_TIME) '
echo '::PARTITIONS'; scontrol show partition -o; echo "::RC $?"
echo '::SINFO'; sinfo -h -e -N -o '%N|%R|%t|%c|%m|%G|%f'; echo "::RC $?"
echo '::USER'; sacctmgr -nP show user "$USER" format=User,DefaultAccount
echo '::ASSOC'; sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS,GrpTRES,GrpTRESMins,MaxJobs,MaxSubmit,MaxTRES,MaxWall; echo "::RC $?"
echo '::QOS'; sacctmgr -nP show qos format=Name,Priority,GraceTime,MaxWall,MaxTRES,MaxTRESPU,MaxJobsPU,MaxSubmitPU,GrpTRES,Preempt,PreemptMode,Flags,UsageFactor; echo "::RC $?"
echo '::SSHARE'; sshare -nP -U -u "$USER" -o Account,User,FairShare,GrpTRESMins,GrpTRESRaw; echo "::RC $?"
echo '::RESV'; scontrol -o show reservation 2>/dev/null
echo '::TOOLS'; for t in tar sacct squeue sbatch scontrol sinfo sacctmgr scancel srun sshare sha256sum stat timeout setsid rsync jq seff flock; do printf '%s=' "$t"; command -v "$t" >/dev/null 2>&1 && echo 1 || echo 0; done
echo '::CAP_O'; squeue --me -h -t all -O 'JobID:0|,RestartCnt:0|,tres-per-node:0|,tres-per-job:0|' >/dev/null 2>&1; echo "rc=$?"
echo '::DF'; for p in "$HOME" <remote_root> <control_root> "${PROJECT:-}" "${GROUP:-}" <profile.quota_paths…>; do [ -n "$p" ] && [ -d "$p" ] && df -Pk "$p" 2>/dev/null | tail -n +2 | sed "s|\$| $p|"; done
echo '::BALANCE'; <profile.balance_command> 2>/dev/null | head -40        # only when configured (Bridges-2: projects)
echo '::HELPER'; cat <control_root>/bin/VERSION 2>/dev/null
echo '::END'
```

Parse rules (`slurm/parse.py`, golden-tested on `scontrol_config.out`, `scontrol_partitions.out`, `sinfo_nodes.out`, `sacctmgr_*.out`, `sshare_me.out`, `tools.out`, `df_home.out`, `env.out`):

- `::ENV` → `home, user, hostname, project, scratch, local, remote_now, tz_offset ("-0400" → -14400 s), group`. `::CONFIG` → `Key = Value` (regex `^(\w+)\s*=\s*(.*)$`); `BOOT_TIME` all digits ⇒ `caps.epoch_format=True`; `MinJobAge`/`KillWait`/`MessageTimeout` are `N sec` (`caps.cmd_timeout_s = profile.cmd_timeout_s or max(120, MessageTimeout + 60)`: 260 on Bridges-2, 120 on TRACE); `PreemptMode` may be a comma list (`GANG,REQUEUE`); `JobRequeue` `0|1` (⇒ `caps.job_requeue`); `SchedulerParameters` split on `,` into a dict (`bf_max_job_user=10`, `bf_max_job_user_part=20`, `kill_invalid_depend`, `defer`, `max_rpc_cnt`); `AccountingStoreFlags` contains `job_comment` ⇒ `caps.comment_stored`.
- `::PARTITIONS` (`scontrol show partition -o`): one line per partition, `Key=Value` tokens split on single spaces (partition values never contain spaces; `(null)`/`N/A` → None). Keys used: `PartitionName, AllowGroups, AllowAccounts, AllowQos, Default, QoS, DefaultTime, GraceTime, MaxNodes, MaxTime, PriorityTier, OverSubscribe, PreemptMode, State, TotalNodes, TotalCPUs, DefMemPerCPU, DefMemPerNode, MaxMemPerNode, MaxMemPerCPU, JobDefaults (e.g. DefMemPerGPU=63000), TRES (e.g. gres/gpu:v100-32=200), TRESBillingWeights`. Times: `UNLIMITED|NONE → None`, `D-HH:MM:SS`, `HH:MM:SS`, `MM:SS`, `MM`.
- `::SINFO` (`sinfo -h -e -N -o '%N|%R|%t|%c|%m|%G|%f'`; `-e` prevents grouping heterogeneous nodes; `%R` has no `*` suffix; one line per node per partition): `%t` compact state with suffixes `*~#!%$@^-` stripped (`idle|mix|alloc|comp|down|drain|drng|resv|maint|plnd|fail|unk`), `%G` `gpu:a40:1`, `gpu:v100-32:16(S:0-95)`, `(null)` → strip the `(…)` suffix, regex `^gpu(?::([^:]+))?:(\d+)$` → `{type, count}` (untyped `gpu:8` → type None); `%f` comma features or `(null)`. Aggregate per partition: node counts by state, gres types present with per-node counts, max cpus (`%c`), max mem MB (`%m`).
- `::USER` → `DefaultAccount`. `::ASSOC` (11 fields; may be several rows) → `account, partition (empty = any), qos_list (comma), default_qos, grp_tres, grp_tres_mins, max_jobs, max_submit, max_tres, max_wall`. TRES strings `cpu=24,gres/gpu=16,node=8` → dict.
- `::QOS` (13 fields) → per QOS `priority, grace_time_s, max_wall_s, max_tres (per job), max_tres_pu, max_jobs_pu, max_submit_pu, grp_tres, preempt, preempt_mode, flags, usage_factor`. Fixture check (`sacctmgr_qos.out`, captured with `format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu,grptres,maxtres,flags`): `gpusmallpartition|0|08:00:00||2|10||gres/gpu=16|DenyOnLimit` ⇒ GPU-small `MaxWall=8h, MaxJobsPU=2, MaxSubmitPU=10, MaxTRES(per job)=gres/gpu=16`; `rmsharedpartition` `MaxWall=3-00:00:00, MaxTRESPU=cpu=25600, MaxTRES=cpu=64,node=1`; `gpupartition` `MaxWall=2-00:00:00, MaxTRES=gres/gpu=64,node=8`; `gpusharedpartition` no per-job TRES limit (the "max 4 GPUs" rule is a job_submit plugin, surfaced only as the `GPU-shared maximum is 4` submit error); `low` flags `NoReserve,OverPartQOS`; TRACE `batchpartition`/`priorityphase1node1` `MaxWall=2-00:00:00`, `cpuonly-debug-qos` `MaxTRESPU=cpu=24`. Empty field = no limit. The field names `MaxTRES` (per-job) and `MaxTRESPU` are QOS fields; `MaxTRES` is also the per-job field on associations (`MaxTRESPJ`/`MaxTRESPU` are **not** accepted on `show assoc`).
- **Effective limits per partition** = partition `MaxTime` ∧ QOS(`QoS=` of the partition).`max_wall` ∧ chosen job QOS `max_wall` ∧ assoc `max_wall` (None ignored). `max_jobs_pu/max_submit_pu/max_tres_pj` from the partition QOS. Charge: `TRESBillingWeights` when present, else `profile.su_rates`, else free.
- **QOS selection** (`render.choose_qos(partition)`): `spec.qos` → `profile.qos_map[partition]` → if partition `AllowQos == ALL` and the assoc default QOS is set: no `--qos`; else candidates = `AllowQos ∩ assoc.qos_list` ordered: assoc `default_qos` first, then names whose lowercase prefix matches the partition family (`gpu` for `GPU-shared`, `rm` for `RM-shared`, `em`), then `low`, then the rest excluding names containing `interact`; the first candidate that passes `--test-only` is cached in `caps.qos_for_partition[partition]`. Example outcomes: TRACE `batch` → `normal` (assoc default, `AllowQos=normal,batch`); Bridges-2 `GPU-shared` → `gpu`; `RM-shared` → `low`. `--qos` is always passed explicitly when a candidate was chosen (Bridges-2 rejects the default: `allocation failure: Invalid qos specification`).
- `::SSHARE` → `fair_share`; `GrpTRESMins`/`GrpTRESRaw` `billing=N` → `su_balance = (mins_limit − mins_used)/60`; missing (both clusters today) ⇒ `::BALANCE` + `profile.balance_regex` (`left`/`total` groups, commas stripped); still missing ⇒ `null`.
- `::RESV` → `ReservationName, StartTime, EndTime, Nodes, PartitionName, Flags`; maintenance = `MAINT` in Flags or name matching `/maint/i`. `::TOOLS` gates features (no `setsid` ⇒ alloc-agent kills only the shell; `seff` ⇒ efficiency block). `::CAP_O` `rc=0` ⇒ `caps.squeue_O_zero` (the `-O …:0|` unpadded form works **and** `tres-per-job` is accepted); else the tick omits the `::RESTARTS` section and the snapshot falls back to `%b` (tres-per-node only). `::DF`: one POSIX row per existing path, the queried path appended as the last column (`df` prints the mount point, not the argument) → `[{path, mount, kb_total, kb_used, kb_free, used_pct, role}]`; rows with the same mount are de-duplicated. On TRACE `$HOME` reports the 932 TB `/trace` volume at 1 % (fixture `trace/df_home.out`) — that is why `remote_root` and `quota_paths` (the 3 TB group volume) are always queried; `clusters()`/`quota_warning` use the row whose `path` is the destination of the operation.
- Transfer-host capabilities (separate connection): `port` (banner probe 22, then 2222 if the 22 banner lacks `hpn` and 2222 answers with one), `exec_ok` (`echo ok`), `sftp_ok`, `mb_per_s`.
- Helper deploy (on first W tool, not in discovery): if `::HELPER` ≠ packaged `sha8`, SFTP `put` `wrap.sh`, `submit.sh`, `alloc-agent.sh` to `<control_root>/bin/<sha8>/`, `chmod 755`, write `<control_root>/bin/VERSION`. Running jobs keep their directory (content-addressed, never overwritten).
- 30-day wait-history back-fill (once per cluster): `sacct -nP -X -u "$USER" -S now-30days -o JobIDRaw,Partition,QOS,ReqTRES,Submit,Start,State` → rows with `Start` not `Unknown|None` feed `wait_history(source="backfill")`, target key = `<cluster>:<partition>[:<type>]` with type from `ReqTRES` `gres/gpu:h100-80=1`.

### 6.2 Tick, enrichment, snapshot

Tick (`slurm/commands.tick(ids, ctrl_dirs, rc_paths, recover, enrich_ids)`):

```
export SLURM_TIME_FORMAT=%s LC_ALL=C
echo "::NOW $(date +%s) $(hostname -s)"
echo '::SQUEUE'; squeue --me -h -r -t all -o '%A|%i|%F|%K|%T|%P|%q|%S|%e|%V|%l|%M|%Q|%N|%b|%k|%o|%Z|%r'; echo "::RC $?"
echo '::RESTARTS'; squeue --me -h -r -t all -O 'JobID:0|,RestartCnt:0|,Requeue:0|'; echo "::RC $?"          # only if caps.squeue_O_zero
echo '::SACCT'; sacct -n -P -X -D -j <id1,id2,…> -o JobIDRaw,JobID,State,ExitCode,DerivedExitCode,Partition,QOS,NodeList,Submit,Start,End,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES,Reason,WorkDir; echo "::RC $?"
echo '::FILES'; for d in <ctrl_dir…>; do for f in jobid status.json heartbeat; do [ -f "$d/$f" ] && printf '%s|%s|%s\n' "$d" "$f" "$(head -c 1000 "$d/$f" | tr '\n\r|' '   ')"; done; [ -f "$d/progress.json" ] && printf '%s|progress.json|%s\n' "$d" "$(tail -c 1024 "$d/progress.json" | tail -n 1 | tr '|' ' ')"; done
echo '::CMDS'; for f in <rc path…>; do [ -f "$f" ] && printf '%s|%s\n' "$f" "$(cat "$f")"; done
echo '::RECOVER'; sacct -n -P -X -u "$USER" -S now-2hours -o JobIDRaw,Submit,State,WorkDir,SubmitLine; echo "::RC $?"   # only while UNCONFIRMED attempts exist
echo '::ENRICH'; sacct -n -P -j <tid1,tid2,…> -o JobIDRaw,JobID,State,ExitCode,MaxRSS,ReqMem,ElapsedRaw,AllocTRES; echo "::RC $?"; for f in <stdout path…>; do printf '::L %s|%s\n' "$f" "$(tail -n 1 "$f" 2>/dev/null | head -c 300 | tr '|' ' ')"; done   # only when there are terminal, un-enriched jobs
echo '::END'
```

`<ids>` = SLURM ids of live attempts + terminal ones not yet enriched; chunks of 100 ids produce additional `::SACCT` sections. Only wrapped RUNNING jobs, `UNCONFIRMED` attempts and live allocations contribute `ctrl_dir`s. A command longer than 8 KB is split into two execs (the second carries only `::FILES`/`::CMDS`). No `-S` is needed with `-j` (sacct then defaults the start window to epoch 0); `-S` values, when used, are relative forms (`now-2hours`, `now-30days`) or ISO `YYYY-MM-DDTHH:MM:SS` — never a bare epoch (sacct's parser rejects it).

squeue parse (`-r`: one line per array element): `%A` unique id per element, `%i` display id (`123` or `123_4`; pending collapsed arrays never appear with `-r`), `%F` array base id, `%K` array index (`N/A` if not an array), `%T` long state name, `%P` partition (comma list while a multi-partition job pends; the chosen partition first once started), `%q` QOS, `%S` start (`N/A` or epoch/ISO), `%e` end, `%V` submit, `%l` limit (`D-HH:MM:SS`, `UNLIMITED`), `%M` elapsed (`0:00`, `14:47`, `1:02:03`, `2-01:02:03`), `%Q` priority int, `%N` nodes (empty while pending), `%b` tres-per-node: `gres:gpu:h100-80:1`, `gres:gpu:1`, `gres:gpu:h100-80` (count omitted = 1) or `N/A` (**`N/A` also for every job that asked for GPUs with `--gpus`/per-job TRES** — the TRACE house style; fixtures `trace/squeue_me.out` and `squeue_all_counts.out` show all batch jobs as `N/A` although `scontrol` shows `TresPerJob=gres:gpu:a40`; verified on 22.05), `%k` comment (`(null)` when unset; `slurm-mcp:<handle>:<attempt>:<token>` for ours — used to confirm `UNCONFIRMED` attempts, §5.2 step 8), `%o` command = the batch script path (`<ctrl_dir>/job.sbatch` for ours; unique per attempt), `%Z` workdir, `%r` bare reason, last (`%o`/`%Z` may contain spaces; never `|`). `::RESTARTS` rows `615427|0|1|` → `rstrip('|').split('|')`.

sacct parse: 17 `|` fields, `WorkDir` last; `State` first whitespace token, `by <uid>` remembered (`CANCELLED by 2692968`); `ExitCode`/`DerivedExitCode` `rc:sig`; `Submit/Start/End` epoch, or `Unknown`/`None` → null; `NodeList` `None assigned` → null; `ElapsedRaw` seconds; **`TimelimitRaw` is minutes** (`1440` for `1-00:00:00`), `UNLIMITED|Partition_Limit` → null; `AllocTRES`/`ReqTRES` `billing=64,cpu=64,gres/gpu=1,gres/gpu:h100-80=2,mem=512G,node=1` → dict; `Reason` `None|Dependency|…`. Multiple rows per `JobIDRaw` come from `-D` (requeues); current incarnation = the non-terminal row if any, else the row with the latest `End`; `incarnations = len(rows)`. Array rows: `JobID` `123_4`, `JobIDRaw` the element's own id ⇒ handle `j18[4]` ↔ `(123, 4)` via `%F/%K`. An unknown id yields no row (rc 0) — except that very old ids can exist (TRACE returns `1|CANCELLED by 51559` for `-j 1`), so ids are never used as existence probes.

`::FILES` lines: `<ctrl_dir>|<file>|<content ≤ 1000 chars, newlines and | replaced>`; `status.json` is parsed with `json.loads` after restoring nothing (its values contain no newlines by construction). `::CMDS` lines `<rc path>|<rc>`. `::ENRICH`: step rows `123.batch`, `123.extern`, `123.0`; `MaxRSS` `56459172K` / `1.5G` → bytes; `ReqMem` `80Gn`/`8048Mc` (`n` per node, `c` per core × allocated cpus) or plain `128G` (22.05 prints without suffix on the allocation row when set per node) → bytes; `::L` lines carry the last stdout line.

Snapshot for `cluster_status`/placer (cached 60 s). With `caps.squeue_O_zero` (both clusters) the demand sections use `-O` so that **both** TRES views are visible — `tres-per-node` (`--gres`, what `%b` shows) and `tres-per-job` (`--gpus`, invisible to `%b`):
```
export SLURM_TIME_FORMAT=%s LC_ALL=C
echo '::NODES'; sinfo -h -e -N -o '%R|%t|%G|%C'; echo "::RC $?"
echo '::PD'; squeue -h -t PD -O 'Partition:0|,tres-per-node:0|,tres-per-job:0|' | sort | uniq -c; echo "::RC ${PIPESTATUS[0]}"
echo '::R';  squeue -h -t R  -O 'Partition:0|,tres-per-node:0|,tres-per-job:0|' | sort | uniq -c; echo "::RC ${PIPESTATUS[0]}"
echo '::MINE'; squeue --me -h -t PD -O 'JobID:0|,Partition:0|,tres-per-node:0|,tres-per-job:0|,PriorityLong:0|,StartTime:0|,Reason:0|'
echo '::RESV'; scontrol -o show reservation 2>/dev/null
echo '::END'
```
Fallback without the capability: `-o '%P|%b'` / `-o '%A|%P|%b|%Q|%S|%r'` (tres-per-job then unknown ⇒ every `N/A` row is untyped demand, below). `uniq -c` lines `   3433 GPU-shared|gres:gpu:h100-80:1|N/A|` → `(count, partition, tres_per_node, tres_per_job)`; aggregation is remote so the output is O(partitions × gres types). `%C` = `A/I/O/T` CPUs. **Demand classification** (`parse.classify_demand(row, partition_caps)`): the GPU request of a row is `tres_per_node` if not `N/A`, else `tres_per_job` if not `N/A` (`gres:gpu:a40`, `gres:gpu:2`, `gres/gpu:h100-80=2` forms all accepted: regex `^gres[:/]gpu(?::([^:=]+))?(?:[:=](\d+))?$` → `{type, count}`, count default 1); a row with **both** `N/A` in a partition whose `TRES` advertises `gres/gpu` (all TRACE `batch`/`biosimmlab` rows, 283 pending GPU-shared rows on Bridges-2 per `squeue_all_counts.out`) is **untyped GPU demand** counted against *every* gres type of that partition (`gres=None`, never zero); in a partition without `gres/gpu` it is CPU demand. Fake cluster (§10) renders `N/A` rows exactly like the fixtures.

### 6.3 Render, submit, estimate, control

Rendered `job.sbatch` — **the same shape for spec jobs and user scripts**: a user script's `#SBATCH` directives have already been converted into spec fields and removed by `parse_sbatch()` (§5.1 step 1); only its unknown directives survive as `extra_sbatch`, and its body becomes `user_body.sh`:
```
#!/bin/bash
#SBATCH -J <name>
#SBATCH -N <nodes>
#SBATCH --ntasks-per-node=<tasks>          # if tasks
#SBATCH --cpus-per-task=<cpus>             # if cpus
#SBATCH --exclusive                        # if exclusive
#SBATCH -C <constraint>                    # if constraint
#SBATCH --open-mode=append
#SBATCH <extra_sbatch line>                # each, verbatim (only directives the server does not manage)
# slurm-mcp handle=<handle> attempt=<n> token=<token> rendered=<iso_local>
source <ctrl_dir>/env.sh                   # exports SLURM_MCP_GRACE, SLURM_MCP_ON_TIMEOUT, SLURM_MCP_MAX_RESTARTS, SLURM_MCP_CHILD_SIGNAL (only if set),
                                           # SLURM_MCP_TIMELIMIT_S, user env, then: module load …; <setup>
cd <workdir>
exec <control_root>/bin/<sha8>/wrap.sh <ctrl_dir> -- bash <ctrl_dir>/user_body.sh     # user_body.sh = spec.command or the user script body
```
Files written by `render` are passed through `textio.normalize_text()` (CRLF → LF, BOM stripped, NUL refused) before the SFTP write, so a script edited in Notepad/VS Code on Windows never reaches bash with `\r` (`$'\r': command not found`, `bad interpreter`); `SubmitResult.warnings` says `crlf_normalized` when that happened. `--export` is **not** injected (default `ALL`; Bridges-2 already blocks login-env inheritance, and `--export=NONE` would break `module` functions and `srun` inheritance inside the script). Nothing target-specific, no `-t`, no `--mem`, no `-o/-e` is in the file — all of it is on the command line so one script serves every candidate and a rebalanced attempt cannot inherit stale directives. `render.target_args(target, spec, policy, attempt)` builds the command line: `-p <partitions joined by ,>`, `--qos=<q>` (§6.1 selection), `-A <account>` (spec → profile.default_account → discovered DefaultAccount), `-t <time>`, `--mem=<mem>` (unless the partition is in `profile.no_mem_flag` — then dropped with a warning, for spec and parsed-script memory alike), `--gres=gpu:<type>:<gpus>` when `gpus > 0` (per node, **always typed**, §8; `--gpus` is never sent — its count is per job and `%b` cannot see it), `--requeue --open-mode=append` / `--no-requeue` (§5.3 rule), `--signal=B:USR1@<grace_s>` (wrapped jobs; warning to `wrap.sh` only), `-o <stdout_pattern> -e <stderr_pattern>` (**always absolute**: the user's pattern resolved against `workdir`, default `<ctrl_root>/out/slurm-%j.out`, arrays `%A_%a`; the directory is created before submit), `--array=<array>[%<parallel>]`, `--dependency=<list> --kill-on-invalid-dep=yes`, `--mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90 --mail-user=<email>` (when configured), `--exclude=<nodes>` after a NODE_FAIL **on this cluster**, `--hold` when requested, `--comment=slurm-mcp:<handle>:<attempt>:<token>`, `--parsable`. `submit_line` in the result shows the exact line. Golden test: the TRACE script from `docs/clusters.md` (`-p batch --gpus=a40 --ntasks-per-node=64 --mem=512G -t 24:00:00 --requeue -J wobl -o logs/sweep/wobl_%j.out -e logs/sweep/wobl_%j.err`) with `workdir=/trace/group/biosimmlab/wxu2/vascular_super_resolution` renders `job.sbatch` with only `-J wobl`, `--ntasks-per-node=64`, `--open-mode=append` and the tracking block, and CLI args `-p batch --qos=normal -A biosimmlab -t 24:00:00 --mem=512G --gres=gpu:a40:1 --requeue --open-mode=append --signal=B:USR1@120 -o /trace/group/biosimmlab/wxu2/vascular_super_resolution/logs/sweep/wobl_%j.out -e …/wobl_%j.err …`; the Bridges-2 script (`--partition=GPU-shared --account=mch250030p --gres=gpu:h100-80:1 --cpus-per-task=8 --mem=80G --time=04:00:00 --qos=gpu`) renders `--cpus-per-task=8` in the file and `-p GPU-shared --qos=gpu -A mch250030p -t 04:00:00 --mem=80G --gres=gpu:h100-80:1 --no-requeue …` on the line (Bridges-2 charges and has `JobRequeue=1`).

**Output path expansion** (`render.expand_pattern(pattern, slurm_id, name, user, array_index=None)`): once `JOBID` is known the attempt's `stdout_path/stderr_path` are computed from the pattern per sbatch(1): `%j`/`%J` → job id, `%x` → name, `%u` → user, `%A` → array base id, `%a` → array index (element paths live in `array_tasks`), `%%` → `%`, `%<n>j` zero-padded; a pattern containing `%N`, `%n`, `%t` or `%s` cannot be expanded before the job starts — such paths are filled from `scontrol -o show job <id>` (`StdOut=`/`StdErr=`, expanded by the controller, fixture `scontrol_show_job_615411`) on the first tick after `RUNNING`, and `job_logs` answers `E_NO_LOG_YET` until then. Adopted jobs (`wrap=False`, `<cluster>:<slurm_id>`) take `StdOut/StdErr/WorkDir/Command` from the same `scontrol` call while the job is in controller memory (`sacct … WorkDir` afterwards; stdout unknown ⇒ `E_NO_LOG_YET` with the hint to pass the path to `remote_read`).

Submit through the helper (§7.2): `cd <workdir> && bash <bin>/submit.sh <ctrl_dir> <token> -- <args> <ctrl_dir>/job.sbatch`. stdout first line `JOBID <id>` (`id;cluster` already split) or `ERR <rc>` followed by the sbatch stderr. `ERR 1` = sbatch rejected the job (map below); `ERR 2` = ctrl dir missing (server bug, `E_HELPER`); **`ERR 3` = lock timeout — ambiguous, another `submit.sh` may be mid-`sbatch` — handled as `UNCONFIRMED`, never through this map**; any output whose first line is neither `JOBID` nor `ERR` is ambiguous too. Error map (substring of stderr → code; both the `sbatch: error: Batch job submission failed: X` form and the `sbatch: error: X` detail lines are scanned; `--test-only` prints `allocation failure: X` instead):

| stderr contains | code |
|---|---|
| `Invalid partition name specified` / `invalid partition specified` | `E_PARTITION` |
| `No partition specified or system default partition` | `E_PARTITION_REQUIRED` |
| `Invalid account or account/partition combination specified` | `E_ACCOUNT` |
| `Invalid qos specification` | `E_QOS` |
| `QOSMaxWallDurationPerJobLimit`, `Requested time limit is invalid`, `PartitionTimeLimit` | `E_QOS_MAXWALL` |
| `QOSMaxCpuPerJobLimit`, `QOSMaxGRESPerJob`, `QOSMaxNodePerJobLimit`, `GPU-shared maximum is`, `use GPU partition for multiple nodes`, `PartitionNodeLimit` | `E_QOS_SIZE` |
| `QOSMaxSubmitJobPerUserLimit`, `AssocMaxSubmitJobLimit`, `QOSMaxJobsPerUserLimit` | `E_SUBMIT_LIMIT` |
| `Job violates accounting/QOS policy` (no more specific match) | `E_QOS_POLICY` |
| `Requested node configuration is not available` | `E_NODE_CONFIG` |
| `Invalid generic resource (gres) specification` | `E_GRES` |
| `Memory required by task is not available` | `E_MEM` |
| `Job dependency problem` | `E_DEPENDENCY` |
| `Access/permission denied` | `E_PERMISSION` |
| `Disk quota exceeded`, `No space left on device` | `E_QUOTA` |
| `Socket timed out`, `Unable to contact slurm controller`, `Zero Bytes were transmitted` | `E_CTLD_BUSY` (retried by submit.sh 3× 10 s; the jobid file guards duplicates) |
| `This does not look like a batch script` | `E_SCRIPT` |

Fixture evidence: TRACE `-p no_such_partition` → `sbatch: error: invalid partition specified: no_such_partition` + `allocation failure: Invalid partition name specified` (rc 1); `-t 99-00:00:00` → `sbatch: error: QOSMaxWallDurationPerJobLimit` + `allocation failure: Job violates accounting/QOS policy …`; bad gres → `allocation failure: Requested node configuration is not available` (no `sbatch: error:` line); Bridges-2 without `--qos` → `allocation failure: Invalid qos specification`.

Estimate — **one exec per target** (≤ 4 targets, run concurrently under the channel semaphore, each with `cmd_timeout_s`; a slow or timed-out pass affects only its own target, `est_wait_src="none"`), `2>&1` so the stderr line and `::RC` stay ordered (`--test-only` writes its estimate to **stderr**, fixtures `sbatch_test_only_ok.err`):
```
export SLURM_TIME_FORMAT=%s LC_ALL=C; cd <workdir>
echo '::T1'; sbatch --test-only <target_args T1 without --parsable/--comment/--hold> <ctrl_dir>/job.sbatch 2>&1; echo "::RC $?"; echo '::END'
```
Success (rc 0): `^sbatch: Job (\d+) to start at (\S+) using (\d+) processors on nodes (\S+) in partition (\S+)$` → `est_start_ts` (epoch, or ISO → `to_epoch`), `nodes`, `partition` (the one SLURM chose from a comma list). Failure (rc 1): `^allocation failure: (.+)$` → infeasible with that reason, plus any preceding `^sbatch: error: (.+)$` detail lines mapped through the table above (the detail decides when present). `--test-only` is a single will-run pass: pessimistic (ignores backfill) and noisy; it is a relative signal (§8). No job is created.

Control (per-id decision in §4 `job_control`): pending → `scancel <ids>`; running graceful → `scancel --signal=TERM --full <ids>` then `scancel <ids>` at `cancel_hard_ts` (= request + `spec.grace_s`) or immediately (`graceful=False`); array element `scancel 123_4`, whole array `scancel 123`; `scancel --signal=<sig> --full <ids>` for `signal` (a checkpoint trigger for a payload that declared `child_signal` is `scancel --signal=USR1 --batch <id>`, which reaches `wrap.sh` and is forwarded); `scontrol hold|release|requeue <ids>`; `scontrol update JobId=<id> Dependency=<list>` for dependency repointing (§5.2 step 11); `scontrol -o show job <id>` for the stale-job check, output-path resolution, adoption and `job_status(detail="full")` — parsed with a regex over the known key set (`JobState, Reason, Restarts, Requeue, StartTime, EndTime, NodeList, BatchHost, StdOut, StdErr, WorkDir, Command, Comment, TresPerJob, TresPerNode, Dependency`) because `Command`, `WorkDir`, `Comment` and `Reason` may contain spaces; `Invalid job id specified` on stderr (rc 1) means "not in controller memory", not an error. `scancel` of a finished job exits 0 (fixture `scancel 1`); per-id acceptance is judged by the next tick.

Allocation `alloc.sbatch`: same header with `-J alloc-<handle>`, `-t <hours>`, plus CLI `--signal=B:TERM@60 --no-requeue`; body `source <ctrl_dir>/env.sh; cd <workdir>; exec <bin>/alloc-agent.sh <ctrl_dir> <idle_release_s>`.
## 7. Remote helpers (package data `slurm_mcp/helpers/`, deployed to `<control_root>/bin/<sha8>/`)

POSIX bash 4 (RHEL 8), GNU coreutils, no `jq`, no python. `sha8` = first 8 hex chars of `sha256(wrap.sh + submit.sh + alloc-agent.sh)`; `<control_root>/bin/VERSION` holds it. Both `flock` and `setsid` exist on both clusters (`tools.out`), but the submit lock uses `mkdir` because `flock` semantics over NFS (TRACE VAST) are mount-dependent.

### 7.1 `wrap.sh` — runs inside every wrapped batch job

```bash
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
  SIGNALED=USR1; classify USR1; write_status signaled "" "$CAUSE"
  if [ -n "$CSIG" ]; then
    FWD=$CSIG; echo "slurm-mcp: time-limit warning (cause=$CAUSE); forwarding SIG$CSIG to child ${CHILD:-none}"; killchild "$CSIG"
  else
    echo "slurm-mcp: time-limit warning (cause=$CAUSE); no child_signal declared, payload keeps running"
  fi
  write_status signaled "" "$CAUSE"
}
on_term() {  # scancel / time limit / preemption: forward TERM as TERM, never $CSIG
  SIGNALED=TERM; classify TERM; FWD=TERM; write_status signaled "" "$CAUSE"
  echo "slurm-mcp: SIGTERM (cause=$CAUSE); forwarding SIGTERM to child ${CHILD:-none}"
  killchild TERM
}
trap on_usr1 USR1
trap on_term TERM
if [ "$HAVE_SETSID" = 1 ]; then setsid "$@" & else "$@" & fi
CHILD=$!
wait "$CHILD"; RC=$?
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; RC=$?; done   # re-wait when a trap interrupted wait
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
```

Notes: the payload is started with `setsid` so `kill -- -pgid` reaches grandchildren. **Default behaviour (`child_signal` unset):** the USR1 warning is only recorded — a python/torch payload without a USR1 handler would otherwise die with rc 138 and be reported `FAILED` — and the payload runs until SLURM's TERM at the limit, which `wrap.sh` forwards as TERM; sacct then says `TIMEOUT`. **Opt-in (`child_signal="USR1"`, payload checkpoints on it):** the warning is forwarded; a payload that exits 0 after checkpointing is **not** requeued unless it touched `$SLURM_MCP_CTRL/requeue.requested` (avoids rerunning finished work); one that exits non-zero from the signal is requeued when `on_timeout=requeue` (which the server only accepts together with `child_signal` and `checkpoint_interval_h`, §3.2) — `scontrol requeue` requires the job to be requeueable, which the server guarantees by injecting `--requeue` whenever `on_timeout=requeue` (needed on TRACE where `JobRequeue=0`). `cause` in `status.json` is what the Monitor uses to upgrade a sacct `FAILED 138:0`/`143:0` near the limit to `TIMEOUT` (§5.3). On preemption with `GraceTime=0` the TERM handler may not get to run before SIGKILL; the Monitor never depends on it. Graceful cancel: `job_control` writes `cancel.requested` first, so the forwarded TERM is classified `cancel` and never triggers a requeue.

### 7.2 `submit.sh` — idempotent sbatch

```bash
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
```

The caller always passes `--comment=slurm-mcp:<handle>:<attempt>:<token>` and the script path `<ctrl_dir>/job.sbatch`, so the attempt is identifiable three ways: in the live queue by `%o` (Command) and `%k` (Comment — always visible in squeue/scontrol even where sacct does not store it), and in `sacct -o SubmitLine` on both clusters (TRACE `AccountingStoreFlags=(null)` stores no Comment; Bridges-2 stores it). `ERR 3` (lock held for 30 s) is **not** a failure: the holder may be inside `sbatch` on a busy controller (`MessageTimeout=200 s` on Bridges-2), so the caller treats it as `UNCONFIRMED` and the Monitor confirms from `%o`/`%k`/`jobid`/`SubmitLine`. Retrying only controller-timeout errors is safe because a job created by a timed-out `sbatch` is found by the recovery sweep and the duplicate is cancelled. The script never removes a lock it did not create; a stale lock is cleared only by the Monitor when it finalises the attempt (§5.2 step 9).

### 7.3 `alloc-agent.sh` — batch step of a reusable allocation

```bash
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
```

Command files written by the server (`cmds/007.sh`, `.tmp` + `posix_rename`):
```bash
# slurm-mcp cmd=a3.c7
cd <cwd or workdir> || exit 97
set -o pipefail
<command>
```
`ls`-free globbing in the loop keeps submission order because names are zero-padded (`001.sh`, `002.bg.sh`, …); the server (`alloc.py`) records `alloc_cmds.out_path = <ctrl>/cmds/007.out` (or `007.bg.out`) and `kill_path = <ctrl>/cmds/007.kill` (or `007.bg.kill`) — the same `<base>.kill` rule the agent uses — and the tick polls `<ctrl>/cmds/007.rc`. Because the heartbeat, `release`, new `.bg.sh` files and `.kill` files are all handled every second regardless of a running foreground command, a 40-minute `alloc_run(detach=False)` never trips `heartbeat_stale` and never blocks `job_control(['a3.c4'],'cancel')` on a detached command.
## 8. Placement / balancing algorithm (placer.py)

**Target key** grammar: `<cluster>:<partition[,partition]>[:<gres-type>][@<qos>]`; account is implied by the profile. Candidate generation for the spec's `cluster` (or every cluster whose discovery succeeded and that is not `auth_failed`/unreachable): every partition the user can use (`AllowGroups` contains the user's group or `ALL`; `AllowAccounts`; assoc partition filter), crossed with the GPU types present in that partition that match `resources.gpu_types` (**one candidate per type** when `gpus > 0` and `gpu_types is None` — an untyped `--gres=gpu:N` is never rendered, because on Bridges-2 such jobs land on h100 nodes (`squeue_all_counts.out`) and would be priced at the `gpu:*` rate while costing 2 SU/h; none when `gpus == 0`), with the QOS chosen per §6.1. Partitions listed together in `profile.partition_groups` additionally form a joint candidate (`GPU-small,GPU-shared`) when every member is individually feasible (`EnforcePartLimits=ALL` on both clusters rejects the whole submission otherwise). Explicit `placement` strings/lists restrict the set; `policy.targets_allow/deny` and `profile.target_overrides[key].enabled` filter it. With dependencies the cluster is pinned.

**Feasibility** (hard; each failure yields a `why` string; infeasible options are still returned, ≤ 3): `time_s ≤ effective max_wall`; `gpus × nodes ≤ max_tres_pj gres/gpu` and `nodes ≤ MaxNodes`/`max_tres_pj node`; `cpus/mem ≤ node size`; partition `State=UP`; my running on the target `< max_running_per_target[glob]` (etiquette, e.g. `{"trace:biosimmlab*": 1}`); QOS `max_jobs_pu`/`max_submit_pu` not exhausted (GPU-small 2 running / 10 submitted); `cost_worst_su ≤ su_balance − su_reserve` when the cluster charges and the balance is known (`why: "worst case 128 SU (4 runs × 32) would leave 304 SU < reserve 50"`-style text names both numbers); maintenance: `now + est_wait + time` must end before the next `MAINT` reservation covering the partition's nodes (else `why: "maintenance <start>; job would not finish"`); cross-cluster moves need portable inputs; circuit breaker: `target_stats.breaker_open_until_local > now` (opened for 60 min after ≥ 2 NODE_FAIL/BOOT_FAIL or 3 submit errors within an hour) or `infeasible_until_local > now` (30 min after a QOS/limit submit error); **self-preemption**: a target partition whose `PriorityTier` is higher than that of a partition sharing its nodes (TRACE `biosimmlab` tier 20 over `batch` tier 1 on `trace[01-29]`) is infeasible with `why: "would preempt my own batch job(s) (j12, j14)"` whenever I have running jobs in the lower-tier partition **and** the target has no idle node carrying the requested gres (a job that can start on a free node preempts nobody), unless `policy.allow_self_preempt` or `target_overrides[key].allow_self_preempt` is true; `biosimmlab` is additionally disabled by default in the TRACE profile example (§2.1).

**Pending caps are throttles, not placement signals.** My pending jobs on a cluster must stay `< bf_max_job_user − 2` (TRACE 10 ⇒ 8) or per partition `< bf_max_job_user_part − 2` (Bridges-2 20 ⇒ 18), and per target `< policy.max_pending_per_target` when set (both discovered from `SchedulerParameters`, both counted from this tick's `::SQUEUE` **plus** our own `SUBMITTING`/`UNCONFIRMED` attempts). A job whose *best* feasible target has no free slot is **not** re-placed on the next target because of the cap (that is what sent the third sweep job to `biosimmlab`/Bridges-2): it is held locally as `QUEUED{target, why:"cap 8/8 pending on trace"}` with its target fixed, and the Monitor submits it when a slot frees (§5.2 step 10; HTCondor idle-throttle behaviour). The rebalancer may still move a `QUEUED` job like a pending one (it has no SLURM id, so the move is just a target change). `plan_job` shows such options as feasible with `why: "queued locally until a slot frees (8/8)"`.

**Estimates** per feasible target: `--test-only` for the top 4 by pre-score (`hist_p50` or depth) → `est_wait_h = max(0, est_start_ts − now)/3600`, `src="test_only"`; `hist_p50_h` from `wait_history` (same target key, last 30 d, ≥ 3 samples; else partition-level); from the snapshot (`::PD` classified per §6.2): `depth_typed` = pending jobs on the same partition requesting this gres type, `depth_untyped` = pending rows of that partition with `N/A` in both TRES views (untyped GPU demand: every `--gpus` job on TRACE, hundreds of GPU-shared rows on Bridges-2), `depth = depth_typed + depth_untyped`; `ahead` = the subset with priority ≥ mine (from `::MINE`/`::PD`; untyped rows count as ahead). `est_wait_h = test_only if available else hist_p50 if available else min(policy.unknown_wait_h, 0.25 × ahead)`; when both exist and `hist_p50 < test_only/3` report `hist_p50` (`src="history"`) and keep the test-only number in `why`. The shortcut `est_wait_h = 0` for "idle nodes carrying the requested gres" applies **only when `depth == 0` including untyped rows** — on TRACE `batch` (242 pending `N/A` rows in `squeue_all_counts.out`) it therefore never fires and the 6 h `--test-only` estimate stands; a `--test-only` result is never overridden downwards by depth. `queue_ahead`/`queue_ahead_untyped` are both reported. Placer test (§10): with `trace/squeue_all_counts.out` and one idle `batch` node, `trace:batch:a40` must report `queue_ahead_untyped=242`, `est_wait_src="test_only"`, never `est_wait_h=0`.

**Cost**: `cost_su = rate × units × hours` for one run; `rate` from `TRESBillingWeights` (per gres type when present) else `profile.su_rates` (`gpu:<type>` → `gpu:*` → `cpu`; candidates are always typed, so the `gpu:*` fallback is used only for a type absent from `su_rates`); units = GPUs (shared partitions) or `gpus_per_node × nodes` (whole-node GPU partitions, i.e. `OverSubscribe=EXCLUSIVE`), cores for CPU partitions; free clusters ⇒ 0. **Worst case**: `cost_worst_su = cost_su × (1 + spec.max_restarts)` when the attempt will be requeueable (`--requeue` rendered: `on_timeout=requeue`, `PreemptMode` contains `REQUEUE` with `requeue` True/None, or `spec.requeue=True`; or nothing rendered on a `JobRequeue=1` cluster — which §5.3 avoids on charging clusters by rendering `--no-requeue`), else `cost_worst_su = cost_su`. Feasibility and `max_extra_su` use `cost_worst_su`; the score uses `cost_su + risk-weighted rework`; both are returned. Example: Bridges-2 `GPU-shared:h100-80:2`, 8 h, `on_timeout=requeue`, `max_restarts=3` → `cost_su = 2 × 2 × 8 = 32`, `cost_worst_su = 128` — infeasible against 482 − 50 only if the balance is below 178, but reported so the agent sees what a non-checkpointing payload could burn (and `on_timeout=requeue` without `child_signal`+`checkpoint_interval_h` is refused at validation, §3.2).

**Risk**: for a partition whose `PreemptMode` contains `REQUEUE` or `CANCEL` and that shares nodes with a higher-`PriorityTier` partition (TRACE `batch` tier 1 vs `biosimmlab` tier 20 on `trace[01-29]`): `preempt_pct = min(80, 10 + 5 × pending jobs in the higher-tier partitions)` (typed and untyped rows alike; +2 for a NODE_FAIL in the last day). Expected rework `risk_h = preempt_pct/100 × lost_h` where `lost_h = min(hours, spec.checkpoint_interval_h)` when the user declares periodic checkpoints, else `hours/2`. **No halving for signal-driven checkpoints**: on both clusters `GraceTime=0` and `PreemptParameters` is unset, so preemption is an immediate kill; the placer only credits a checkpoint hook when the discovered partition/QOS `GraceTime ≥ grace_s` or `PreemptParameters` contains `send_user_signal`.

**Score** (hours, lower is better):
```
score_h = est_wait_h
        + hours + risk_h                                  # wall clock including expected rework
        + cost_su × su_to_hours                           # 0.25 balanced | 0.02 fastest | 2.0 cheapest (policy override)
        + etiquette_h × [my running on the target ≥ soft_caps[glob]]
        + 0.5 × [target cluster ≠ cluster holding the inputs]      # staging penalty
        + profile.target_overrides[key].penalty_h (default 0)
```
`cheapest` additionally prefers free clusters unless `est_wait_h > 24`. Ties (< 0.25 h) prefer: non-preemptible, then the cluster with the data, then `policy.prefer_cluster`, then shorter queue depth. Every option carries `why`, e.g. `"test-only start 05:40 (5.6 h); 4 SU; preempt risk 35 %"`.

Bridges-2 example (survey numbers): a 4 h `v100-32:1` job → `bridges2:GPU-small,GPU-shared:v100-32@gpu` (test-only ≈ 20 min, 4 SU) ≈ 0.3 + 4 + 1 = 5.3 h; `bridges2:GPU-shared:v100-32@gpu` (2.6 d) ≈ 66 h; `trace:batch:a40` (6 h, 0 SU, risk 35 % × 2 h = 0.7 h) ≈ 10.7 h; `trace:biosimmlab:a40` (immediate; etiquette +2 h if a lab job is already running) ≈ 4–6 h. With `placement="auto"` the server picks the first feasible row; the agent can see the whole table via `plan_job`.

Knobs: `configure(placement=…)` fields of `PlacementPolicy` (§3.2); `configure(notify=…)` fields of `NotifyPolicy`; per-target overrides in the profile.

## 9. Failure handling matrix

### 9.1 Error catalogue (errors.py)

`E_AUTH, E_HOSTKEY, E_UNREACHABLE, E_SSH, E_CTLD_BUSY, E_SUBMIT_AMBIGUOUS, E_SUBMIT_FAILED, E_PARTITION, E_PARTITION_REQUIRED, E_ACCOUNT, E_QOS, E_QOS_MAXWALL, E_QOS_SIZE, E_QOS_POLICY, E_SUBMIT_LIMIT, E_NODE_CONFIG, E_GRES, E_MEM, E_DEPENDENCY, E_DEP_CROSS_CLUSTER, E_PERMISSION, E_SCRIPT, E_QUOTA, E_NO_TARGET, E_UNKNOWN_ID, E_NO_LOG_YET, E_ALLOC_NOT_READY, E_ALLOC_ENDED, E_CMD_TOO_LONG, E_TOO_MANY_FILES, E_TOO_MANY_BYTES, E_UPLOAD, E_CONFIRM_REQUIRED, E_PLAN_EXPIRED, E_INVALID_SPEC, E_HELPER, E_STATE` — each with a fixed `fix` template; the message is `f"{code}: {message} — fix: {fix}"`.

### 9.2 Matrix

| Failure | Detection | Handling | What the agent sees |
|---|---|---|---|
| No password stored / auth denied | `RuntimeError` from credentials / `PermissionDenied` | never retried; transport `auth_failed`; `clusters()` warning | `E_AUTH: bridges2 rejected the password — fix: run \`slurm-mcp auth set bridges2\` in a terminal, then clusters(refresh=True)` |
| Host key changed for a known IP | `validate_host_public_key` returns False | refuse | `E_HOSTKEY: … old SHA256:… new SHA256:… — fix: confirm with the site, then \`slurm-mcp hostkeys forget <cluster>\`` |
| New pool member (round-robin login node) | unknown key from an unseen IP | accept + append + log | none (`clusters()` warning `new host key accepted from <ip>`) |
| Cluster unreachable (TRACE without VPN, outage) | TCP probe fails / connect errors | Monitor backoff 30 s → 5 min; other clusters continue; `cluster_unreachable` once; placement excludes the cluster | `E_UNREACHABLE: trace.cmu.edu:22 not reachable — fix: <requires_vpn_hint>` |
| SSH drops mid command | `ConnectionLost`, `ChannelOpenError`, `exit_status None` | reconnect + one retry for `idempotent=True`; submit → `UNCONFIRMED` + recovery (§5.2.9) | transparent, or `state="SUBMITTING"` result / later `submitted` event |
| slurmctld busy (`Socket timed out`) | stderr match | `submit.sh` retries 3× guarded by the jobid file; tick skipped without state change | `E_CTLD_BUSY` only when all retries fail |
| Partial probe output | missing `::END` / non-zero `::RC` | probe discarded, tick counted as failed | none |
| sacct lag / job in neither | §5.2 steps 2–3 | `COMPLETING` ≤ 2 ticks; helper status fallback; `scontrol show job`; `LOST` after 3 stale ticks and > 120 s | event `lost` with a `run_command('sacct -j …')` hint; `job_status` later re-adopts if sacct catches up |
| Preempted / requeued | §5.3 | count restarts; hold + `needs_attention` after `max_restarts` or a requeue loop | events `preempted`, `requeued`, `needs_attention` |
| Timeout / OOM / NODE_FAIL | §5.3 | classify, enrich, `--exclude` the failed node for 2 h | terminal event with the hint |
| Quota full | `df`/`quota_command` headroom; `Disk quota exceeded` on submit/SFTP | refuse upload; `quota_warning` at 90 % | `E_QUOTA` with numbers and the 5 largest entries |
| SU balance below reserve | snapshot | targets infeasible; rebalance never adds cost above `max_extra_su` | option `why: "would leave 12 SU < reserve 50"` |
| Maintenance reservation | discovery/snapshot `::RESV` | colliding targets infeasible; rebalancer treats reason `Reserved for maintenance` as movable | `why: "maintenance 2026-09-10 08:00; job would not finish"` |
| Helper missing / outdated | `::HELPER` ≠ packaged sha8 | redeploy on the next W tool; live jobs keep their own copy | `E_HELPER` only if deployment fails (then `wrap=False` submissions still work) |
| Rebalance race (old job started between plan and cancel) | `squeue --me -h -o '%A\|%T'` re-check after the new submit (client-side filter; `-j` exits 1 for purged ids) | cancel the new job instead; re-check rc ≠ 0 / timeout ⇒ keep both, retry next tick | `skipped: "started meanwhile"` / `moving` |
| Submit ambiguous (`ERR 3` lock timeout, dropped channel, `CommandTimeout`, no `JOBID`/`ERR` line) | §5.1 step 7 | attempt `UNCONFIRMED`; confirmed from `%o`/`%k` in the next tick's squeue, the `jobid` file, or `SubmitLine`; failed only after 15 min of *healthy* observation with no matching untracked row | `state="SUBMITTING"`, then `submitted` or `needs_attention{submit_unconfirmed}` |
| Submit task died / server restarted mid-submit | attempt still `INTENT` (never invoked) | swept to `FAILED` after 10 min | `needs_attention{submit_stuck}` |
| Windows CRLF / BOM in scripts | `textio.normalize_text` at render/`remote_write` | normalised, never sent with `\r` | `warnings: ["crlf_normalized"]` |
| Remote name invalid on NTFS / > 260 chars | `textio.local_safe_name`, `long_path` | saved under a safe name, `\\?\` prefix | `renamed: [{remote, local}]` |
| Dependency on a job past `MinJobAge` / failed / rebalanced | ledger resolution (§5.1 step 2), tick step 11 | evaluated semantically; `--dependency` only on live ids; `scontrol update … Dependency=` on id change; hold before `kill_invalid_depend` | `dependencies_resolved`, `dependency_updated`, `needs_attention{dependency_unsatisfiable}` |
| Payload has no handler for the time-limit warning | `child_signal=None` default | warning recorded, not forwarded; TERM forwarded as TERM; rc `128+sig` near the limit ⇒ `TIMEOUT` | `timeout{source:"helper+sacct"}` |
| Restarts multiply the bill | `cost_worst_su` (§8) | feasibility on the worst case; `--no-requeue` on charging `JobRequeue=1` clusters; `on_timeout=requeue` needs checkpointing | `cost_worst_su`, `needs_attention{restart_cost}` |
| Own pending cap reached | `bf_max_job_user[_part]`, `max_pending_per_target` | job held locally (`QUEUED`), submitted when a slot frees; never escalated to a costlier/preempting target | `queued{target, why}` |
| Duplicate job with our token (retry after controller timeout) | `::RECOVER` finds two rows | cancel the newer | `needs_attention{why:"duplicate_cancelled"}` |
| Tool call exceeds Claude Code timeout / subagent call (never backgrounded) | server caps: waits 600 s, submits/transfers/moves return a handle after `wait_s`, `run_command ≤ 600 s`; a client abort never cancels a server-side task | `.mcp.json` `timeout: 900000` documented | non-terminal `state` + `next` naming the event to wait for |
| Slow slurmctld (`MessageTimeout=200 s`, `max_rpc_cnt`) | discovery | per-command timeout `MessageTimeout + 60`; one exec per `--test-only` target | a slow target gets `est_wait_src="none"`, the plan still returns |
| Server killed mid transfer | `transfer_files` rows | re-run/restart resumes; `.part` files resumed by offset | `transfer_done` absent → call again or `job_status(['t4'])` |
| Laptop sleeps | local monotonic / `ClusterClock` jump > 120 s between ticks | lease re-acquired (or found lost) before any write; full reconcile sweep, `observed_late` events | events with `observed_late: true` or `needs_attention{lease_lost}` |
| Two processes (second Claude session, CLI) | fenced `lease` row | second instance runs no Monitor but serves tools from the ledger; a live holder is never taken over automatically; a stale holder loses on its next fenced write | `clusters().monitor = "held by pid N"` / `"lost to pid N"` |
| Backgrounded `wait_for_events` result lost (compaction, session closed) | per-client deliver-then-ack cursor | nothing was acknowledged; the next call replays the same events | same `events` again; `unread_events` unchanged |
| Unknown handle | store lookup | | `E_UNKNOWN_ID: no job 'j99' — fix: list_jobs(); raw jobs as '<cluster>:<slurm_id>'` |
| Oversized output | limits | truncation with `truncated: true` and `next_offset` | actionable paging hint |
| SQLite locked / corrupt | sqlite3 errors at start | `busy_timeout=5000`; on corruption rename to `state.db.corrupt-<ts>`, start fresh, `needs_attention`; `slurm-mcp db recover` rebuilds jobs from `<control_root>/jobs/*/a*/spec.json` + `sacct` | one `needs_attention` event |
## 10. Testing strategy

**Fake cluster** (`testing/fake_slurm.py`): `FakeCluster(config)` models partitions (limits, gres types, `PreemptMode`, `PriorityTier`, `GraceTime`, `MinJobAge`, `MessageTimeout`, dbd lag, controller busy), nodes, associations/QOS, an in-memory filesystem, and a controllable clock; its state is also serialisable to JSON so the stub binaries below can read it. Scenario methods: `advance(seconds)`, `start(job_id)`, `finish(job_id, rc=0, state="COMPLETED")`, `preempt(job_id)` (creates a `PREEMPTED` `-D` row and a new incarnation when the job is requeueable), `oom(job_id)`, `node_fail(job_id)`, `timeout(job_id, killed_by="slurm"|"forwarded")`, `set_sacct_lag(s)`, `set_ctld_busy(s)` (sbatch/squeue answer `Socket timed out` for that long), `set_estimate(target, seconds)`, `drop_connection()`, `deny_auth()`, `inject_sbatch_error(text)`, `write_file(path, text)`, `add_untracked(partition, tres_per_node="N/A", tres_per_job="gres:gpu:a40")`. It renders **exactly** the text of §6 (format strings copied from `tests/fixtures`, including `allocation failure:` lines on stderr, `CANCELLED by <uid>`, `None assigned`, `Unknown`, `(null)`, `%b`/`tres-per-node` **`N/A` rows for `--gpus` jobs**, `uniq -c` shapes, `df` mount-point rows). `FakeTransport` implements the `SSHTransport` surface (`run`, `run_with_stdin_file`, `run_to_file`, `sftp`, `tcp_probe`, `banner_probe`) by splitting composite commands on the `echo '::SECTION'` sentinels and dispatching each segment by its first word, and an in-memory SFTP (`put/get/glob/stat/makedirs/posix_rename/open/listdir`). It is kept for fast pure-Python monitor/property tests only; it is **not** the arbiter of shell semantics.

**Real Linux bash is mandatory** (`tests/bash/`, run under WSL locally and on a Linux CI job; Git Bash runs only a smoke subset because it lacks `setsid`, cannot `kill -- -pgid`, and its `hostname` rejects `-s`): stub executables `squeue sacct sinfo scontrol sbatch scancel sacctmgr sshare` (`testing/stubs/`, POSIX sh + python3) are put on `PATH`; they load the `FakeCluster` JSON state, honour the exact flags the design uses (`-h -r -t all -o …`, `-O …:0|`, `-n -P -X -D -j …`, `--test-only` writing to stderr, `--parsable`, `-j` on a purged id exiting 1 with `Invalid job id specified`) and print the fixture formats. The **real composite strings** from `slurm/commands.py` (tick, discovery, snapshot, estimate, `submit.sh` invocation) are executed through `bash -lc <shlex-quoted composite>` exactly as the transport would — including a fake MOTD in the login profile, ctrl dirs containing spaces and `$`, `2>&1` interleaving in the estimate, and `tr '\n\r|'` — and a golden test asserts that `parse.py` yields the same objects from the stub-binary output as from `FakeTransport` (drift between the two is a failure). Helper tests under the same harness: `wrap.sh` — USR1 recorded and **not** forwarded by default (a python payload survives), forwarded to the whole process group when `SLURM_MCP_CHILD_SIGNAL=USR1`, TERM forwarded as TERM, `cause=timeout` and rc 138/143 classification near the limit, `cause=cancel` with `cancel.requested`, requeue-on-timeout rules (rc≠0, `requeue.requested`, max restarts); `submit.sh` — `jobid` idempotency, mkdir lock, `ERR 3` on a held lock, controller-busy retry; `alloc-agent.sh` — fg/bg completion, **kill a bg command while a fg command runs** (`003.bg.kill` honoured within 2 s), heartbeat refreshed every second during a 30 s fg command, release, idle release, restart of the agent with already-started commands.

**Host-key pool** (`tests/ssh/`): an in-process `asyncssh` server with two host keys and two loopback addresses verifies `known_hosts`/`validate_host_public_key`: a new key from an unseen address of the same alias is accepted and appended; the same address presenting a different key of the same type is refused with `HostKeyChanged`; `hostkeys forget` clears both stores.

**Windows client** (`tests/unit/test_textio.py`, plus a `windows-latest` CI job for the transfer tests): CRLF/BOM scripts render byte-identical to their LF form and produce `crlf_normalized`; NUL is refused; `rel_path` is POSIX for backslash inputs; scope keys collapse `D:\x`/`D:/x`/`d:\X`; downloads of `a:b`, `nul`, `con.txt`, trailing-dot names and 300-char paths succeed with the documented renames; `remote_write` never sends `\r`.

Layers: (1) parser goldens from `tests/fixtures/{trace,bridges2}` (every file in `index.json` has a test; `slurm-mcp record <cluster>` regenerates them via `scripts/collect_fixtures.py`); (2) `slurm/commands.py` golden strings equal to the blocks in §6; (3) render goldens: spec → `job.sbatch` + CLI args per target, **the two real user scripts from `docs/clusters.md`** (stripped directives, `--gpus=a40` → typed `--gres=gpu:a40:1`, absolute `-o/-e`, `--mem` dropped on `RM-shared`, `--no-requeue` on Bridges-2), `-o/-e` pattern expansion, QOS selection; (4) placer tests with fixed snapshots (feasibility reasons, scoring order, **`trace/squeue_all_counts.out` with an idle node ⇒ no zero-wait shortcut**, untyped demand on Bridges-2, hold-locally caps, self-preemption rule, worst-case cost, breaker, maintenance windows, GraceTime rule); (5) monitor scenarios on `FakeCluster`: normal completion; terminal in squeue before sacct; sacct lag then row; MinJobAge expiry; preempt+requeue twice then cap; timeout by SLURM and timeout by forwarded signal (`FAILED 138:0` + `cause=timeout` ⇒ `TIMEOUT`); `on_timeout=requeue` with checkpointing; OOM on a step; NODE_FAIL with same-cluster exclude; pending cancel (immediate) and running graceful → hard cancel at `grace_s`; unconfirmed submit confirmed by `%o`/`%k` before sacct, by jobid file, by SubmitLine, `ERR 3` treated as unconfirmed, failure only after 15 healthy minutes with dbd lag not counting; duplicate cancelled; rebalance race with rc≠0 re-check (both kept, retried); **cross-cluster move then `job_logs`/`collect_results`/`cancel` on the new cluster**; dependency on a `COMPLETED`/`FAILED`/purged job, dependency repointed after a move; `QUEUED` job submitted when a slot frees; `INTENT` sweep; allocation ready/expiring/cmd_done; restart recovery (a new Monitor on the same store emits late events exactly once); **two Monitors** (a second process takes a stale lease after the holder "sleeps"; the woken holder's next write raises `LeaseLost` and emits nothing); property test (hypothesis): applying the same probe twice emits no events, and any interleaving of {squeue present/absent, sacct present/absent/terminal, status.json present} with lag ≤ 2 ticks yields exactly one terminal event; (6) transfer tests on temp dirs (manifest diff, ignore rules, tar vs sftp choice, quota refusal on the destination path, `.part` resume, member-wise `filter="data"` extraction with renames); (7) contract tests via in-memory `mcp.Client(server)`: every tool's `structuredContent` validates against its `outputSchema`, `summary` non-empty, annotations present, descriptions < 2 KB, error text starts with `E_`, `wait_for_events` wakes on an appended event, emits progress, **replays unacknowledged events and acknowledges only on `ack_seq`**, filtered waits leave `unread_events` accurate, `submit_job` returns a handle before the fake upload finishes, `job_control` >10 ids requires `confirm`; (8) `slurm-mcp serve --fake` for manual Claude Code sessions; (9) opt-in live smoke (`SLURM_MCP_LIVE=trace,bridges2`): bootstrap (incl. `::CAP_O` with `tres-per-job` and `::DF` on the group volume), `--test-only`, a 1-minute `hostname` job on the cheapest target (`cpuonly-debug` / `RM-shared@low`), a 3-minute job with `-t 2` and no `child_signal` that must end `TIMEOUT` (not `FAILED`), wait → logs → collect, a 5-minute allocation with two commands, tar upload to a temp dir.

## 11. Decisions on the open questions

**(a) Helper-in-job vs pure sacct.** Both, layered, with `sacct`/`squeue` authoritative for state. `wrap.sh` is on by default for spec jobs because it alone provides the exit code independent of slurmdbd lag, a heartbeat, user progress JSON, the `cause` that lets a forwarded-signal death be classified as `TIMEOUT`, TERM forwarding to the whole process group, and (opt-in) checkpoint signalling and requeue-on-timeout. It **never kills the payload on its own**: the time-limit warning is forwarded only when the spec declares `child_signal`. It costs one `cat` per running job inside the existing tick. It is **not** relied on for preemption classification (GraceTime=0 makes that impossible) nor for restart counts (`RestartCnt`/`-D` rows). `wrap=False` and adopted raw jobs degrade to sacct-only tracking. Helpers are deployed to a content-addressed directory so running jobs never see a changed script.

**(b) Rebalancing mechanism.** Cancel+resubmit with lineage, in the order submit-new → re-check-old → cancel-old (§5.4); it works across clusters, keeps a single charge, and age-priority loss is negligible on both sites. `--partition=a,b` is used only for configured `partition_groups` with identical charging and limits (Bridges-2 `GPU-small,GPU-shared`, the PSC-documented recipe); mixing charge models (`GPU,GPU-shared`) can land on a whole node and cost 8×. Racing duplicates are out of v1: double staging, SUs and fairshare burned on both sites, and a real race window at 30 s polling.

**(c) SQLite vs JSON.** SQLite (WAL): two processes (server + CLI) read it, the event log needs a monotonic sequence with cursor reads, transfers need per-file crash-safe rows, and restart recovery is a handful of queries. Migrations are a `PRAGMA user_version` ladder.

**(d) Auto-injection.** Inject only what tracking correctness or a discovered site rule requires, echo every injected option in `SubmitResult.injected`, never change resources: `-J` (spec), `-o/-e` (always, absolute — the user's own pattern when given, resolved against `workdir`, directory pre-created), `--open-mode=append`, `--signal=B:USR1@<grace>` for every wrapped job (warning only), `--requeue` when the partition preempts by requeue or `on_timeout=requeue`, `--no-requeue` on charging `JobRequeue=1` clusters unless the spec asks for requeue, `-A` from spec/profile/discovered default, `--qos` from the selection rule, typed `--gres` from `resources`, `--mail-*` only when `notify.email` is set, `--exclude` after a same-cluster NODE_FAIL, `--kill-on-invalid-dep=yes` with dependencies, `--comment=slurm-mcp:…` and `--parsable` for recovery. Never `--export=NONE`, `--time-min`, `--deadline`, or `--mem` guesses; on `no_mem_flag` partitions `--mem` is dropped with a warning. **User scripts do not keep the `#SBATCH` lines the server manages**: they are parsed into the spec and stripped, because command-line options only override the *same* option — a script's `--gpus=a40` next to an injected `--gres=gpu:a40:1`, or its `--mem=512G` on `RM-shared`, would be a conflicting request, and its `-p/-t/--qos` would be wrong after a move. Unmanaged directives are kept verbatim; everything stripped is reported in `stripped_directives`.

**(e) Allocation reuse.** Not `srun --jobid --overlap` as the primary path (a step launched over an SSH channel dies with the channel, `--overlap` GRES semantics vary, PTY prompts are fragile, no rc/out persistence). The sleeper batch job + file-queue agent gives per-command rc/out, detached commands, kill, release, survives reconnects and server restarts, and is observed with `cat`. `srun --jobid --overlap` remains available through `run_command`.

**(f) tar vs SFTP.** Tar stream when ≥ 16 changed files or many small files (< 1 MB) totalling < 256 MB; SFTP `block_size=65536`/64 requests otherwise (OpenSSH 8.0 advertises no limits); files ≥ 64 MB resumable by offset. Tar is extracted on the transfer host when it permits exec (probed), else uploaded to it by SFTP and extracted on the login host (no transfer is initiated from a login node — PSC policy). Hard caps 2 GB / 2000 files per call with explicit refusal.

**(g) Event delivery to Claude Code.** Durable event log + `wait_for_events` long-poll (≤ 600 s, progress every 30 s ⇒ auto-backgrounded after 2 min and delivered as a task notification) + `unread_events` on every response + a **per-client deliver-then-ack cursor**: returning an event never consumes it; the client acknowledges the previous delivery by passing its `next_seq` as `ack_seq`, so a task-notification result that was compacted or never read is replayed, filtered waits cannot swallow unmatched events, and a second Claude session (its own server process, its own `session_id`) has its own cursor. `notifications/message` is not used for anything the agent must see (Claude Code drops it); Channels can be added as a fourth sink later. Humans get toasts, webhooks and SLURM mail.

**(h) Long tool calls.** Every operation that can exceed a few seconds on a busy login node — submit (upload + render + ≤ 4 `--test-only` + `sbatch`), transfer, allocation command, rebalance move — is a server-side task: the intent row is committed first, the tool returns a handle, and the durable event closes the loop. This also makes the tools usable from subagents (never auto-backgrounded) and safe against the per-server wall-clock timeout cancelling the handler mid-`sbatch`. Per-command SSH timeouts are derived from the cluster's `MessageTimeout` rather than a fixed 120 s.

**(i) Concurrency across processes.** One fenced SQLite lease (pid + token) decides who runs the Monitor; SQLite's `BEGIN IMMEDIATE` writer lock serialises handle/seq allocation; a lease held by a live pid is never taken over automatically. This replaced the "older than 5 min ⇒ take over" rule, under which a laptop waking from sleep produced two reconciling Monitors.

**Also decided:** `SLURM_TIME_FORMAT=%s` everywhere with a bootstrap probe and a tz-offset fallback; `sacct -S` only in relative/ISO forms; `squeue -O RestartCnt:0|` gated by a capability probe; per-call `_meta` is impossible, so bulk cancels use `confirm=True`; TCP-22 probe (never ICMP) before connecting to VPN-gated clusters; host keys stored per alias with multiple keys per pool; DTN port chosen by banner probe; Bridges-2 `$HOME` (25 GB) is avoided by defaulting `control_root` under `remote_root`.

## 12. Implementation plan (ordered slices)

1. **Core skeleton (≈ 1,000 LOC)** — `errors.py`, `clock.py`, `textio.py`, `models.py`, `store.py` (schema incl. `jobs_current`, `lease`, `event_acks`; fenced writes; DAO), `events.py` (per-client ack cursors), `service.py` skeleton, `server.py` with `clusters`, `cluster_status` (snapshot only), `run_command`, `remote_ls/read/write`, `configure`; transport extensions (§2.2, per-cluster timeouts). Contract tests via in-memory client; host-key pool test.
2. **SLURM layer + fake harness (≈ 1,300 LOC)** — `slurm/commands.py`, `slurm/parse.py` (incl. `classify_demand`), `slurm/states.py`, `slurm/discovery.py` (caps, QOS selection, helper deploy, back-fill), `testing/fake_slurm.py`, `testing/fake_transport.py`, `testing/stubs/*`, the Linux-bash composite harness, parser goldens from `tests/fixtures`.
3. **Submit / track / events (≈ 1,600 LOC)** — `render.py` (directive stripping, pattern expansion), helpers `wrap.sh`/`submit.sh`, `submitter.py` (`SubmitTask`, `QUEUED`), `submit_job` (explicit target), `list_jobs`, `job_status`, `job_logs`, `job_control` (per-state cancel), `monitor.py` tick + reconciliation + recovery (`%o`/`%k` confirmation, dependencies, `INTENT` sweep) + lease fencing, `wait_for_events`, restart recovery. Scenario tests for §5.2/5.3 incl. the two-Monitor and forwarded-timeout cases.
4. **Failure semantics + notifications (≈ 500 LOC)** — requeue/timeout/OOM/NODE_FAIL classification, enrichment, guards, `notify.py` (toast, webhook, mail options, quiet hours, coalescing).
5. **Transfers (≈ 900 LOC)** — `transfer.py`: manifests, ignore rules, tar/SFTP, quota, resumable rows, `upload`, `download`, `collect_results`, inputs on submit, `t<N>` handles in `job_status`/`job_control`.
6. **Placement (≈ 900 LOC)** — `placer.py`: candidates, feasibility, test-only batching, history, scoring, `plan_job`, auto placement in `submit_job`, `rebalance` + Monitor scheduling, breaker, maintenance windows.
7. **Allocations (≈ 500 LOC)** — `alloc-agent.sh`, `alloc.py`, `allocate`, `alloc_run`, cmd events, `job_logs` for `a3.c2`.
8. **CLI parity, prompts, resources, docs (≈ 400 LOC)** — generic `slurm-mcp <tool>` mirror from the `Service` signatures, `slurm-mcp wait|logs -f|record|hostkeys forget|db recover`, `serve --fake`, README (`claude mcp add`, allow-list, `.mcp.json` timeout, VPN hint).

Total ≈ 7,100 LOC plus ≈ 3,500 LOC of tests. Slices 1–3 deliver requirements 2, 3 and 5 on an explicit target; 5 adds 6; 6 adds 4; 7 completes 1. Slices 2 and 5–7 can be built in parallel against the contracts in §3, §4, §6 and §7.

## 13. Changelog (review round 2 — contract changes against the previous revision)

Every item below changes a name, field, column, command string or documented behaviour that implementers may already have coded against. Unchanged: all 20 tool names, module paths (additions only), handle grammar, event kind names (additions only), the error catalogue (additions only).

1. **`JobSpec.child_signal: str | None = None`** (was `"USR1"`). The `--signal=B:USR1@grace` warning is no longer forwarded to the payload unless the spec declares the signal; `wrap.sh` forwards TERM as TERM (was: TERM → `child_signal`). `status.json` is `v:2` with `forwarded`, `limit_s`, `grace_s`. `on_timeout="requeue"` now requires `child_signal` and `checkpoint_interval_h` (`E_INVALID_SPEC`). (§3.2, §5.3, §7.1)
2. **Timeout classification**: sacct `FAILED`/`CANCELLED` with exit `128+signum(child_signal)` or signal `signum(child_signal)`, `status.json.cause=timeout` and `ElapsedRaw ≥ TimelimitRaw×60 − grace_s − 60` ⇒ `TIMEOUT` (`source="helper+sacct"`); events carry `source`. (§5.3, §3.4)
3. **User scripts are stripped, not kept**: `render.parse_sbatch()` converts every server-managed `#SBATCH` directive into spec fields and removes it; `--gpus` maps to typed `resources.gpus/gpu_types`; new result fields `stripped_directives`, `crlf_normalized` warning. `-t`, `--mem`, `-o/-e` moved from `job.sbatch` to the command line; `-o/-e` are always absolute with pre-created directories; new attempt columns `stdout_pattern/stderr_pattern`, expanded `stdout_path/stderr_path`. (§5.1 step 1, §6.3, §11d)
4. **Snapshot demand queries** use `squeue -O 'Partition:0|,tres-per-node:0|,tres-per-job:0|'` (gated by `caps.squeue_O_zero`, whose probe now includes `tres-per-job`), fallback `%b`; `N/A` rows are untyped GPU demand for every gres type of a GPU partition; the `est_wait_h = 0` shortcut requires `depth == 0` including untyped rows; `plan_job` options gain `queue_ahead_untyped`. `::MINE` moved to `-O`. (§6.2, §8)
5. **`wait_for_events`**: new args `ack_seq`, `client_id`; `ack: bool` removed; result gains `delivered_seqs`, `acked`, `unread_unmatched`; cursor is per client (`kv.cursor.<client_id>`, `event_acks`, `deliveries` tables), deliver-then-ack; `kv.cursor.agent` removed. `clusters()` returns `session_id`. (§4, §5.6, §3.3)
6. **Lease**: `kv.lease.monitor` replaced by the `lease` table with a fencing token; `store.write_fenced`; a lease held by a live pid is never taken over automatically (`slurm-mcp monitor takeover --force` for humans); Monitor stops on `LeaseLost`; `clusters().monitor` gains `"lost to pid N"`; new `needs_attention{why:"lease_lost"}`. Handle allocation via `kv.counter.handle` inside `BEGIN IMMEDIATE`. (§3.3, §5.2, §5.8)
7. **Ambiguous submits**: `ERR 3` and any non-`JOBID`/`ERR` first line ⇒ `UNCONFIRMED`; attempts move to `UNCONFIRMED` *before* `submit.sh` runs (new column `attempts.invoked_local`); tick squeue format gains `%k|%o|%Z` (confirmation from Command/Comment before sacct); the 15 min deadline counts healthy ticks only; rebalance re-check uses `squeue --me -h -o '%A|%T'` and treats rc≠0 as unknown (keep both, retry). New `needs_attention{why:"submit_stuck"}` for stale `INTENT` rows. (§5.1 step 7, §5.2 steps 8–10, §5.4, §6.2, §7.2)
8. **Per-attempt cluster fields**: `jobs` lost `cluster, slurm_id, ctrl_root, workdir, stdout_path, stderr_path, node`; they live on `attempts` (plus `ctrl_root`, `stdout_pattern`, `stderr_pattern`) and are read through the new `jobs_current` view; moves derive paths for the target cluster and re-stage inputs; `excluded_nodes` is same-cluster only; `rebalanced` payload gains `new_workdir`. (§3.3, §5.4)
9. **Dependencies** are resolved semantically in the ledger (terminal deps evaluated, `--dependency` only on live ids); raw SLURM ids in `-d` are refused; `jobs.depends_on_json` stores `[{handle, type, resolved_slurm_id}]`; the tick repoints dependents with `scontrol update … Dependency=` and holds them before `kill_invalid_depend`; new event `dependency_updated`, new `needs_attention{why:"dependency_unsatisfiable"}`; `job_status` gains `dependencies`/`dependents`. (§5.1 step 2, §5.2 step 11)
10. **Windows client**: new module `textio.py`; render/`remote_write` normalise CRLF/BOM and refuse NUL; local text read as `utf-8-sig`; POSIX `rel_path` everywhere; normalised scope keys; downloads rename NTFS-invalid names (`renamed` in `TransferResult`/`transfer_done`, `transfer_files.local_name`) and use `\\?\` paths; member-wise tar extraction. (§4 download, §5.5)
11. **Submit as a server-side task**: `submit_job`/`allocate`/`rebalance` gain `wait_s` (default 90); `submit_job` returns the handle once the intent is committed and may return `state=UPLOADING|SUBMITTING|QUEUED`; new module `submitter.py`; new event `queued`; `submitted` payload carries the full result fields. `SSHTransport.run(timeout=None)` resolves to `caps.cmd_timeout_s = max(120, MessageTimeout + 60)` (new profile field `cmd_timeout_s`, `MessageTimeout` added to the discovery grep); `--test-only` is one exec per target. (§2.2, §4, §5.1, §6.1, §6.3)
12. **Cost**: new `cost_worst_su` (job column, plan option, submit result) = `cost_su × (1 + max_restarts)` when requeueable; feasibility and `max_extra_su` use it; candidates are always typed (no untyped `--gres`); `requeue=None` renders `--no-requeue` on charging `JobRequeue=1` clusters; new `needs_attention{why:"restart_cost"}`. (§3.2, §5.3, §8)
13. **Cancel semantics**: per-state decision in `job_control` (pending ⇒ plain `scancel` now; running ⇒ TERM + hard kill at `cancel_requested_ts + spec.grace_s`, new column `jobs.cancel_hard_ts`); the separate 30 s Monitor grace is gone; `ControlResult` rows gain `outcome`, `hard_kill_ts`. (§4, §5.2 step 5, §6.3)
14. **`alloc-agent.sh` v2**: single event loop; heartbeat every second regardless of foreground commands; kill file is `<base>.kill` (`002.kill`, `003.bg.kill`) in both the agent and `alloc.py`; `status.json` v2 with `fg`/`running`. (§7.3)
15. **Tests**: real Linux bash with stub SLURM binaries is mandatory (Git Bash = smoke only); composite-command golden against stub output; in-process asyncssh host-key test; Windows CI job for textio/transfers; new scenarios listed in §10. (§10)
16. **Quota / throttles / lab partition**: `::DF` queries `$HOME`, `remote_root`, `control_root`, `$PROJECT`, `$GROUP` and `profile.quota_paths`, rows carry the queried path; `clusters().quota` is a list with `role`; `PlacementPolicy.max_pending_per_target` defaults to `None` = discovered cap and means "hold locally" (new `JobState.QUEUED`, attempt `cause=queued`), never re-placement; new `PlacementPolicy.allow_self_preempt` and `target_overrides[*].allow_self_preempt`; `biosimmlab` disabled by default in the TRACE profile example. (§2.1, §3.1, §3.2, §6.1, §8)
18. **Per-directory quotas (2026-09-02, measured on TRACE)**: `::DF` rows are de-duplicated by `(mount, kb_total, kb_used)`, **not** by mount alone, and the quota guard picks the row governing the *destination path* via the new `parse.df_row_for_path(rows, path)` (longest queried-path prefix). On TRACE's VAST-over-NFS mount `df -Pk $HOME` reports 932 TB at 1 % while `df -Pk /trace/group/biosimmlab` reports 3 TB at 74 %; collapsing them by mount kept the home view and made the §5.5 quota guard unconditionally pass for uploads to the group volume. (§5.5, §6.1)
17. **File I/O without SFTP (2026-09-02, measured on Bridges-2)**: new `caps.login_sftp_ok`; every small-file operation of `SlurmClient` (write/mkdirs/ls/stat/deploy_helpers/ctrl-dir writes/`remote_write`) has an exec-channel implementation used when the login host refuses SFTP; bulk transfers extract tar on the login host after an SFTP put to the transfer host when the transfer host refuses exec (the PSC DTN does); the fake harness gains `login_sftp=False` and transfer-host `exec=False` modes. (§2.3, §5.1 step 6, §5.5, §6.1, §10)
