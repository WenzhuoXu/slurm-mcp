# slurm-mcp design brief

An MCP server (Python, `mcp` SDK 2.x `MCPServer`, asyncssh, keyring) that lets a Claude Code session on a laptop
drive SLURM jobs on any SSH-reachable cluster. Companion facts: `docs/clusters.md` (the two real clusters this must
work on first: CMU TRACE and PSC Bridges-2).

## Requirements (from the user, verbatim intent)

1. **Provision compute** — request resources on a cluster: batch allocations, and reusable interactive-style
   allocations (reserve a GPU node for N hours, then run many short commands in it without re-queuing).
2. **Send compute jobs** — upload code/data, render and submit sbatch scripts (from a spec or a user script),
   job arrays, dependencies.
3. **Monitor queue and progress** — cluster load, my queue, per-job state, logs, progress signals from running jobs.
4. **Auto-balance between partitions (and clusters)** when queues are long — choose the best target at submit time,
   and re-place still-pending jobs when a better option appears; respect cost (SU balance) and etiquette caps.
5. **Notifications** on completion / failure / timeout / preemption / requeue / OOM, delivered to the Claude Code
   session (pollable) and to the human (desktop toast; optional e-mail via SLURM mail, webhook).
6. **Collect results** — pull outputs back by glob/manifest, incrementally, to a local directory.
7. **General** — any SSH+SLURM cluster once given endpoint + credentials; nothing cluster-specific hard-coded.

## Hard constraints discovered

- Windows laptop client (Python 3.12, no rsync locally, Git Bash available). Clusters: SLURM 22.05, no REST API,
  no `--json` output plugin assumed. Password auth only (no SSH keys), stored in OS keyring; asyncssh answers
  keyboard-interactive prompts. One SSH connection per cluster, ≤8 concurrent sessions (sshd MaxSessions 10).
- Login nodes are shared (TRACE's is tiny: 8 CPU/31 GB). Poll cheaply and in batches (one `squeue` + one `sacct`
  per cluster per tick, 30–60 s), never a process per job.
- `sacct` on 22.05 lacks `Restarts`; `scontrol show job -o` has it but only while the job is in memory
  (`MinJobAge` 200–300 s after end). `sbatch --test-only` gives a start estimate and prints
  `sbatch: Job N to start at <ISO> using P processors on nodes X in partition Y` (no job is created).
- Preemption only on TRACE `batch` (`PreemptMode=REQUEUE`, cluster `JobRequeue=0` → must pass `--requeue`).
  Bridges-2: no preemption, but SU charging (482 SU left), `GPU-small` 8 h fast lane, RM only via `--qos=low`.
- MCP client is Claude Code: tools are request/response with a per-call timeout (assume ~60 s default, configurable);
  server-initiated notifications may or may not be surfaced, so the reliable path is a `wait_for_events`
  long-poll tool with bounded wait plus a durable event log; keep tool outputs compact (token-limited).
- `rsync` cannot be assumed on the remote side; `tar`, `sftp`, `jq` can (verify at bootstrap).

## Proposed architecture (to be challenged by the design panel)

```
Claude Code ──stdio──> slurm-mcp server (one process, asyncio)
                        ├── ClusterRegistry: profiles (~/.slurm-mcp/config.json) + keyring
                        ├── SSHTransport per cluster (asyncssh; run(), sftp(), tar-stream)
                        ├── SlurmClient per cluster: typed wrappers + parsers for sbatch/squeue/sacct/sinfo/scontrol/scancel/sprio
                        ├── RemoteHelper (optional, deployed to ~/.slurm-mcp on cluster):
                        │     wrap.sh   – runs inside jobs: traps USR1/TERM, writes status.json/heartbeat/progress
                        │     probe.sh  – one round trip returning squeue+sacct+sinfo as JSON (jq)
                        ├── JobStore: SQLite ~/.slurm-mcp/state.db (jobs, attempts, events, transfers, allocations, policies)
                        ├── Monitor: background task; per-cluster tick → reconcile store vs cluster → emit events
                        ├── Placer: candidate targets → feasibility → estimate (--test-only, queue depth) → cost → policy → choice
                        ├── Rebalancer: periodic re-placement of pending auto-placed jobs (cancel+resubmit w/ lineage)
                        ├── Transfer: upload/download/sync with ignore rules, manifests, tar-stream, quota check
                        └── Notifier: event log + wait_for_events long-poll; Windows toast; webhook; SLURM --mail
```

Key models: `ClusterProfile`, `Target` (cluster, partition, qos, gres/gpu type, constraints), `JobSpec` (name,
command or script, workdir, resources, env, modules/conda, requeue+grace, inputs, outputs, targets/placement
policy), `JobRecord` (id, spec, attempts[] with cluster/jobid/state/exit/reason/restarts, current state, events),
`Event` (seq, time, job, kind, payload, acked).

Job lifecycle states (ours, superset of SLURM): DRAFT → UPLOADING → SUBMITTED(PENDING) → RUNNING →
{COMPLETED, FAILED, TIMEOUT, OOM, CANCELLED, PREEMPTED→REQUEUED→PENDING, NODE_FAIL} → COLLECTED.

Tool surface (draft, ~25 tools; names are the contract):
- clusters: `list_clusters`, `cluster_status(cluster)`, `bootstrap_cluster(cluster)` (deploy helper, verify tools,
  discover partitions/gres/qos/quotas; cached), `run_command(cluster, cmd, timeout)` (escape hatch).
- files: `upload(cluster, local, remote, ignore?, mode=sftp|tar)`, `download(cluster, remote_globs, local)`,
  `sync_project(cluster, local_dir, remote_dir)` (manifest-based incremental), `remote_ls`, `remote_read(path,
  head/tail/grep)`, `remote_write(path, text)`.
- jobs: `plan_placement(spec)` (dry run: options table with estimates/costs), `submit_job(spec, placement=auto|target)`,
  `submit_script(cluster, script_path_or_text, partition?, extra_args)`, `list_jobs(filters)`, `job_status(id)`,
  `job_logs(id, tail_lines, grep?)`, `cancel_job(id)`, `hold_job/release_job/requeue_job`, `rebalance_jobs(ids?, dry_run)`,
  `set_policy(...)`.
- allocations: `reserve_allocation(cluster, resources, hours)`, `run_in_allocation(alloc_id, cmd)`, `release_allocation`.
- events: `wait_for_events(timeout_s≤600, kinds?, job_ids?)`, `list_events(since_seq)`, `ack_events`,
  `configure_notifications(toast, webhook_url, email)`.
- results: `collect_results(job_id, local_dir)` (uses spec.outputs; records manifest).

Open questions for the panel: (a) helper-script-in-job vs pure sacct tracking; (b) rebalancing by cancel+resubmit vs
SLURM multi-partition (`--partition=a,b`) vs racing duplicates; (c) SQLite vs JSON state; (d) how much to auto-inject
(`--requeue`, `--signal`, `-A`, `--qos`) vs leave to the user; (e) allocation reuse via `srun --jobid --overlap`
reliability; (f) tar-stream vs SFTP thresholds; (g) event delivery to Claude Code in practice.
