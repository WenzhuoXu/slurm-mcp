# Cluster survey (2026-09-01)

Empirical facts gathered through the slurm-mcp transport. Both clusters: SLURM 22.05.11, RHEL 8, OpenSSH 8.0,
`select/cons_tres` with `CR_CPU_MEMORY`, `AccountingStorageEnforce=associations,limits,qos,safe`,
`PriorityType=priority/multifactor` (FairShare 1e6, QOS 5e6, Age 1e4), backfill scheduler, `MaxArraySize=1001`,
`jq`, `tar`, `seff`, `sacct` (fields incl. JobIDRaw, DerivedExitCode, Reason, SubmitLine, WorkDir, Flags,
TRESUsageInTot, MaxRSS, ElapsedRaw; **no `Restarts` field** -> use `scontrol show job`). Neither accepts SSH keys
(publickey advertised, authorized_keys ignored); password auth, no 2FA.

## CMU TRACE (`trace.cmu.edu`, user wxu2)

| item | value |
|---|---|
| account / group | `biosimmlab` (gid 5017); QOS allowed: normal, batchpartition, prioritypartition, cpuonly-debug-qos |
| home | `/trace/home/wxu2` (NFS, 932 TB volume, no per-user quota shown) |
| work dir | `/trace/group/biosimmlab/wxu2/` (3 TB group volume, 76% used) |
| env vars | none of PROJECT/SCRATCH/LOCAL set |
| login node | 8 CPU / 31 GB — keep MCP-side work light |
| preemption | `PreemptType=preempt/partition_prio`, `PreemptMode=GANG,REQUEUE`; **`JobRequeue=0`** so jobs must pass `--requeue` to survive preemption; `KillWait=300s` |
| queue visibility | `MinJobAge=300s`; `bf_max_job_user=10` (only 10 pending jobs per user are considered by backfill) |
| rsync | `/usr/bin/rsync` available; `mail`/`sendmail` available |
| modules | anaconda3/2023.03-1 (user's scripts still load anaconda3/2021.05 + aocc/3.2.0 + cuda/11.7), cuda/12.8, python/3.11.2 |
| conda | `~/.conda/envs/pyg` (activated via `source ~/.bashrc; conda activate pyg`) |

Partitions (`sinfo`/`scontrol show partition`):

| partition | nodes | per node | max time | notes |
|---|---|---|---|---|
| `batch` | trace[01-29] (29) | 128c, 2 TB, **1x A40** | QoS batchpartition MaxWall **2 d** | tier 1, PreemptMode=REQUEUE (victim of priority partitions); DefMemPerCPU 8048 MB; AllowQos normal,batch |
| `biosimmlab` | same trace[01-29] | same | 2 d (QoS priorityphase1node1, prio 10000) | **tier 20 → preempts batch**; AllowGroups=biosimmlab; use for urgent work, mind lab etiquette (name suggests one node's worth) |
| `cpuonly` | trace[100-122] (23) | 96c, 773 GB, no GPU | 2 d | GANG,REQUEUE; was mostly idle at survey time |
| `cpuonly-debug` | trace123 (1) | 96c | 4 h | QoS cpu=24 |

Observed at survey: batch had 245 pending / 27 running; the user's 6 GPU jobs were top-priority (fairshare 0.25 →
priority ~251k vs next user 55k) but blocked on GPUs. `sbatch --test-only` estimates: batch 1-GPU job ~6 h wait,
biosimmlab immediate, cpuonly immediate. User job style (`jobs/train_wobl.job`):

```
#SBATCH -p batch
#SBATCH --gpus=a40
#SBATCH --ntasks-per-node=64
#SBATCH --mem=512G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -J wobl
#SBATCH -o logs/sweep/wobl_%j.out
#SBATCH -e logs/sweep/wobl_%j.err
module load aocc/3.2.0; module load cuda/11.7; module load anaconda3/2021.05
source /trace/home/wxu2/.bashrc; conda activate pyg
cd /trace/group/biosimmlab/wxu2/vascular_super_resolution
python3 run_DS_3D.py --dataset=... --exp_config=configs/sweep/runs/wobl_exp.yaml ...
```

## PSC Bridges-2 (`bridges2.psc.edu`, user wxu7; transfer node `data.bridges2.psc.edu`)

| item | value |
|---|---|
| account | `mch250030p` — **GPU-only allocation**, 482 / 3,639 SU left, ends 2026-12-08 |
| QOS allowed | `gpu`, `gpuinteract`, `low`, `push`, `unlimited`, `ft` — **no `rm`** → RM/RM-shared only via `--qos=low` (priority 0, NoReserve scavenger; started immediately in test) |
| home | `/jet/home/wxu7` — 25 GB quota, 70% used |
| project | `$PROJECT=/ocean/projects/mch250030p/wxu7` — project quota 405 GB, **95% used** (383 GB) |
| login node | 256c / 251 GB, load ~30, 87 users |
| preemption | `PreemptMode=OFF`; `JobRequeue=1`; `KillWait=400s`; `MinJobAge=200s` |
| scheduler | `bf_window=7200` (5 d), `bf_max_job_user_part=20`, `bf_resolution=3600`, `defer`, `kill_invalid_depend` |
| rsync | **not on login node** (tar only) |
| SFTP / exec (measured 2026-09-02) | **login node refuses the SFTP subsystem** (`Session request failed`) but exec channels work, incl. stdin writes (`cat > file`); **`data.bridges2.psc.edu` refuses exec** (`Login denied: <cmd> is not an allowed command`) but serves SFTP over the same `/ocean` and `/jet` filesystems |
| modules | anaconda3/2024.10-1, cuda/12.6.1 (D), cuda-h100/13.3.1, cuda-l40s, cuda-v100/12.9.2, AI/pytorch_23.02 |
| conda | `~/.conda/envs/my_env`, `$PROJECT/envs/battery` (plain venv-style python) |
| `interact` | `/opt/packages/interact/bin/interact -gpu -t ... -A ...` (salloc wrapper) |

GPU inventory (`--gres=gpu:<type>:N`; node features usable with `-C`):

| type | nodes | per node | mem/node |
|---|---|---|---|
| `v100-16` | 9 (v025-v033) | 8 GPU, 40c | 192 GB |
| `v100-32` | 23 (v002-v024) + 1 DGX-2 16-GPU (v034) | 8 GPU, 40c | 515 GB |
| `h100-80` | 10 (w001-w010) | 8 GPU, 104c | 2 TB |
| `l40s-48` | 3 (gl001-003) | 8 GPU, 192c | 1 TB |

Partitions the user can use:

| partition | limits | notes |
|---|---|---|
| `GPU-shared` | MaxNodes 1, MaxWall 2 d, DefMemPerGPU 63000 MB, OverSubscribe 4 | the workhorse; **3,434 pending h100-80:1 jobs** at survey; test-only waits: v100-32 ~2.6 d, h100-80 ~2.6 d, l40s ~2.7 d, v100-16 ~3 d |
| `GPU` | exclusive whole nodes, MaxWall 2 d, GrpTRES gpu=64,node=8 | test-only wait for v100-32:8 ~13 h |
| `GPU-small` | node v001 only (8x v100-32), MaxWall **8 h**, MaxJobsPU 2, MaxSubmit 10, GrpTRES gpu=16 | **fast lane**: test-only wait ~20 min |
| `RM-shared` (`--qos=low`) | 128c/256 GB nodes, DefMemPerCPU 1900 | CPU work; user's battery scripts use `--ntasks-per-node=4`, starts immediately |
| `RM` (`--qos=low`) | exclusive 128c nodes | |

User job style (`llm_finetune/slurm/*.sbatch`, logs in `logs/slurm/%x_%j.out`):

```
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --qos=gpu
set -euo pipefail
module load cuda/12.6.1; module load anaconda3/2024.10-1; conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONPATH=...; export HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models
python scripts/....py ...
```

Recent history shows FAILED (exit 1), TIMEOUT at the 8 h wall, and CANCELLED-by-user jobs on GPU-shared with 2x h100;
many short RM-shared `--qos=low` jobs. Results are pulled to `D:\LLM_FINETUNE\{logs,outputs,wandb,figures,emnlp_2026}`
with `scp -r` from `data.bridges2.psc.edu`.

## Design implications

- Auth: password from OS keyring; asyncssh answers keyboard-interactive. Keep one connection per cluster, ≤8 sessions.
- Login nodes are shared; keep remote helper work cheap (batched `squeue`/`sacct` every 30-60 s, not per job).
- Preemption exists only on TRACE `batch`; always inject `--requeue` and `--signal=B:USR1@<grace>` there and record
  `Restarts` from `scontrol show job`. On Bridges-2 the relevant failure modes are TIMEOUT and OOM, not preemption.
- Balancing targets are per (cluster, partition, gres, qos) tuples: TRACE batch / biosimmlab / cpuonly;
  Bridges-2 GPU-shared by GPU type / GPU-small (≤8 h) / GPU exclusive / RM-shared low. Score with `sbatch --test-only`
  estimated start, pending depth per gres, SU cost (Bridges-2 only, 482 SU left), and user policy caps
  (e.g. max concurrent jobs on `biosimmlab`).
- Transfers: SFTP everywhere; tar-stream for many small files; prefer `data.bridges2.psc.edu` for bulk; check
  quota headroom (`/ocean` at 95%) before uploading; results pulled by glob/manifest, not whole trees.
- `rsync` cannot be assumed (absent on Bridges-2 login node).
