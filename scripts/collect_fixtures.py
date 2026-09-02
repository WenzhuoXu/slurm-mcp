"""Capture raw SLURM command outputs from a real cluster into tests/fixtures/<cluster>/.

Usage: python scripts/collect_fixtures.py <cluster-profile> [job_id_running] [job_id_pending]
Each command's stdout/stderr/rc is stored so parser tests run against genuine output.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slurm_mcp.config import get_profile  # noqa: E402
from slurm_mcp.transport import SSHTransport  # noqa: E402

SQUEUE_FMT = "%i|%j|%T|%P|%R|%M|%l|%D|%C|%b|%S|%V|%Q|%r|%N|%o|%Z|%u|%a|%q|%k|%e|%L"
SACCT_FMT = ("JobID,JobIDRaw,JobName,State,ExitCode,DerivedExitCode,Elapsed,ElapsedRaw,Start,End,Submit,"
             "Partition,Account,QOS,NodeList,AllocTRES,ReqTRES,MaxRSS,TotalCPU,Reason,WorkDir,Timelimit,"
             "TimelimitRaw,NCPUS,NNodes,Flags,SubmitLine")

COMMANDS = {
    "sinfo_partitions": 'sinfo -h -o "%P|%a|%l|%D|%t|%C|%G|%m|%F|%N"',
    "sinfo_nodes": 'sinfo -h -N -o "%N|%P|%t|%c|%m|%G|%f|%C|%e|%O"',
    "sinfo_summary": 'sinfo -s -h -o "%P|%a|%l|%F|%N"',
    "scontrol_partitions": "scontrol show partition -o",
    "scontrol_config": "scontrol show config",
    "squeue_me": f'squeue -u $USER -h -o "{SQUEUE_FMT}"',
    "squeue_me_start": 'squeue -u $USER --start -h -o "%i|%j|%P|%S|%R|%Q|%T"',
    "squeue_all_counts": 'squeue -h -o "%P|%T|%b" | sort | uniq -c',
    "sacct_me_alloc": f'sacct -u $USER -X -n -P -S $(date -d "45 days ago" +%F) --format={SACCT_FMT}',
    "sacct_me_steps": f'sacct -u $USER -n -P -S $(date -d "10 days ago" +%F) --format={SACCT_FMT}',
    "sacct_bad_job": "sacct -j 1 -n -P --format=JobID,State",
    "sprio_me": 'sprio -u $USER -o "%i|%r|%Y|%A|%F|%J|%P|%Q|%T"',
    "sshare_me": "sshare -U -P",
    "sacctmgr_assoc": "sacctmgr -n -P show assoc user=$USER format=cluster,account,partition,qos,maxjobs,maxsubmit,grptres",
    "sacctmgr_qos": "sacctmgr -n -P show qos format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu,grptres,maxtres,flags",
    "sbatch_test_only_ok": 'sbatch --test-only -N 1 -n 1 -t 0:10:00 --wrap hostname',
    "sbatch_test_only_bad_partition": 'sbatch --test-only -p no_such_partition -N 1 -n 1 -t 0:10:00 --wrap hostname',
    "sbatch_test_only_bad_gres": 'sbatch --test-only --gres=gpu:nosuchgpu:1 -N 1 -n 1 -t 0:10:00 --wrap hostname',
    "sbatch_test_only_bad_time": 'sbatch --test-only -N 1 -n 1 -t 99-00:00:00 --wrap hostname',
    "scancel_bad_job": "scancel 1",
    "scontrol_show_job_missing": "scontrol show job 1 -o",
    "env": 'echo "HOME=$HOME"; echo "USER=$USER"; echo "PROJECT=${PROJECT:-}"; echo "SHELL=$SHELL"; echo "SLURM=$(sinfo --version)"',
    "tools": 'for t in sbatch squeue sacct sinfo scontrol scancel sprio sshare sacctmgr seff sstat tar jq rsync python3 flock timeout; do printf "%s=" $t; command -v $t || echo -; done',
    "df_home": "df -hP $HOME | tail -1; df -hP ${PROJECT:-$HOME} | tail -1",
    "quota": "quota -s 2>&1 | head -20",
}


async def main() -> None:
    name = sys.argv[1]
    extra_jobs = sys.argv[2:]
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / name
    out.mkdir(parents=True, exist_ok=True)
    cmds = dict(COMMANDS)
    for j in extra_jobs:
        cmds[f"scontrol_show_job_{j}"] = f"scontrol show job {j} -o"
        cmds[f"sacct_job_{j}"] = f"sacct -j {j} -n -P --format={SACCT_FMT}"
    index = {}
    async with SSHTransport(get_profile(name)) as t:
        for key, cmd in cmds.items():
            r = await t.run(cmd, timeout=120)
            (out / f"{key}.out").write_text(r.stdout, encoding="utf-8")
            if r.stderr:
                (out / f"{key}.err").write_text(r.stderr, encoding="utf-8")
            index[key] = {"cmd": cmd, "rc": r.returncode, "stdout_bytes": len(r.stdout), "stderr_bytes": len(r.stderr)}
            print(f"{key:32s} rc={r.returncode:<3d} out={len(r.stdout):>7d}B err={len(r.stderr):>5d}B")
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
