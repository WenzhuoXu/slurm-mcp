"""Golden tests for slurm_mcp.slurm.commands (design sections 6.0-6.3).

Every composite command is compared verbatim with the block printed in design.md section 6,
modulo the ``<placeholders>`` (remote_root, control_root, ids, ctrl dirs, rc paths ...).
"""
from __future__ import annotations

import pytest

from slurm_mcp.slurm import commands as c

PRE = "export SLURM_TIME_FORMAT=%s LC_ALL=C"
RC = 'echo "::RC $?"'

TRACE_LIKE = {
    "remote_root": "/trace/group/biosimmlab/wxu2",
    "control_root": None,
    "quota_paths": ["/trace/group/biosimmlab"],
    "balance_command": None,
}
BRIDGES_LIKE = {
    "remote_root": "/ocean/projects/mch250030p/wxu7",
    "control_root": "$HOME/.slurm-mcp",
    "quota_paths": ["/ocean/projects/mch250030p"],
    "balance_command": "projects",
}


# -- quoting helpers ------------------------------------------------------------------------------

def test_shell_quote():
    assert c.shell_quote("batch") == "batch"
    assert c.shell_quote(615411) == "615411"
    assert c.shell_quote("") == "''"
    assert c.shell_quote("a b") == "'a b'"
    assert c.shell_quote("it's") == "'it'\"'\"'s'"
    assert c.shell_quote("--comment=slurm-mcp:j1:1:tok") == "--comment=slurm-mcp:j1:1:tok"
    assert c.shell_quote("$HOME") == "'$HOME'"


def test_path_quote_keeps_dollar_expansions():
    assert c.path_quote("$HOME/.slurm-mcp") == '"$HOME/.slurm-mcp"'
    assert c.path_quote('$HOME/we"ird`x') == '"$HOME/we\\"ird\\`x"'
    assert c.path_quote("/plain/path") == "/plain/path"
    assert c.path_quote("/with space/x") == "'/with space/x'"


def test_join_args_and_ids():
    assert c.join_args(["-p", "batch", "--mem=512G", "my arg"]) == "-p batch --mem=512G 'my arg'"
    assert c.join_ids([615411, "615427", "123_4", "123_[1-5]"]) == "615411,615427,123_4,123_[1-5]"
    assert c.join_ids([1, 2], " ") == "1 2"
    with pytest.raises(ValueError):
        c.join_ids(["1; rm -rf /"])
    with pytest.raises(ValueError):
        c.join_ids([""])
    with pytest.raises(ValueError):
        c.join_ids(["$(id)"])


def test_chunks_and_size_helpers():
    assert [list(x) for x in c.chunks(list(range(5)), 2)] == [[0, 1], [2, 3], [4]]
    assert list(c.chunks([], 3)) == []
    with pytest.raises(ValueError):
        list(c.chunks([1], 0))
    assert c.command_bytes("abc") == 3
    assert c.command_bytes("\u00e9") == 2
    assert c.MAX_CMD_BYTES == 8 * 1024
    assert not c.needs_split("x" * 8192)
    assert c.needs_split("x" * 8193)
    assert c.rc_echo() == RC
    assert c.section_echo("SACCT") == "echo '::SACCT'"


def test_effective_control_root():
    assert c.effective_control_root({"control_root": "/x/.slurm-mcp"}) == "/x/.slurm-mcp"
    assert c.effective_control_root({"remote_root": "/r"}) == "/r/.slurm-mcp"
    assert c.effective_control_root({}) == "$HOME/.slurm-mcp"

    class P:
        control_root = None
        remote_root = "/obj"

    assert c.effective_control_root(P()) == "/obj/.slurm-mcp"


# -- 6.1 discovery ---------------------------------------------------------------------------------

def _discovery_expected(remote_root: str, control_root_q: str, quota_q: str, balance: str | None) -> str:
    lines = [
        PRE,
        "echo '::ENV'; echo \"$HOME|$USER|$(hostname -f)|${PROJECT:-}|${SCRATCH:-}|${LOCAL:-}|$(date +%s)|"
        "$(date +%z)|$(id -gn)|$(id -Gn)\"",
        "echo '::VERSION'; sinfo --version",
        "echo '::CONFIG'; scontrol show config | grep -E '^(ClusterName|SLURM_VERSION|MinJobAge|MessageTimeout|"
        "PreemptMode|PreemptType|PreemptExemptTime|PreemptParameters|GraceTime|JobRequeue|KillWait|MaxArraySize|"
        "MaxJobCount|SchedulerParameters|DefMemPerCPU|DefMemPerNode|MaxMemPerCPU|MailProg|AccountingStorageEnforce|"
        "AccountingStoreFlags|EnforcePartLimits|PriorityWeight(Age|FairShare|QOS|Partition|JobSize)|BOOT_TIME) '",
        f"echo '::PARTITIONS'; scontrol show partition -o; {RC}",
        f"echo '::SINFO'; sinfo -h -e -N -o '%N|%R|%t|%c|%m|%G|%f'; {RC}",
        "echo '::USER'; sacctmgr -nP show user \"$USER\" format=User,DefaultAccount",
        "echo '::ASSOC'; sacctmgr -nP show assoc where user=\"$USER\" format=Cluster,Account,Partition,QOS,DefaultQOS,"
        f"GrpTRES,GrpTRESMins,MaxJobs,MaxSubmit,MaxTRES,MaxWall; {RC}",
        "echo '::QOS'; sacctmgr -nP show qos format=Name,Priority,GraceTime,MaxWall,MaxTRES,MaxTRESPU,MaxJobsPU,"
        f"MaxSubmitPU,GrpTRES,Preempt,PreemptMode,Flags,UsageFactor; {RC}",
        f"echo '::SSHARE'; sshare -nP -U -u \"$USER\" -o Account,User,FairShare,GrpTRESMins,GrpTRESRaw; {RC}",
        "echo '::RESV'; scontrol -o show reservation 2>/dev/null",
        "echo '::TOOLS'; for t in tar sacct squeue sbatch scontrol sinfo sacctmgr scancel srun sshare sha256sum stat "
        "timeout setsid rsync jq seff flock; do printf '%s=' \"$t\"; command -v \"$t\" >/dev/null 2>&1 && echo 1 || "
        "echo 0; done",
        "echo '::CAP_O'; squeue --me -h -t all -O 'JobID:0|,RestartCnt:0|,tres-per-node:0|,tres-per-job:0|' "
        ">/dev/null 2>&1; echo \"rc=$?\"",
        f"echo '::DF'; for p in \"$HOME\" {remote_root} {control_root_q} \"${{PROJECT:-}}\" \"${{GROUP:-}}\" {quota_q}; "
        "do [ -n \"$p\" ] && [ -d \"$p\" ] && df -Pk \"$p\" 2>/dev/null | tail -n +2 | sed \"s|\\$| $p|\"; done",
    ]
    if balance:
        lines.append(f"echo '::BALANCE'; {balance} 2>/dev/null | head -40")
    helper = control_root_q.rstrip('"') + "/bin/VERSION" + ('"' if control_root_q.endswith('"') else "")
    lines.append(f"echo '::HELPER'; cat {helper} 2>/dev/null")
    lines.append("echo '::END'")
    return "\n".join(lines)


def test_discovery_golden_trace_like():
    expected = _discovery_expected("/trace/group/biosimmlab/wxu2", "/trace/group/biosimmlab/wxu2/.slurm-mcp",
                                   "/trace/group/biosimmlab", None)
    assert c.discovery(TRACE_LIKE) == expected
    assert "::BALANCE" not in c.discovery(TRACE_LIKE)


def test_discovery_golden_bridges_like():
    expected = _discovery_expected("/ocean/projects/mch250030p/wxu7", '"$HOME/.slurm-mcp"',
                                   "/ocean/projects/mch250030p", "projects")
    out = c.discovery(BRIDGES_LIKE)
    assert out == expected
    assert "echo '::BALANCE'; projects 2>/dev/null | head -40" in out
    assert "cat \"$HOME/.slurm-mcp/bin/VERSION\" 2>/dev/null" in out


def test_discovery_accepts_object_profile_and_defaults():
    class Profile:
        remote_root = None
        control_root = None
        quota_paths = ()
        balance_command = None

    out = c.discovery(Profile())
    assert out.startswith(PRE + "\n")
    assert out.endswith("\necho '::END'")
    assert 'for p in "$HOME" "$HOME/.slurm-mcp" "${PROJECT:-}" "${GROUP:-}"; do' in out


def test_helper_deploy_check_and_backfill_and_recheck():
    assert c.helper_deploy_check("$HOME/.slurm-mcp") == f'{PRE}; cat "$HOME/.slurm-mcp/bin/VERSION" 2>/dev/null'
    assert c.helper_deploy_check("/r/.slurm-mcp/") == f"{PRE}; cat /r/.slurm-mcp/bin/VERSION 2>/dev/null"
    assert c.backfill_history() == (f'{PRE}; sacct -nP -X -u "$USER" -S now-30days '
                                    "-o JobIDRaw,Partition,QOS,ReqTRES,Submit,Start,State")
    assert c.backfill_history("wxu2") == (f"{PRE}; sacct -nP -X -u wxu2 -S now-30days "
                                          "-o JobIDRaw,Partition,QOS,ReqTRES,Submit,Start,State")
    assert c.recheck_pending() == f"{PRE}; squeue --me -h -o '%A|%T'"


# -- 6.2 tick ---------------------------------------------------------------------------------------

SQUEUE_LINE = ("echo '::SQUEUE'; squeue --me -h -r -t all -o "
               f"'%A|%i|%F|%K|%T|%P|%q|%S|%e|%V|%l|%M|%Q|%N|%b|%k|%o|%Z|%r'; {RC}")
RESTARTS_LINE = f"echo '::RESTARTS'; squeue --me -h -r -t all -O 'JobID:0|,RestartCnt:0|,Requeue:0|'; {RC}"
SACCT_FIELDS = ("JobIDRaw,JobID,State,ExitCode,DerivedExitCode,Partition,QOS,NodeList,Submit,Start,End,ElapsedRaw,"
                "TimelimitRaw,AllocTRES,ReqTRES,Reason,WorkDir")


def _files_line(dirs: str) -> str:
    return (f"echo '::FILES'; for d in {dirs}; do for f in jobid status.json heartbeat; do "
            "[ -f \"$d/$f\" ] && printf '%s|%s|%s\\n' \"$d\" \"$f\" \"$(head -c 1000 \"$d/$f\" | tr '\\n\\r|' '   ')\"; "
            "done; [ -f \"$d/progress.json\" ] && printf '%s|progress.json|%s\\n' \"$d\" "
            "\"$(tail -c 1024 \"$d/progress.json\" | tail -n 1 | tr '|' ' ')\"; done")


def _cmds_line(paths: str) -> str:
    return f"echo '::CMDS'; for f in {paths}; do [ -f \"$f\" ] && printf '%s|%s\\n' \"$f\" \"$(cat \"$f\")\"; done"


def test_tick_golden_full():
    execs = c.tick([615411, 615427], ["/h/.slurm-mcp/jobs/j1/a1"], ["/h/.slurm-mcp/jobs/a1/c1.rc"], recover=True,
                   enrich_ids=[615400], stdout_paths=["/w/out.txt"], caps={"squeue_O_zero": True})
    expected = "\n".join([
        PRE,
        'echo "::NOW $(date +%s) $(hostname -s)"',
        SQUEUE_LINE,
        RESTARTS_LINE,
        f"echo '::SACCT'; sacct -n -P -X -D -j 615411,615427 -o {SACCT_FIELDS}; {RC}",
        _files_line("/h/.slurm-mcp/jobs/j1/a1"),
        _cmds_line("/h/.slurm-mcp/jobs/a1/c1.rc"),
        f"echo '::RECOVER'; sacct -n -P -X -u \"$USER\" -S now-2hours -o JobIDRaw,Submit,State,WorkDir,SubmitLine; {RC}",
        f"echo '::ENRICH'; sacct -n -P -j 615400 -o JobIDRaw,JobID,State,ExitCode,MaxRSS,ReqMem,ElapsedRaw,AllocTRES; {RC}; "
        "for f in /w/out.txt; do printf '::L %s|%s\\n' \"$f\" \"$(tail -n 1 \"$f\" 2>/dev/null | head -c 300 | "
        "tr '|' ' ')\"; done",
        "echo '::END'",
    ])
    assert execs == [expected]


def test_tick_minimal_sections():
    execs = c.tick([])
    assert len(execs) == 1
    out = execs[0]
    assert out == "\n".join([PRE, 'echo "::NOW $(date +%s) $(hostname -s)"', SQUEUE_LINE, "echo '::SACCT'",
                             "echo '::FILES'", "echo '::CMDS'", "echo '::END'"])
    assert "::RESTARTS" not in out and "::RECOVER" not in out and "::ENRICH" not in out


def test_tick_restarts_only_with_capability():
    assert "::RESTARTS" not in c.tick([1], caps={"squeue_O_zero": False})[0]
    assert "::RESTARTS" not in c.tick([1], caps=None)[0]
    assert RESTARTS_LINE in c.tick([1], caps={"squeue_O_zero": True})[0]


def test_tick_enrich_without_stdout_paths():
    out = c.tick([1], enrich_ids=["2", "3"])[0]
    assert (f"echo '::ENRICH'; sacct -n -P -j 2,3 -o JobIDRaw,JobID,State,ExitCode,MaxRSS,ReqMem,ElapsedRaw,AllocTRES; {RC}"
            "\necho '::END'") in out
    assert "::L" not in out


def test_tick_sacct_chunks_of_100():
    ids = list(range(1000, 1250))
    out = c.tick(ids, caps={})[0]
    assert out.count("echo '::SACCT'; sacct -n -P -X -D -j ") == 3
    assert "-j " + ",".join(str(i) for i in ids[:100]) + " -o" in out
    assert "-j " + ",".join(str(i) for i in ids[200:]) + " -o" in out


def test_tick_quotes_paths_with_spaces_and_dollar():
    out = c.tick([1], ["$HOME/.slurm-mcp/jobs/j1/a1", "/tmp/with space"], ["$HOME/x.rc"])[0]
    assert 'for d in "$HOME/.slurm-mcp/jobs/j1/a1" \'/tmp/with space\'; do' in out
    assert 'for f in "$HOME/x.rc"; do' in out


def test_tick_split_rule_8kb():
    dirs = [f"/trace/group/biosimmlab/wxu2/.slurm-mcp/jobs/job{i:04d}/a1" for i in range(150)]
    rcs = [f"/trace/group/biosimmlab/wxu2/.slurm-mcp/jobs/alloc{i:04d}/cmds/c1.rc" for i in range(40)]
    execs = c.tick([615411], dirs, rcs, caps={"squeue_O_zero": True})
    assert len(execs) >= 2
    first = execs[0]
    assert first.startswith(PRE + "\n" + 'echo "::NOW $(date +%s) $(hostname -s)"')
    assert SQUEUE_LINE in first and "echo '::SACCT'; sacct" in first
    assert "::FILES" not in first and "::CMDS" not in first
    assert first.endswith("echo '::END'")
    assert c.command_bytes(first) <= c.MAX_CMD_BYTES
    seen_dirs: list[str] = []
    seen_rcs: list[str] = []
    for ex in execs[1:]:
        assert ex.startswith(PRE + "\n")
        assert ex.endswith("\necho '::END'")
        assert "::NOW" not in ex and "::SQUEUE" not in ex and "::SACCT" not in ex
        assert "echo '::FILES'" in ex and "echo '::CMDS'" in ex
        assert c.command_bytes(ex) <= c.MAX_CMD_BYTES
        seen_dirs += [d for d in dirs if f" {d} " in ex or f" {d};" in ex]
        seen_rcs += [p for p in rcs if f" {p} " in ex or f" {p};" in ex]
    assert seen_dirs == dirs
    assert seen_rcs == rcs


def test_tick_split_respects_custom_limit():
    dirs = [f"/ctrl/j{i}/a1" for i in range(20)]
    execs = c.tick([1], dirs, [], limit=600)
    assert len(execs) > 2
    assert all(c.command_bytes(ex) <= 600 for ex in execs[1:])


# -- 6.2 snapshot -----------------------------------------------------------------------------------

def test_snapshot_golden_with_O_capability():
    expected = "\n".join([
        PRE,
        f"echo '::NODES'; sinfo -h -e -N -o '%R|%t|%G|%C'; {RC}",
        "echo '::PD'; squeue -h -t PD -O 'Partition:0|,tres-per-node:0|,tres-per-job:0|' | sort | uniq -c; "
        "echo \"::RC ${PIPESTATUS[0]}\"",
        "echo '::R';  squeue -h -t R  -O 'Partition:0|,tres-per-node:0|,tres-per-job:0|' | sort | uniq -c; "
        "echo \"::RC ${PIPESTATUS[0]}\"",
        "echo '::MINE'; squeue --me -h -t PD -O 'JobID:0|,Partition:0|,tres-per-node:0|,tres-per-job:0|,PriorityLong:0|,"
        "StartTime:0|,Reason:0|'",
        "echo '::RESV'; scontrol -o show reservation 2>/dev/null",
        "echo '::END'",
    ])
    assert c.snapshot({"squeue_O_zero": True}) == expected


def test_snapshot_fallback_without_capability():
    out = c.snapshot({})
    assert "squeue -h -t PD -o '%P|%b' | sort | uniq -c" in out
    assert "squeue -h -t R  -o '%P|%b' | sort | uniq -c" in out
    assert "squeue --me -h -t PD -o '%A|%P|%b|%Q|%S|%r'" in out
    assert "-O" not in out
    assert c.snapshot(None) == out


# -- 6.3 submit / estimate / control ---------------------------------------------------------------

def test_submit_golden():
    out = c.submit("/w", "/h/.slurm-mcp/bin/abcd1234", "/h/.slurm-mcp/jobs/j1/a1", "tok",
                   ["-p", "batch", "--comment=slurm-mcp:j1:1:tok", "--parsable"])
    assert out == (f"{PRE}; cd /w && bash /h/.slurm-mcp/bin/abcd1234/submit.sh /h/.slurm-mcp/jobs/j1/a1 tok -- "
                   "-p batch --comment=slurm-mcp:j1:1:tok --parsable /h/.slurm-mcp/jobs/j1/a1/job.sbatch")


def test_submit_quoting_and_explicit_script():
    out = c.submit("/w dir", "$HOME/.slurm-mcp/bin/aa/", "$HOME/.slurm-mcp/jobs/j1/a1", "t o k", [],
                   script_path="/x/alloc.sbatch")
    assert out == (f"{PRE}; cd '/w dir' && bash \"$HOME/.slurm-mcp/bin/aa/submit.sh\" \"$HOME/.slurm-mcp/jobs/j1/a1\" "
                   "'t o k' -- /x/alloc.sbatch")


def test_test_only_golden():
    args = ["-p", "batch", "--qos=normal", "-A", "biosimmlab", "-t", "24:00:00", "--mem=512G", "--gres=gpu:a40:1",
            "--requeue", "--open-mode=append", "--signal=B:USR1@120", "-o", "/w/logs/wobl_%j.out",
            "--hold", "--comment=slurm-mcp:j1:1:tok", "--parsable"]
    out = c.test_only("/trace/group/biosimmlab/wxu2/vascular_super_resolution", args, "/ctrl/a1/job.sbatch")
    assert out == (f"{PRE}; cd /trace/group/biosimmlab/wxu2/vascular_super_resolution\n"
                   "echo '::T1'; sbatch --test-only -p batch --qos=normal -A biosimmlab -t 24:00:00 --mem=512G "
                   "--gres=gpu:a40:1 --requeue --open-mode=append --signal=B:USR1@120 -o /w/logs/wobl_%j.out "
                   f"/ctrl/a1/job.sbatch 2>&1; {RC}; echo '::END'")


def test_test_only_section_and_two_token_comment():
    out = c.test_only("/w", ["--comment", "x y", "-p", "GPU-shared"], "/c/job.sbatch", section="T3")
    assert out == f"{PRE}; cd /w\necho '::T3'; sbatch --test-only -p GPU-shared /c/job.sbatch 2>&1; {RC}; echo '::END'"
    assert c.test_only("/w", [], "/c/job.sbatch") == (f"{PRE}; cd /w\necho '::T1'; sbatch --test-only /c/job.sbatch 2>&1; "
                                                       f"{RC}; echo '::END'")


def test_strip_for_test_only():
    assert c.strip_for_test_only(["--parsable", "-p", "x", "--hold", "--comment=a", "--comment", "b", "-t", "1"]) == \
        ["-p", "x", "-t", "1"]
    assert c.strip_for_test_only([]) == []


def test_cancel_variants():
    assert c.cancel([615411]) == f"{PRE}; scancel 615411"
    assert c.cancel([1, 2]) == f"{PRE}; scancel 1 2"
    assert c.cancel(["123_4"]) == f"{PRE}; scancel 123_4"
    assert c.cancel([1, 2], signal="TERM", full=True) == f"{PRE}; scancel --signal=TERM --full 1 2"
    assert c.cancel([7], signal="USR1", batch=True) == f"{PRE}; scancel --signal=USR1 --batch 7"
    assert c.cancel([7], signal="USR1", full=True, batch=True) == f"{PRE}; scancel --signal=USR1 --full --batch 7"
    with pytest.raises(ValueError):
        c.cancel([])
    with pytest.raises(ValueError):
        c.cancel(["1 2; ls"])


def test_hold_release_requeue():
    assert c.hold([1, 2]) == f"{PRE}; scontrol hold 1,2"
    assert c.release(["123_4"]) == f"{PRE}; scontrol release 123_4"
    assert c.requeue([9]) == f"{PRE}; scontrol requeue 9"
    for fn in (c.hold, c.release, c.requeue):
        with pytest.raises(ValueError):
            fn([])


def test_update_dependency_and_show_job():
    assert c.update_dependency(5, ["afterok:3", "afterany:4"]) == \
        f"{PRE}; scontrol update JobId=5 Dependency=afterok:3,afterany:4"
    assert c.update_dependency("5", "afterok:3?afternotok:4") == \
        f"{PRE}; scontrol update JobId=5 Dependency='afterok:3?afternotok:4'"
    assert c.update_dependency(5, []) == f"{PRE}; scontrol update JobId=5 Dependency="
    assert c.show_job(615411) == f"{PRE}; scontrol -o show job 615411"
    assert c.show_job("123_4") == f"{PRE}; scontrol -o show job 123_4"
    with pytest.raises(ValueError):
        c.show_job("1;ls")


def test_every_command_starts_with_preamble():
    outs = [c.discovery(TRACE_LIKE), c.snapshot({}), c.submit("/w", "/b", "/c", "t", []),
            c.test_only("/w", [], "/s"), c.cancel([1]), c.hold([1]), c.release([1]), c.requeue([1]),
            c.update_dependency(1, []), c.show_job(1), c.backfill_history(), c.recheck_pending(),
            c.helper_deploy_check("/r")] + c.tick([1])
    for out in outs:
        assert out.startswith(PRE)
        assert "\r" not in out


def test_all_exports_exist():
    for name in c.__all__:
        assert hasattr(c, name), name
