"""Unit tests for slurm_mcp.render (design sections 5.1 step 1, 5.3, 6.1, 6.3, 11d; changelog items 1-3, 12).

The golden tests use the two real user scripts from docs/clusters.md (TRACE ``train_wobl.job`` and the Bridges-2
GPU-shared script) with discovery caps shaped like ``slurm/parse.py`` output built from the fixtures in
``tests/fixtures/{trace,bridges2}`` (scontrol_partitions.out, sacctmgr_assoc.out, scontrol_config.out).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from slurm_mcp import render
from slurm_mcp.config import ClusterProfile
from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.models import JobSpec, Target
from slurm_mcp.render import (ALLOC_CLI_ARGS, MAIL_TYPES, NO_VAL_ARRAY_INDEX, ParsedScript, RenderedArgs,
                              build_target_args, choose_qos, cluster_charges, expand_pattern, format_submit_line,
                              merge_spec, parse_sbatch, partition_family, pattern_needs_controller, render_alloc_sbatch,
                              render_env_sh, render_job_sbatch, render_user_body, requeue_flag, requeueable,
                              resolve_output_patterns, strip_for_test_only, target_args)

# --- the two real scripts (docs/clusters.md) --------------------------------------------------------

TRACE_SCRIPT = """#!/bin/bash
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
"""
TRACE_WORKDIR = "/trace/group/biosimmlab/wxu2/vascular_super_resolution"

BRIDGES_SCRIPT = """#!/bin/bash
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
"""
BRIDGES_WORKDIR = "/ocean/projects/mch250030p/wxu7/llm_finetune"

# --- discovery caps shaped like slurm/parse.py output, values from tests/fixtures ---------------------

TRACE_CAPS = {
    "cluster_name": "trace", "job_requeue": False, "charges": False, "default_account": "biosimmlab",
    "partitions": {
        "batch": {"name": "batch", "allow_qos": ["normal", "batch"], "qos": "batchpartition",
                  "preempt_mode": ["REQUEUE"], "priority_tier": 1},
        "biosimmlab": {"name": "biosimmlab", "allow_qos": ["normal", "batch"], "qos": "priorityphase1node1",
                       "preempt_mode": ["OFF"], "priority_tier": 20},
        "cpuonly": {"name": "cpuonly", "allow_qos": ["ALL"], "qos": None, "preempt_mode": ["GANG", "REQUEUE"]},
    },
    "assoc": {"account": "biosimmlab", "partition": None, "default_qos": None,
              "qos_list": ["batchpartition", "cpuonly-debug-qos", "normal", "prioritypartition"]},
}
BRIDGES_CAPS = {
    "cluster_name": "bridges2", "job_requeue": True, "charges": True, "default_account": "mch250030p",
    "partitions": {
        "GPU-shared": {"name": "GPU-shared", "allow_qos": ["gpu", "gpuinteract", "low", "push", "unlimited"],
                       "qos": "gpusharedpartition", "preempt_mode": ["OFF"]},
        "RM-shared": {"name": "RM-shared", "allow_qos": ["rm", "rminteract", "low", "push", "unlimited"],
                      "qos": "rmsharedpartition", "preempt_mode": ["OFF"]},
        "GPU-small": {"name": "GPU-small", "allow_qos": ["gpu", "gpuinteract", "low", "push", "unlimited"],
                      "qos": "gpusmallpartition", "preempt_mode": ["OFF"]},
    },
    "assoc": {"account": "mch250030p", "partition": None, "default_qos": None,
              "qos_list": ["ft", "gpu", "gpuinteract", "low", "push", "unlimited"]},
}
TRACE_PROFILE = ClusterProfile(name="trace", host="trace.example", user="u1", no_mem_flag=[])
BRIDGES_PROFILE = ClusterProfile(name="bridges2", host="b2.example", user="u2", no_mem_flag=["RM-shared"])
STAMP = "2026-09-02T10:00:00-04:00"
SHA8 = "0123abcd"


def _spec(**kw) -> JobSpec:
    base = {"name": "t", "command": "echo hi", "resources": {"time": "01:00:00"}}
    base.update(kw)
    return JobSpec.parse(base)


def _forbidden_lines(text: str) -> list[str]:
    """``#SBATCH`` lines the server must never leave in job.sbatch (design section 6.3)."""
    bad = re.compile(r"^#SBATCH\s+(-p\b|--partition|--gpus|--gres|--mem|-o\b|--output|-e\b|--error|-q\b|--qos|"
                     r"-t\b|--time|-A\b|--account|--requeue|--no-requeue|--signal|--mail|--comment|--array|-a\b|"
                     r"--dependency|-d\b|--export|--chdir|-D\b)")
    return [ln for ln in text.splitlines() if bad.match(ln)]


# =================================================================================================
# parse_sbatch (5.1 step 1)
# =================================================================================================

def test_parse_trace_script_fields_and_stripping():
    p = parse_sbatch(TRACE_SCRIPT)
    assert isinstance(p, ParsedScript)
    fields, extra, body, stripped, warnings = p
    assert fields["partition"] == "batch"
    assert fields["name"] == "wobl"
    assert fields["stdout"] == "logs/sweep/wobl_%j.out"
    assert fields["stderr"] == "logs/sweep/wobl_%j.err"
    assert fields["requeue"] is True
    assert fields["resources"] == {"tasks": 64, "gpus": 1, "gpu_types": ["a40"], "mem": "512G", "time": "24:00:00"}
    assert extra == []
    assert stripped == ["-p batch", "--gpus=a40", "--ntasks-per-node=64", "--mem=512G", "-t 24:00:00", "--requeue",
                        "-J wobl", "-o logs/sweep/wobl_%j.out", "-e logs/sweep/wobl_%j.err"]
    assert warnings == []
    assert body.startswith("module load aocc/3.2.0;")
    assert body.endswith("wobl_exp.yaml ...\n")
    assert "#SBATCH" not in body and "#!/bin/bash" not in body
    assert body.count("\n") == 4


def test_parse_bridges_script_fields():
    fields, extra, body, stripped, warnings = parse_sbatch(BRIDGES_SCRIPT)
    assert fields["partition"] == "GPU-shared" and fields["account"] == "mch250030p" and fields["qos"] == "gpu"
    assert fields["resources"] == {"cpus": 8, "gpus": 1, "gpu_types": ["h100-80"], "mem": "80G", "time": "04:00:00"}
    assert "name" not in fields and "stdout" not in fields
    assert extra == [] and warnings == []
    assert len(stripped) == 7
    assert body.startswith("set -euo pipefail\n")


def test_parse_crlf_input_normalised_with_warning():
    crlf = TRACE_SCRIPT.replace("\n", "\r\n")
    p = parse_sbatch("\ufeff" + crlf)
    assert "crlf_normalized" in p.warnings
    assert "\r" not in p.body
    assert p.spec_fields["name"] == "wobl"


def test_parse_nul_refused():
    with pytest.raises(SlurmMcpError) as ei:
        parse_sbatch("#!/bin/bash\n#SBATCH -J x\necho \x00\n")
    assert ei.value.code == "E_INVALID_SPEC"


def test_parse_gpus_per_job_divided_by_nodes():
    p = parse_sbatch("#!/bin/bash\n#SBATCH -N 2\n#SBATCH --gpus=v100-32:4\nsrun x\n")
    assert p.spec_fields["resources"] == {"nodes": 2, "gpus": 2, "gpu_types": ["v100-32"]}
    p = parse_sbatch("#!/bin/bash\n#SBATCH --gpus=2\nsrun x\n")
    assert p.spec_fields["resources"] == {"gpus": 2}


def test_parse_gpus_not_divisible_refused():
    with pytest.raises(SlurmMcpError) as ei:
        parse_sbatch("#!/bin/bash\n#SBATCH -N 2\n#SBATCH --gpus=3\nsrun x\n")
    assert ei.value.code == "E_INVALID_SPEC"
    assert "divisible" in str(ei.value)


@pytest.mark.parametrize("directive, expected", [
    ("--gpus-per-node=v100-32:2", {"gpus": 2, "gpu_types": ["v100-32"]}),
    ("--gpus-per-node=2", {"gpus": 2}),
    ("--gres=gpu:2", {"gpus": 2}),
    ("--gres=gpu", {"gpus": 1}),
    ("--gres=gpu:a40", {"gpus": 1, "gpu_types": ["a40"]}),
    ("--gres gpu:h100-80:1", {"gpus": 1, "gpu_types": ["h100-80"]}),
])
def test_parse_gpu_forms(directive: str, expected: dict):
    p = parse_sbatch(f"#!/bin/bash\n#SBATCH {directive}\nsrun x\n")
    assert p.spec_fields["resources"] == expected
    assert p.stripped_directives == [directive]


def test_parse_gres_non_gpu_entries_kept_in_extra():
    p = parse_sbatch("#!/bin/bash\n#SBATCH --gres=gpu:a40:1,nvme:1\nsrun x\n")
    assert p.spec_fields["resources"] == {"gpus": 1, "gpu_types": ["a40"]}
    assert p.extra_sbatch == ["--gres=nvme:1"]
    assert any("nvme:1" in w for w in p.warnings)


def test_parse_mem_variants():
    p = parse_sbatch("#!/bin/bash\n#SBATCH --mem-per-cpu=2G\n#SBATCH -c 4\nsrun x\n")
    assert p.spec_fields["resources"]["mem"] == "8G" and p.spec_fields["resources"]["cpus"] == 4
    assert any("mem-per-cpu" in w for w in p.warnings)
    p = parse_sbatch("#!/bin/bash\n#SBATCH --mem-per-gpu=40G\n#SBATCH --gres=gpu:2\nsrun x\n")
    assert p.spec_fields["resources"]["mem"] == "80G"
    p = parse_sbatch("#!/bin/bash\n#SBATCH --mem 512000M\nsrun x\n")
    assert p.spec_fields["resources"]["mem"] == "512000M"


def test_parse_output_error_chdir_mapping():
    p = parse_sbatch("#!/bin/bash\n#SBATCH -o out/%x_%j.out\n#SBATCH --error=err/%j.err\n#SBATCH -D /work/here\nx\n")
    assert p.spec_fields["stdout"] == "out/%x_%j.out"
    assert p.spec_fields["stderr"] == "err/%j.err"
    assert p.spec_fields["workdir"] == "/work/here"


def test_parse_dependency_handles_accepted():
    p = parse_sbatch("#!/bin/bash\n#SBATCH -d afterok:j12:j13,afterany:j7\n#SBATCH --kill-on-invalid-dep=yes\nx\n")
    assert p.spec_fields["depends_on"] == ["afterok:j12", "afterok:j13", "afterany:j7"]
    assert "--kill-on-invalid-dep=yes" in p.stripped_directives
    p = parse_sbatch("#!/bin/bash\n#SBATCH --dependency=singleton\nx\n")
    assert p.spec_fields["depends_on"] == ["singleton"]


def test_parse_dependency_raw_slurm_id_refused():
    with pytest.raises(SlurmMcpError) as ei:
        parse_sbatch("#!/bin/bash\n#SBATCH -d afterok:615408\nx\n", cluster="trace")
    assert ei.value.code == "E_DEPENDENCY"
    assert "615408" in str(ei.value)
    assert "job_status(['trace:615408'])" in str(ei.value)


def test_parse_unknown_directives_kept_verbatim():
    src = ("#!/bin/bash\n#SBATCH --time-min=01:00:00\n#SBATCH --nice=10\n#SBATCH --licenses=foo:1\n"
           "#SBATCH --tmp=10G\n#SBATCH -p batch --nice=5\n#SBATCH --begin now+1hour\nx\n")
    p = parse_sbatch(src)
    assert p.extra_sbatch == ["--time-min=01:00:00", "--nice=10", "--licenses=foo:1", "--tmp=10G", "--nice=5",
                              "--begin now+1hour"]
    assert p.stripped_directives == ["-p batch"]
    assert p.spec_fields["partition"] == "batch"


def test_parse_trailing_comment_and_short_forms():
    p = parse_sbatch("#!/bin/bash\n#SBATCH -t 1:00:00  # one hour\n#SBATCH -N2\n#SBATCH -n 8\n#SBATCH -c4\n"
                     "#SBATCH -C 'a40|l40s'\n#SBATCH --exclusive\nx\n")
    assert p.spec_fields["resources"] == {"nodes": 2, "time": "1:00:00", "cpus": 4, "tasks": 4,
                                          "constraint": "a40|l40s", "exclusive": True}


def test_parse_ntasks_not_divisible_refused():
    with pytest.raises(SlurmMcpError):
        parse_sbatch("#!/bin/bash\n#SBATCH -N 2\n#SBATCH -n 3\nx\n")


def test_parse_only_leading_block_is_parsed():
    src = "#!/bin/bash\n\n# a comment\n#SBATCH -J early\n\necho start\n#SBATCH -p late\necho end\n"
    p = parse_sbatch(src)
    assert p.spec_fields == {"name": "early"}
    assert p.body == "echo start\n#SBATCH -p late\necho end\n"


def test_parse_signal_directive():
    p = parse_sbatch("#!/bin/bash\n#SBATCH --signal=B:USR1@300\nx\n")
    assert p.spec_fields == {"grace_s": 300}
    assert any("stripped" in w for w in p.warnings)
    p = parse_sbatch("#!/bin/bash\n#SBATCH --signal=USR1@60\nx\n")
    assert p.spec_fields == {"grace_s": 60, "child_signal": "USR1"}


def test_parse_array_requeue_flags():
    p = parse_sbatch("#!/bin/bash\n#SBATCH -a 0-9%4\n#SBATCH --no-requeue\nx\n")
    assert p.spec_fields == {"array": "0-9", "array_parallel": 4, "requeue": False}


def test_parse_stripped_with_warning_only_directives():
    p = parse_sbatch("#!/bin/bash\n#SBATCH --mail-type=END\n#SBATCH --mail-user=a@b\n#SBATCH --comment=x\n"
                     "#SBATCH -x node1\n#SBATCH -H\n#SBATCH --export=NONE\n#SBATCH --open-mode=truncate\nx\n")
    assert p.spec_fields == {}
    assert len(p.stripped_directives) == 7
    joined = " ".join(p.warnings)
    for word in ("mail", "comment", "exclude", "hold", "export", "open-mode"):
        assert word in joined


def test_parse_repeated_directive_last_wins_with_warning():
    p = parse_sbatch("#!/bin/bash\n#SBATCH -p a\n#SBATCH -p b\nx\n")
    assert p.spec_fields["partition"] == "b"
    assert any("repeats" in w for w in p.warnings)


def test_parse_empty_body_refused():
    with pytest.raises(SlurmMcpError) as ei:
        parse_sbatch("#!/bin/bash\n#SBATCH -J x\n\n")
    assert ei.value.code == "E_SCRIPT"


def test_parse_bad_time_refused():
    with pytest.raises(SlurmMcpError) as ei:
        parse_sbatch("#!/bin/bash\n#SBATCH -t forever\nx\n")
    assert ei.value.code == "E_INVALID_SPEC"


# =================================================================================================
# merge_spec
# =================================================================================================

def test_merge_script_fields_fill_missing_spec_fields():
    parsed = parse_sbatch(TRACE_SCRIPT)
    spec, warnings = merge_spec({"name": "wobl", "script": TRACE_SCRIPT}, parsed)
    assert spec.partition == "batch" and spec.requeue is True
    assert spec.resources.time == "24:00:00" and spec.resources.gpus == 1 and spec.resources.gpu_types == ["a40"]
    assert spec.resources.tasks == 64 and spec.resources.mem == "512G"
    assert spec.stdout == "logs/sweep/wobl_%j.out"
    assert warnings == []


def test_merge_explicit_spec_wins_with_warning():
    parsed = parse_sbatch(TRACE_SCRIPT)
    spec, warnings = merge_spec({"name": "wobl", "script": TRACE_SCRIPT, "partition": "biosimmlab",
                                 "resources": {"time": "02:00:00", "mem": "64G"}}, parsed)
    assert spec.partition == "biosimmlab"
    assert spec.resources.time == "02:00:00" and spec.resources.mem == "64G"
    assert spec.resources.gpus == 1 and spec.resources.tasks == 64          # unset fields still come from the script
    assert any("spec.partition='biosimmlab' overrides script value 'batch'" in w for w in warnings)
    assert any("spec.resources.time='02:00:00' overrides script value '24:00:00'" in w for w in warnings)
    assert any("spec.resources.mem=" in w for w in warnings)


def test_merge_accepts_validated_jobspec_and_extends_extra_sbatch():
    parsed = parse_sbatch("#!/bin/bash\n#SBATCH --nice=5\n#SBATCH -J fromscript\nx\n")
    spec = _spec(name="explicit", extra_sbatch=["--licenses=foo"])
    merged, warnings = merge_spec(spec, parsed)
    assert merged.name == "explicit"
    assert merged.extra_sbatch == ["--licenses=foo", "--nice=5"]
    assert any("overrides script value 'fromscript'" in w for w in warnings)


# =================================================================================================
# render_job_sbatch / env.sh / user_body.sh (6.3)
# =================================================================================================

def _trace_spec() -> JobSpec:
    spec, _ = merge_spec({"name": "wobl", "script": TRACE_SCRIPT}, parse_sbatch(TRACE_SCRIPT))
    return spec


def _bridges_spec() -> JobSpec:
    spec, _ = merge_spec({"name": "grpo_h100", "script": BRIDGES_SCRIPT}, parse_sbatch(BRIDGES_SCRIPT))
    return spec


def test_render_job_sbatch_trace_golden():
    ctrl_dir = "/trace/group/biosimmlab/wxu2/.slurm-mcp/jobs/j17/a1"
    text = render_job_sbatch(_trace_spec(), "j17", 1, "t-0123456789ab", ctrl_dir, TRACE_WORKDIR,
                             "/trace/group/biosimmlab/wxu2/.slurm-mcp", SHA8, rendered_at=STAMP)
    assert text == (
        "#!/bin/bash\n"
        "#SBATCH -J wobl\n"
        "#SBATCH -N 1\n"
        "#SBATCH --ntasks-per-node=64\n"
        "#SBATCH --open-mode=append\n"
        f"# slurm-mcp handle=j17 attempt=1 token=t-0123456789ab rendered={STAMP}\n"
        f"source {ctrl_dir}/env.sh\n"
        f"cd {TRACE_WORKDIR}\n"
        f"exec /trace/group/biosimmlab/wxu2/.slurm-mcp/bin/{SHA8}/wrap.sh {ctrl_dir} -- bash {ctrl_dir}/user_body.sh\n"
    )
    assert _forbidden_lines(text) == []


def test_render_job_sbatch_bridges_golden():
    ctrl_dir = "/ocean/projects/mch250030p/wxu7/.slurm-mcp/jobs/j18/a1"
    text = render_job_sbatch(_bridges_spec(), "j18", 1, "t-abcdef012345", ctrl_dir, BRIDGES_WORKDIR,
                             "/ocean/projects/mch250030p/wxu7/.slurm-mcp", SHA8, rendered_at=STAMP)
    lines = text.splitlines()
    assert lines[:5] == ["#!/bin/bash", "#SBATCH -J grpo_h100", "#SBATCH -N 1", "#SBATCH --cpus-per-task=8",
                         "#SBATCH --open-mode=append"]
    assert _forbidden_lines(text) == []
    assert "--cpus-per-task=8" in text and "--gres" not in text and "--qos" not in text and "80G" not in text


def test_render_job_sbatch_extra_and_optional_header_lines():
    spec = _spec(resources={"time": "1:00:00", "nodes": 2, "tasks": 4, "cpus": 8, "exclusive": True,
                            "constraint": "a40"}, extra_sbatch=["--nice=5", "#SBATCH --time-min=00:30:00"])
    text = render_job_sbatch(spec, "j1", 2, "t-x", "/c/j1/a2", "/w", "/c", SHA8, rendered_at=STAMP)
    assert text.splitlines()[:10] == [
        "#!/bin/bash", "#SBATCH -J t", "#SBATCH -N 2", "#SBATCH --ntasks-per-node=4", "#SBATCH --cpus-per-task=8",
        "#SBATCH --exclusive", "#SBATCH -C a40", "#SBATCH --open-mode=append", "#SBATCH --nice=5",
        "#SBATCH --time-min=00:30:00",
    ]
    assert "# slurm-mcp handle=j1 attempt=2 token=t-x rendered=" in text


def test_render_job_sbatch_unwrapped_and_quoting():
    spec = _spec(wrap=False)
    text = render_job_sbatch(spec, "j1", 1, "t", "/c d/j1/a1", "/w d", "/c d", SHA8, rendered_at=STAMP)
    assert text.endswith("cd '/w d'\nexec bash '/c d/j1/a1/user_body.sh'\n")
    assert "wrap.sh" not in text
    assert text.count("\r") == 0


def test_render_job_sbatch_default_timestamp_is_iso():
    text = render_job_sbatch(_spec(), "j1", 1, "t", "/c", "/w", "/c", SHA8)
    m = re.search(r"rendered=(\S+)", text)
    assert m and re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", m.group(1))


def test_render_env_sh_defaults_and_child_signal_only_when_set():
    spec = _spec(resources={"time": "02:00:00"}, env={"A": "1", "B": "two words"}, modules=["cuda/12.8"],
                 setup="source ~/.bashrc\nconda activate pyg")
    text = render_env_sh(spec)
    assert text.splitlines()[1:] == [
        "export SLURM_MCP_GRACE=120", "export SLURM_MCP_ON_TIMEOUT=fail", "export SLURM_MCP_MAX_RESTARTS=3",
        "export SLURM_MCP_TIMELIMIT_S=7200", "export A=1", "export B='two words'", "module load cuda/12.8",
        "source ~/.bashrc", "conda activate pyg",
    ]
    assert "SLURM_MCP_CHILD_SIGNAL" not in text


def test_render_env_sh_explicit_arguments():
    text = render_env_sh(_spec(), grace_s=300, on_timeout="requeue", max_restarts=5, child_signal="USR1",
                         timelimit_s=3600, modules=["m1"], setup="echo setup", env={"X": "y"})
    assert text.splitlines()[1:] == [
        "export SLURM_MCP_GRACE=300", "export SLURM_MCP_ON_TIMEOUT=requeue", "export SLURM_MCP_MAX_RESTARTS=5",
        "export SLURM_MCP_CHILD_SIGNAL=USR1", "export SLURM_MCP_TIMELIMIT_S=3600", "export X=y", "module load m1",
        "echo setup",
    ]


def test_render_env_sh_child_signal_from_spec():
    spec = _spec(child_signal="usr1", checkpoint_interval_h=1.0, on_timeout="requeue")
    text = render_env_sh(spec)
    assert "export SLURM_MCP_CHILD_SIGNAL=USR1\n" in text and "export SLURM_MCP_ON_TIMEOUT=requeue\n" in text


def test_render_user_body_command_and_script_modes():
    assert render_user_body(_spec(command="echo a\r\necho b")) == "echo a\necho b\n"
    assert render_user_body(_trace_spec()) == parse_sbatch(TRACE_SCRIPT).body
    spec = _spec(command=None, script_path="/remote/x.sh")
    assert render_user_body(spec, script_text=BRIDGES_SCRIPT) == parse_sbatch(BRIDGES_SCRIPT).body
    with pytest.raises(SlurmMcpError):
        render_user_body(spec)


# =================================================================================================
# requeue_flag (5.3) and choose_qos (6.1)
# =================================================================================================

@pytest.mark.parametrize("requeue, on_timeout, pcaps, ccaps, expected", [
    (True, "fail", {}, {}, ["--requeue", "--open-mode=append"]),
    (False, "fail", {"preempt_mode": ["REQUEUE"]}, {}, ["--no-requeue"]),
    (None, "fail", TRACE_CAPS["partitions"]["batch"], TRACE_CAPS, ["--requeue", "--open-mode=append"]),
    (None, "fail", TRACE_CAPS["partitions"]["cpuonly"], TRACE_CAPS, ["--requeue", "--open-mode=append"]),
    (None, "fail", {"preempt_mode": "GANG,REQUEUE"}, {}, ["--requeue", "--open-mode=append"]),
    (None, "fail", TRACE_CAPS["partitions"]["biosimmlab"], TRACE_CAPS, []),
    (None, "requeue", TRACE_CAPS["partitions"]["biosimmlab"], TRACE_CAPS, ["--requeue", "--open-mode=append"]),
    (None, "fail", BRIDGES_CAPS["partitions"]["GPU-shared"], BRIDGES_CAPS, ["--no-requeue"]),
    (None, "fail", {"preempt_mode": ["OFF"]}, {"job_requeue": True, "charges": False}, []),
    (None, "fail", {"preempt_mode": ["OFF"]}, {"job_requeue": False, "charges": True}, []),
    (None, "fail", {"PreemptMode": "OFF"}, {"JobRequeue": "1", "su_rates": {"gpu": 1.0}}, ["--no-requeue"]),
    (None, "fail", None, SimpleNamespace(job_requeue=True, charges=True), ["--no-requeue"]),
])
def test_requeue_flag_rule(requeue, on_timeout, pcaps, ccaps, expected):
    kw = {"requeue": requeue, "on_timeout": on_timeout}
    if on_timeout == "requeue":
        kw.update(child_signal="USR1", checkpoint_interval_h=0.5)
    assert requeue_flag(_spec(**kw), pcaps, ccaps) == expected


def test_requeueable_and_cluster_charges():
    assert requeueable(_spec(), TRACE_CAPS["partitions"]["batch"], TRACE_CAPS) is True
    assert requeueable(_spec(), TRACE_CAPS["partitions"]["biosimmlab"], TRACE_CAPS) is False
    assert requeueable(_spec(), BRIDGES_CAPS["partitions"]["GPU-shared"], BRIDGES_CAPS) is False
    assert requeueable(_spec(requeue=True), BRIDGES_CAPS["partitions"]["GPU-shared"], BRIDGES_CAPS) is True
    assert requeueable(_spec(), {"preempt_mode": ["OFF"]}, {"job_requeue": True, "charges": False}) is True
    assert cluster_charges(BRIDGES_CAPS) is True and cluster_charges(TRACE_CAPS) is False
    assert cluster_charges({"su_rates": {"gpu": 2.0}}) is True and cluster_charges({}) is False


def test_partition_family():
    assert partition_family("GPU-shared") == "gpu" and partition_family("RM-shared") == "rm"
    assert partition_family("EM") == "em" and partition_family("batch") == "batch" and partition_family("") == ""


def test_choose_qos_examples_from_design():
    assert choose_qos(_spec(), TRACE_PROFILE, TRACE_CAPS["partitions"]["batch"], TRACE_CAPS["assoc"]) == ["normal"]
    assert choose_qos(_spec(), BRIDGES_PROFILE, BRIDGES_CAPS["partitions"]["GPU-shared"], BRIDGES_CAPS["assoc"]) \
        == ["gpu", "low", "push", "unlimited"]
    assert choose_qos(_spec(), BRIDGES_PROFILE, BRIDGES_CAPS["partitions"]["RM-shared"], BRIDGES_CAPS["assoc"]) \
        == ["low", "push", "unlimited"]


def test_choose_qos_precedence_spec_then_profile_map():
    assert choose_qos(_spec(qos="mine"), BRIDGES_PROFILE, BRIDGES_CAPS["partitions"]["GPU-shared"],
                      BRIDGES_CAPS["assoc"]) == ["mine"]
    profile = ClusterProfile(name="b", host="h", user="u", qos_map={"GPU-shared": "push"})
    assert choose_qos(_spec(), profile, BRIDGES_CAPS["partitions"]["GPU-shared"], BRIDGES_CAPS["assoc"]) == ["push"]
    assert choose_qos(_spec(), {"qos_map": {"GPU-shared": "ft"}}, {"name": "GPU-shared"}, None) == ["ft"]


def test_choose_qos_allow_all_with_assoc_default_means_no_qos():
    assoc = {"default_qos": "normal", "qos_list": ["normal", "batchpartition"]}
    assert choose_qos(_spec(), TRACE_PROFILE, TRACE_CAPS["partitions"]["cpuonly"], assoc) == []
    assert choose_qos(_spec(), TRACE_PROFILE, {"name": "p", "allow_qos": []}, assoc) == []
    # AllowQos=ALL without a default: the assoc list, default-first ordering not applicable, interact excluded
    assoc2 = {"default_qos": None, "qos_list": ["gpuinteract", "low", "gpu", "cpuonly-debug-qos"]}
    assert choose_qos(_spec(), TRACE_PROFILE, TRACE_CAPS["partitions"]["cpuonly"], assoc2) \
        == ["cpuonly-debug-qos", "low", "gpu"]


def test_choose_qos_ordering_default_family_low_rest():
    pcaps = {"name": "GPU-shared", "allow_qos": ["unlimited", "low", "gpuinteract", "gpu", "ft", "push"]}
    assoc = {"default_qos": "push", "qos_list": ["gpu", "low", "push", "ft", "unlimited", "gpuinteract"]}
    assert choose_qos(_spec(), {}, pcaps, assoc) == ["push", "gpu", "low", "unlimited", "ft"]
    # no assoc at all: AllowQos is the pool
    assert choose_qos(_spec(), {}, pcaps, None) == ["gpu", "low", "unlimited", "ft", "push"]


# =================================================================================================
# output patterns (6.3)
# =================================================================================================

def test_resolve_output_patterns_relative_absolute_default_array():
    spec = _spec(stdout="logs/%x_%j.out", stderr="/abs/err_%j.txt")
    assert resolve_output_patterns(spec, "/w", "/c/jobs/j1") == ("/w/logs/%x_%j.out", "/abs/err_%j.txt")
    assert resolve_output_patterns(_spec(), "/w", "/c/jobs/j1/") == ("/c/jobs/j1/out/slurm-%j.out",
                                                                     "/c/jobs/j1/out/slurm-%j.out")
    assert resolve_output_patterns(_spec(array="0-3"), "/w", "/c/jobs/j1") == ("/c/jobs/j1/out/slurm-%A_%a.out",
                                                                               "/c/jobs/j1/out/slurm-%A_%a.out")
    assert resolve_output_patterns(_spec(), "/w", "/c", is_array=True)[0].endswith("slurm-%A_%a.out")
    assert resolve_output_patterns(_spec(stdout="o.txt"), "/w", "/c") == ("/w/o.txt", "/w/o.txt")
    assert resolve_output_patterns(_spec(stdout="../o.txt"), "/w/sub", "/c")[0] == "/w/o.txt"


@pytest.mark.parametrize("pattern, expected", [
    ("slurm-%j.out", "slurm-615411.out"),
    ("%J.%x.%u", "615411.wobl.wxu2"),
    ("logs/%x_%j.out", "logs/wobl_615411.out"),
    ("100%%_%j", "100%_615411"),
    ("%8j.out", "00615411.out"),
    ("plain.out", "plain.out"),
    ("%N.out", None), ("%n.out", None), ("%t.out", None), ("%s.out", None), ("%x_%N_%j", None),
])
def test_expand_pattern(pattern, expected):
    assert expand_pattern(pattern, 615411, "wobl", "wxu2") == expected
    assert pattern_needs_controller(pattern) is (expected is None)


def test_expand_pattern_array_fields():
    assert expand_pattern("slurm-%A_%a.out", "615500", "arr", "u", array_index=7) == "slurm-615500_7.out"
    assert expand_pattern("%A_%3a", 615500, "arr", "u", array_index="7") == "615500_007"
    assert expand_pattern("x_%a", 1, "n", "u") == f"x_{NO_VAL_ARRAY_INDEX}"


# =================================================================================================
# target_args (6.3) golden
# =================================================================================================

def test_target_args_trace_golden():
    spec = _trace_spec()
    ctrl_root = "/trace/group/biosimmlab/wxu2/.slurm-mcp/jobs/j17"
    args = target_args("trace:batch:a40", spec, TRACE_PROFILE, TRACE_CAPS, 1, "j17", "t-0123456789ab",
                       workdir=TRACE_WORKDIR, ctrl_root=ctrl_root)
    assert args == [
        "-p", "batch", "--qos=normal", "-A", "biosimmlab", "-t", "24:00:00", "--mem=512G", "--gres=gpu:a40:1",
        "--requeue", "--open-mode=append", "--signal=B:USR1@120",
        "-o", f"{TRACE_WORKDIR}/logs/sweep/wobl_%j.out", "-e", f"{TRACE_WORKDIR}/logs/sweep/wobl_%j.err",
        "--comment=slurm-mcp:j17:1:t-0123456789ab", "--parsable",
    ]
    assert format_submit_line(args).startswith("-p batch --qos=normal -A biosimmlab -t 24:00:00 --mem=512G")


def test_target_args_bridges_golden():
    spec = _bridges_spec()
    ctrl_root = "/ocean/projects/mch250030p/wxu7/.slurm-mcp/jobs/j18"
    r = build_target_args("bridges2:GPU-shared:h100-80", spec, BRIDGES_PROFILE, BRIDGES_CAPS, 1, "j18", "t-abc",
                          workdir=BRIDGES_WORKDIR, ctrl_root=ctrl_root)
    assert isinstance(r, RenderedArgs)
    assert r.args == [
        "-p", "GPU-shared", "--qos=gpu", "-A", "mch250030p", "-t", "04:00:00", "--mem=80G", "--gres=gpu:h100-80:1",
        "--no-requeue", "--signal=B:USR1@120",
        "-o", f"{ctrl_root}/out/slurm-%j.out", "-e", f"{ctrl_root}/out/slurm-%j.out",
        "--comment=slurm-mcp:j18:1:t-abc", "--parsable",
    ]
    assert r.qos == "gpu" and r.account == "mch250030p" and r.requeueable is False and r.warnings == []
    assert r.stdout_pattern == f"{ctrl_root}/out/slurm-%j.out"
    assert "--qos=gpu" not in r.injected and "-A mch250030p" not in r.injected       # both came from the script
    assert "--no-requeue" in r.injected and "--signal=B:USR1@120" in r.injected and "--parsable" in r.injected
    assert r.submit_line == format_submit_line(r.args)


def test_target_args_mem_dropped_on_no_mem_flag_partition():
    spec = _spec(resources={"time": "01:00:00", "mem": "16G", "cpus": 4})
    r = build_target_args("bridges2:RM-shared", spec, BRIDGES_PROFILE, BRIDGES_CAPS, 1, "j2", "t",
                          workdir="/w", ctrl_root="/c")
    assert "--mem=16G" not in r.args and not any(a.startswith("--mem") for a in r.args)
    assert any("--mem=16G dropped" in w and "RM-shared" in w for w in r.warnings)
    assert r.args[:6] == ["-p", "RM-shared", "--qos=low", "-A", "mch250030p", "-t"]
    assert "--gres" not in " ".join(r.args)


def test_target_args_requeue_partition_and_open_mode():
    spec = _spec()                                                    # requeue=None
    args = target_args("trace:batch", spec, TRACE_PROFILE, TRACE_CAPS, 1, "j3", "t", workdir="/w", ctrl_root="/c")
    i = args.index("--requeue")
    assert args[i + 1] == "--open-mode=append" and "--no-requeue" not in args
    args = target_args("trace:biosimmlab", spec, TRACE_PROFILE, TRACE_CAPS, 1, "j3", "t", workdir="/w", ctrl_root="/c")
    assert "--requeue" not in args and "--no-requeue" not in args and "--open-mode=append" not in args


def test_target_args_full_order_with_every_option():
    spec = _spec(resources={"time": "01:00:00", "gpus": 2, "gpu_types": ["v100-32"]}, array="0-9", array_parallel=4,
                 requeue=False, account="acct", qos="myqos")
    r = build_target_args("bridges2:GPU-shared,GPU-small:v100-32", spec, BRIDGES_PROFILE, BRIDGES_CAPS, 3, "j9",
                          "t-9", notify_email="me@example.org", excluded_nodes=["v012", "v013"], hold=True,
                          dependency="afterok:615408", workdir="/w", ctrl_root="/c")
    assert r.args == [
        "-p", "GPU-shared,GPU-small", "--qos=myqos", "-A", "acct", "-t", "01:00:00", "--gres=gpu:v100-32:2",
        "--no-requeue", "--signal=B:USR1@120", "-o", "/c/out/slurm-%A_%a.out", "-e", "/c/out/slurm-%A_%a.out",
        "--array=0-9%4", "--dependency=afterok:615408", "--kill-on-invalid-dep=yes",
        f"--mail-type={MAIL_TYPES}", "--mail-user=me@example.org", "--exclude=v012,v013", "--hold",
        "--comment=slurm-mcp:j9:3:t-9", "--parsable",
    ]
    assert MAIL_TYPES == "END,FAIL,REQUEUE,TIME_LIMIT_90"
    assert "--kill-on-invalid-dep=yes" in r.injected and "--exclude=v012,v013" in r.injected
    assert strip_for_test_only(r.args) == [a for a in r.args
                                           if a not in ("--parsable", "--hold") and not a.startswith("--comment=")]
    assert "--hold" not in strip_for_test_only(r.args) and "--parsable" not in strip_for_test_only(r.args)


def test_target_args_singleton_dependency_and_target_qos_override():
    spec = _spec(depends_on=["singleton"])
    args = target_args(Target.parse("bridges2:GPU-shared:h100-80@low"), spec, BRIDGES_PROFILE, BRIDGES_CAPS, 1,
                       "j4", "t", workdir="/w", ctrl_root="/c")
    assert "--qos=low" in args and "--dependency=singleton" in args and "--kill-on-invalid-dep=yes" in args


def test_target_args_unwrapped_has_no_signal():
    args = target_args("trace:batch", _spec(wrap=False), TRACE_PROFILE, TRACE_CAPS, 1, "j5", "t", workdir="/w",
                       ctrl_root="/c")
    assert not any(a.startswith("--signal") for a in args)


def test_target_args_cached_qos_and_profile_account_precedence():
    caps = dict(TRACE_CAPS, qos_for_partition={"batch": "batchpartition"})
    profile = ClusterProfile(name="trace", host="h", user="u", default_account="labacct")
    r = build_target_args("trace:batch", _spec(), profile, caps, 1, "j6", "t", workdir="/w", ctrl_root="/c")
    assert "--qos=batchpartition" in r.args and r.account == "labacct" and "-A labacct" in r.injected
    r = build_target_args("trace:batch", _spec(), profile, caps, 1, "j6", "t", workdir="/w", ctrl_root="/c",
                          qos="explicit")
    assert r.qos == "explicit"
    # AllowQos=ALL without an assoc default QOS (the TRACE fixture): the assoc list, partition family first
    r = build_target_args("trace:cpuonly", _spec(), TRACE_PROFILE, TRACE_CAPS, 1, "j6", "t", workdir="/w",
                          ctrl_root="/c")
    assert "--qos=cpuonly-debug-qos" in r.args
    # AllowQos=ALL with an assoc default QOS: no --qos at all
    caps_default = dict(TRACE_CAPS, assoc=dict(TRACE_CAPS["assoc"], default_qos="normal"))
    r = build_target_args("trace:cpuonly", _spec(), TRACE_PROFILE, caps_default, 1, "j6", "t", workdir="/w",
                          ctrl_root="/c")
    assert not any(a.startswith("--qos") for a in r.args) and r.qos is None


def test_target_args_untyped_gres_warns_and_explicit_patterns():
    spec = _spec(resources={"time": "01:00:00", "gpus": 1})
    r = build_target_args("trace:batch", spec, TRACE_PROFILE, TRACE_CAPS, 1, "j7", "t", stdout_pattern="/o/%j.out",
                          stderr_pattern="/o/%j.err")
    assert "--gres=gpu:1" in r.args and any("untyped" in w for w in r.warnings)
    assert r.args[r.args.index("-o") + 1] == "/o/%j.out" and r.args[r.args.index("-e") + 1] == "/o/%j.err"
    with pytest.raises(ValueError):
        build_target_args("trace:batch", spec, TRACE_PROFILE, TRACE_CAPS, 1, "j7", "t")


def test_target_args_alloc_mode():
    r = build_target_args("trace:batch", _spec(), TRACE_PROFILE, TRACE_CAPS, 1, "a3", "t", workdir="/w",
                          ctrl_root="/c", mode="alloc")
    assert list(ALLOC_CLI_ARGS) == ["--signal=B:TERM@60", "--no-requeue"]
    i = r.args.index("--signal=B:TERM@60")
    assert r.args[i + 1] == "--no-requeue" and "--requeue" not in r.args and "--signal=B:USR1@120" not in r.args
    assert r.requeueable is False


def test_format_submit_line_quotes():
    assert format_submit_line(["-p", "batch", "-C", "a40|l40s"]) == "-p batch -C 'a40|l40s'"


# =================================================================================================
# alloc.sbatch (6.3 last paragraph)
# =================================================================================================

def test_render_alloc_sbatch():
    spec = _spec(resources={"time": "04:00:00", "cpus": 4, "gpus": 1, "gpu_types": ["a40"]})
    text = render_alloc_sbatch(spec, "a3", 1, "t-alloc", "/c/allocs/a3/a1", "/w", "/c", SHA8, idle_release_s=600,
                               rendered_at=STAMP)
    assert text == (
        "#!/bin/bash\n"
        "#SBATCH -J alloc-a3\n"
        "#SBATCH -N 1\n"
        "#SBATCH --cpus-per-task=4\n"
        "#SBATCH -t 04:00:00\n"
        "#SBATCH --open-mode=append\n"
        f"# slurm-mcp handle=a3 attempt=1 token=t-alloc rendered={STAMP}\n"
        "source /c/allocs/a3/a1/env.sh\n"
        "cd /w\n"
        f"exec /c/bin/{SHA8}/alloc-agent.sh /c/allocs/a3/a1 600\n"
    )
    assert "--gres" not in text and "wrap.sh" not in text


def test_module_exports_everything_public():
    for name in render.__all__:
        assert hasattr(render, name), name
