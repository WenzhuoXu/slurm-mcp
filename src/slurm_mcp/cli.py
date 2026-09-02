"""Command line entry point: profile management, credential entry, connectivity test, ad-hoc exec, serve."""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import shlex
import sys

from . import credentials
from .config import AUTH_METHODS, CONFIG_PATH, ClusterProfile, get_profile, load_profiles, save_profiles


def _cmd_cluster_add(a):
    profiles = load_profiles()
    if a.name in profiles and not a.force:
        sys.exit(f"cluster {a.name!r} already exists (use --force to overwrite)")
    p = ClusterProfile(name=a.name, host=a.host, user=a.user, port=a.port, auth=a.auth,
                       key_path=a.key, data_host=a.data_host, remote_root=a.remote_root,
                       default_account=a.account, default_partition=a.partition)
    p.validate()
    profiles[a.name] = p
    save_profiles(profiles)
    print(f"saved cluster {a.name!r} -> {CONFIG_PATH}")
    if p.auth == "password" and not credentials.has_password(p):
        print(f"next: slurm-mcp auth set {a.name}")


def _cmd_cluster_list(a):
    profiles = load_profiles()
    if not profiles:
        print("no clusters configured; add one with: slurm-mcp cluster add NAME --host H --user U")
        return
    for name, p in sorted(profiles.items()):
        if p.auth == "password":
            cred = "password stored" if credentials.has_password(p) else "NO PASSWORD STORED"
        else:
            cred = p.auth
        extra = f"  data_host={p.data_host}" if p.data_host else ""
        print(f"{name:12s} {p.user}@{p.host}:{p.port}  auth={p.auth} [{cred}]{extra}")


def _cmd_cluster_remove(a):
    profiles = load_profiles()
    p = profiles.pop(a.name, None)
    if p is None:
        sys.exit(f"unknown cluster {a.name!r}")
    save_profiles(profiles)
    if p.auth == "password":
        credentials.delete_password(p)
    print(f"removed {a.name!r}")


def _cmd_auth_set(a):
    p = get_profile(a.name)
    if not sys.stdin.isatty():
        sys.exit("refusing to read a password from a non-interactive stdin; run this in your own terminal")
    pw = getpass.getpass(f"Password for {p.credential_id}: ")
    if not pw:
        sys.exit("empty password; nothing stored")
    credentials.set_password(p, pw)
    print(f"stored in {credentials.backend_name()} under service 'slurm-mcp', id '{p.credential_id}'")


def _cmd_auth_clear(a):
    p = get_profile(a.name)
    print("cleared" if credentials.delete_password(p) else "nothing stored")


def _cmd_auth_status(a):
    print(f"keyring backend: {credentials.backend_name()}")
    for name, p in sorted(load_profiles().items()):
        if p.auth == "password":
            print(f"{name:12s} {'stored' if credentials.has_password(p) else 'MISSING'}")


async def _test(name: str) -> int:
    from .transport import SSHTransport

    p = get_profile(name)
    async with SSHTransport(p) as t:
        r = await t.run("echo OK; hostname -f; whoami; sinfo --version", timeout=60)
        print(r.stdout.strip())
        if r.stderr.strip():
            print(r.stderr.strip(), file=sys.stderr)
        return r.returncode


def _cmd_test(a):
    sys.exit(asyncio.run(_test(a.name)))


async def _exec(name: str, cmd: str, timeout: float, as_json: bool) -> int:
    from .transport import SSHTransport

    p = get_profile(name)
    async with SSHTransport(p) as t:
        r = await t.run(cmd, timeout=timeout)
        if as_json:
            print(json.dumps({"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr,
                              "seconds": r.seconds}))
        else:
            sys.stdout.write(r.stdout)
            if r.stderr:
                sys.stderr.write(r.stderr)
        return r.returncode


def _cmd_exec(a):
    cmd = " ".join(shlex.quote(c) for c in a.cmd) if a.quote else " ".join(a.cmd)
    sys.exit(asyncio.run(_exec(a.name, cmd, a.timeout, a.json)))


def _cmd_serve(a):
    from .server import main as serve_main  # lazy: server imports mcp

    argv = []
    if a.fake:
        argv.append("--fake")       # phase 3: in-process fake cluster; a documented no-op placeholder today
    if a.debug:
        argv.append("--debug")
    serve_main(argv)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slurm-mcp", description="SLURM-over-SSH MCP server and helper CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cluster", help="manage cluster profiles").add_subparsers(dest="sub", required=True)
    add = c.add_parser("add")
    add.add_argument("name")
    add.add_argument("--host", required=True)
    add.add_argument("--user", required=True)
    add.add_argument("--port", type=int, default=22)
    add.add_argument("--auth", choices=AUTH_METHODS, default="password")
    add.add_argument("--key", help="private key path (auth=key)")
    add.add_argument("--data-host", help="dedicated data transfer node hostname")
    add.add_argument("--remote-root", help="default remote directory for uploaded work")
    add.add_argument("--account", help="default SLURM account (-A)")
    add.add_argument("--partition", help="default SLURM partition (-p)")
    add.add_argument("--force", action="store_true")
    add.set_defaults(fn=_cmd_cluster_add)
    c.add_parser("list").set_defaults(fn=_cmd_cluster_list)
    rm = c.add_parser("remove")
    rm.add_argument("name")
    rm.set_defaults(fn=_cmd_cluster_remove)

    au = sub.add_parser("auth", help="store / clear passwords in the OS keyring")
    au = au.add_subparsers(dest="sub", required=True)
    s = au.add_parser("set", help="prompt for a password at the terminal and store it")
    s.add_argument("name")
    s.set_defaults(fn=_cmd_auth_set)
    cl = au.add_parser("clear")
    cl.add_argument("name")
    cl.set_defaults(fn=_cmd_auth_clear)
    au.add_parser("status").set_defaults(fn=_cmd_auth_status)

    t = sub.add_parser("test", help="connect and run a trivial command")
    t.add_argument("name")
    t.set_defaults(fn=_cmd_test)

    e = sub.add_parser("exec", help="run a shell command on a cluster: slurm-mcp exec NAME -- CMD...")
    e.add_argument("name")
    e.add_argument("--timeout", type=float, default=120)
    e.add_argument("--json", action="store_true")
    e.add_argument("--quote", action="store_true", help="shell-quote each argument instead of joining raw")
    e.add_argument("cmd", nargs=argparse.REMAINDER)
    e.set_defaults(fn=_cmd_exec)

    sv = sub.add_parser("serve", help="run the MCP server on stdio")
    sv.add_argument("--fake", action="store_true",
                    help="serve the in-process fake cluster for manual Claude Code sessions (phase 3; no-op today)")
    sv.add_argument("--debug", action="store_true", help="verbose logging to stderr")
    sv.set_defaults(fn=_cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "cmd", None) == "exec" and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    args.fn(args)


if __name__ == "__main__":
    main()
