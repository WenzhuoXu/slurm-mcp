"""Password storage in the OS keyring (Windows Credential Manager, macOS Keychain, Secret Service).

The password is entered by the human at a TTY via `slurm-mcp auth set <cluster>` and read back only
by the SSH transport at connect time. It is never written to config files or logs.

Headless override: environment variable SLURM_MCP_PASSWORD_<CLUSTERNAME> (upper-cased, dashes -> underscores).
"""
from __future__ import annotations

import os

from .config import ClusterProfile

SERVICE = "slurm-mcp"


def _env_name(profile: ClusterProfile) -> str:
    return "SLURM_MCP_PASSWORD_" + profile.name.upper().replace("-", "_")


def get_password(profile: ClusterProfile) -> str | None:
    env = os.environ.get(_env_name(profile))
    if env:
        return env
    import keyring  # imported lazily: keyring backends can be slow to initialise

    return keyring.get_password(SERVICE, profile.credential_id)


def set_password(profile: ClusterProfile, password: str) -> None:
    import keyring

    keyring.set_password(SERVICE, profile.credential_id, password)


def delete_password(profile: ClusterProfile) -> bool:
    import keyring
    from keyring.errors import PasswordDeleteError

    try:
        keyring.delete_password(SERVICE, profile.credential_id)
        return True
    except PasswordDeleteError:
        return False


def has_password(profile: ClusterProfile) -> bool:
    return get_password(profile) is not None


def backend_name() -> str:
    import keyring

    return type(keyring.get_keyring()).__name__
