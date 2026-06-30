from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Flag, auto


class Permission(Flag):
    """
    Coarse-grained capability flags used by both the Python sandbox
    and the bash runner.

    Set via environment variables:
        SANDBOX_ALLOW=READ,WRITE,EXECUTE,NETWORK
        SANDBOX_CONFIRM=DELETE,SYSTEM,INSTALL
    """

    NONE = 0

    # ── filesystem ────────────────────────────────────────────────────────────
    READ = auto()  # read files/dirs; safe path-inspection imports
    WRITE = auto()  # create/overwrite files; includes os, io, shutil (non-delete)
    DELETE = auto()  # rm, os.remove, shutil.rmtree — destructive

    # ── execution ─────────────────────────────────────────────────────────────
    EXECUTE = auto()  # subprocess, threading, asyncio; shell spawning (bash/sh)

    # ── networking ────────────────────────────────────────────────────────────
    NETWORK = auto()  # socket, urllib, requests, httpx, aiohttp …

    # ── elevated / system ─────────────────────────────────────────────────────
    SYSTEM = auto()  # sudo, systemctl, mount, chmod, chown, user management …
    INSTALL = auto()  # apt, pip, npm, snap, flatpak …


@dataclass
class SandboxPolicy:
    """
    Describes what a sandbox run is allowed to do.

    allow
        Permissions that are granted unconditionally.

    require_confirmation
        Permissions that are only granted when the caller explicitly
        passes ``confirmed=True``.  The runner returns a special
        ToolResult (``data["confirmation_required"] = True``) so the
        agent can surface the prompt to the user.

    Example
    -------
    ::

        policy = SandboxPolicy(
            allow={Permission.READ, Permission.WRITE, Permission.EXECUTE},
            require_confirmation={Permission.DELETE, Permission.SYSTEM, Permission.INSTALL},
        )
    """

    allow: set[Permission] = field(default_factory=set)
    require_confirmation: set[Permission] = field(default_factory=set)

    # ── query helpers ─────────────────────────────────────────────────────────

    def is_allowed(self, perm: Permission) -> bool:
        """Granted without confirmation."""
        return perm in self.allow and perm not in self.require_confirmation

    def needs_confirmation(self, perm: Permission) -> bool:
        """Granted only after explicit confirmation."""
        return perm in self.require_confirmation

    def is_blocked(self, perm: Permission) -> bool:
        """Neither allowed nor confirmation-gated — hard block."""
        return perm not in self.allow and perm not in self.require_confirmation

    def check(self, perm: Permission, confirmed: bool = False) -> str:
        """
        Returns one of ``"allow"``, ``"confirm"``, or ``"block"``.

        Parameters
        ----------
        perm:
            The permission to check.
        confirmed:
            Whether the caller has already obtained user confirmation.
        """
        if self.is_allowed(perm):
            return "allow"
        if self.needs_confirmation(perm):
            return "allow" if confirmed else "confirm"
        return "block"

    def has_any(self) -> bool:
        return bool(self.allow or self.require_confirmation)

    @classmethod
    def default(cls):
        """Default Permissions helper."""
        return cls(
            allow={
                Permission.READ,
                Permission.WRITE,
                Permission.EXECUTE,
                Permission.NETWORK,
            },
            require_confirmation={
                Permission.INSTALL,
                Permission.DELETE,
                Permission.SYSTEM,
            },
        )

    @classmethod
    def allow_all(cls):
        """Permissive Permissions helper."""
        return cls(
            allow={
                Permission.READ,
                Permission.WRITE,
                Permission.EXECUTE,
                Permission.NETWORK,
                Permission.SYSTEM,
                Permission.INSTALL,
                Permission.DELETE,
            }
        )


# ── env-var loader ────────────────────────────────────────────────────────────


def _parse_permissions(raw: str) -> set[Permission]:
    result: set[Permission] = set()
    for token in raw.split(","):
        name = token.strip().upper()
        if not name or name == "NONE":
            continue
        try:
            result.add(Permission[name])
        except KeyError:
            pass  # silently skip unknown names
    return result


def policy_from_env() -> SandboxPolicy:
    """
    Build a :class:`SandboxPolicy` from environment variables.

    ``SANDBOX_ALLOW``
        Comma-separated list of :class:`Permission` names that are
        granted without confirmation.
        Default: ``READ,WRITE,EXECUTE``

    ``SANDBOX_CONFIRM``
        Comma-separated list of :class:`Permission` names that require
        explicit confirmation before execution.
        Default: ``DELETE,SYSTEM,INSTALL``
    """
    allow_raw = os.environ.get("SANDBOX_ALLOW", "READ,WRITE,EXECUTE")
    confirm_raw = os.environ.get("SANDBOX_CONFIRM", "DELETE,SYSTEM,INSTALL")
    return SandboxPolicy(
        allow=_parse_permissions(allow_raw),
        require_confirmation=_parse_permissions(confirm_raw),
    )


# ── ready-made presets ────────────────────────────────────────────────────────

#: Medium-trust local agent: read/write/execute allowed; destructive ops need
#: confirmation; networking is off unless added explicitly.
DEFAULT_POLICY: SandboxPolicy = SandboxPolicy(
    allow={Permission.READ, Permission.WRITE, Permission.EXECUTE},
    require_confirmation={Permission.DELETE, Permission.SYSTEM, Permission.INSTALL},
)

#: Strict read-only policy — good for analysis/summarisation tasks.
READ_ONLY_POLICY: SandboxPolicy = SandboxPolicy(
    allow={Permission.READ},
    require_confirmation={Permission.WRITE, Permission.DELETE, Permission.EXECUTE},
)

#: Fully open policy — only use when you review every command yourself.
TRUSTED_POLICY: SandboxPolicy = SandboxPolicy(
    allow={
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.EXECUTE,
        Permission.NETWORK,
        Permission.SYSTEM,
        Permission.INSTALL,
    },
    require_confirmation=set(),
)
