from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from src.schema import ToolResult
from src.policy import Permission, SandboxPolicy, DEFAULT_POLICY


# ── command → permission mapping ──────────────────────────────────────────────
#
#  Each entry maps a command token to the Permission it requires.
#  Commands absent from this table are allowed as long as they don't hit a
#  forbidden path pattern.

_COMMAND_PERMISSIONS: dict[str, Permission] = {
    # ── SYSTEM: privilege escalation ──────────────────────────────────────────
    "sudo": Permission.SYSTEM,
    "su": Permission.SYSTEM,
    "doas": Permission.SYSTEM,
    # ── SYSTEM: power / service management ───────────────────────────────────
    "shutdown": Permission.SYSTEM,
    "reboot": Permission.SYSTEM,
    "poweroff": Permission.SYSTEM,
    "halt": Permission.SYSTEM,
    "systemctl": Permission.SYSTEM,
    "service": Permission.SYSTEM,
    "init": Permission.SYSTEM,
    # ── SYSTEM: disk / filesystem ─────────────────────────────────────────────
    "mount": Permission.SYSTEM,
    "umount": Permission.SYSTEM,
    "fdisk": Permission.SYSTEM,
    "parted": Permission.SYSTEM,
    "mkfs": Permission.SYSTEM,
    "fsck": Permission.SYSTEM,
    "dd": Permission.SYSTEM,
    # ── SYSTEM: user / group management ──────────────────────────────────────
    "useradd": Permission.SYSTEM,
    "usermod": Permission.SYSTEM,
    "userdel": Permission.SYSTEM,
    "passwd": Permission.SYSTEM,
    "groupadd": Permission.SYSTEM,
    "groupdel": Permission.SYSTEM,
    # ── SYSTEM: permissions / ownership ──────────────────────────────────────
    "chown": Permission.SYSTEM,
    "chmod": Permission.SYSTEM,
    "chattr": Permission.SYSTEM,
    # ── SYSTEM: firewall / network config ─────────────────────────────────────
    "iptables": Permission.SYSTEM,
    "ip6tables": Permission.SYSTEM,
    "ufw": Permission.SYSTEM,
    "firewall-cmd": Permission.SYSTEM,
    "nmcli": Permission.SYSTEM,
    "ifconfig": Permission.SYSTEM,
    "ip": Permission.SYSTEM,
    # ── INSTALL: package managers ─────────────────────────────────────────────
    "apt": Permission.INSTALL,
    "apt-get": Permission.INSTALL,
    "apt-cache": Permission.INSTALL,
    "yum": Permission.INSTALL,
    "dnf": Permission.INSTALL,
    "pacman": Permission.INSTALL,
    "zypper": Permission.INSTALL,
    "snap": Permission.INSTALL,
    "flatpak": Permission.INSTALL,
    "brew": Permission.INSTALL,
    "pip": Permission.INSTALL,
    "pip3": Permission.INSTALL,
    "pipx": Permission.INSTALL,
    "npm": Permission.INSTALL,
    "yarn": Permission.INSTALL,
    "pnpm": Permission.INSTALL,
    "cargo": Permission.INSTALL,
    "gem": Permission.INSTALL,
    "go": Permission.INSTALL,  # 'go install' etc.
    # ── DELETE: destructive filesystem ops ────────────────────────────────────
    "rm": Permission.DELETE,
    "rmdir": Permission.DELETE,
    "shred": Permission.DELETE,
    "wipe": Permission.DELETE,
    "truncate": Permission.DELETE,
    # ── EXECUTE: shell spawning ───────────────────────────────────────────────
    "bash": Permission.EXECUTE,
    "sh": Permission.EXECUTE,
    "zsh": Permission.EXECUTE,
    "fish": Permission.EXECUTE,
    "dash": Permission.EXECUTE,
    "ksh": Permission.EXECUTE,
    "tcsh": Permission.EXECUTE,
    "csh": Permission.EXECUTE,
    "xterm": Permission.EXECUTE,
    "gnome-terminal": Permission.EXECUTE,
    # ── NETWORK: network utilities ────────────────────────────────────────────
    "curl": Permission.NETWORK,
    "wget": Permission.NETWORK,
    "ssh": Permission.NETWORK,
    "scp": Permission.NETWORK,
    "rsync": Permission.NETWORK,
    "ftp": Permission.NETWORK,
    "sftp": Permission.NETWORK,
    "nc": Permission.NETWORK,
    "ncat": Permission.NETWORK,
    "netcat": Permission.NETWORK,
    "nmap": Permission.NETWORK,
    "ping": Permission.NETWORK,
    "traceroute": Permission.NETWORK,
    "dig": Permission.NETWORK,
    "nslookup": Permission.NETWORK,
    "host": Permission.NETWORK,
    "whois": Permission.NETWORK,
}

# Path prefixes that require SYSTEM permission regardless of command.
_SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
    "/dev/",
    "/proc/",
    "/sys/",
    "/boot/",
    "/etc/",
    "/run/",
    "/var/run/",
    "/lib/",
    "/lib64/",
    "/usr/lib/",
    "/sbin/",
    "/usr/sbin/",
)

_MAX_TIMEOUT: float = 60.0


def _resolve_safe_cwd(cwd: str | None) -> Path:
    base_dir = Path.cwd().resolve()
    if not cwd:
        return base_dir
    requested = Path(cwd)
    if requested.is_absolute():
        raise ValueError("Absolute paths are not allowed for cwd")
    resolved = (base_dir / requested).resolve()
    if resolved != base_dir and base_dir not in resolved.parents:
        raise ValueError("cwd must stay inside the project directory")
    return resolved


def _classify_command(command: str) -> list[tuple[str, Permission]]:
    """
    Return a list of ``(token, Permission)`` pairs for every token in
    *command* that maps to a restricted permission.

    Tokens not in ``_COMMAND_PERMISSIONS`` and not touching a system path
    are considered unrestricted.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []

    hits: list[tuple[str, Permission]] = []
    for token in tokens:
        if token in _COMMAND_PERMISSIONS:
            hits.append((token, _COMMAND_PERMISSIONS[token]))
        else:
            for prefix in _SYSTEM_PATH_PREFIXES:
                if prefix in token:
                    hits.append((token, Permission.SYSTEM))
                    break
    return hits


def _run_bash_sync(
    command: str,
    policy: SandboxPolicy,
    cwd: str | None = None,
    timeout: float = 10.0,
    confirmed: bool = False,
) -> ToolResult:
    command = command.strip()
    if not command:
        return ToolResult(False, "empty command", {"command": command})

    # Validate shell syntax before anything else
    try:
        shlex.split(command)
    except ValueError as exc:
        return ToolResult(False, f"invalid shell syntax: {exc}", {"command": command})

    # Classify every token against the permission map
    hits = _classify_command(command)

    for token, perm in hits:
        verdict = policy.check(perm, confirmed=confirmed)

        if verdict == "block":
            return ToolResult(
                success=False,
                summary=f"blocked: {token!r} requires {perm.name} permission",
                data={
                    "command": command,
                    "blocked_token": token,
                    "required_permission": perm.name,
                    "policy_action": "block",
                },
            )

        if verdict == "confirm":
            # Return a structured result so the caller can surface the prompt.
            return ToolResult(
                success=False,
                summary=(
                    f"confirmation required: {token!r} requires {perm.name} permission"
                ),
                data={
                    "command": command,
                    "blocked_token": token,
                    "required_permission": perm.name,
                    "policy_action": "confirm",
                    # Callers detect this key and re-invoke with confirmed=True.
                    "confirmation_required": True,
                },
            )

        # verdict == "allow" → continue checking remaining tokens

    # All tokens cleared — execute
    try:
        work_dir = _resolve_safe_cwd(cwd)
    except ValueError as exc:
        return ToolResult(False, str(exc), {"command": command})

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(Path.home()),
    }

    wrapped = f"bash -lc {shlex.quote(command)}"

    try:
        proc = subprocess.run(
            wrapped,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(0.5, min(float(timeout), _MAX_TIMEOUT)),
            shell=True,
            executable="/bin/bash",
            check=False,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        summary = f"exit code {proc.returncode}"
        if stdout.strip():
            summary = stdout.strip().splitlines()[0][:120]
        elif stderr.strip():
            summary = stderr.strip().splitlines()[0][:120]

        return ToolResult(
            success=(proc.returncode == 0),
            summary=summary,
            data={
                "command": command,
                "cwd": str(work_dir),
                "returncode": proc.returncode,
                "stdout": stdout[:12_000],
                "stderr": stderr[:12_000],
            },
        )

    except subprocess.TimeoutExpired:
        return ToolResult(
            False,
            "command timed out",
            {"command": command, "timeout": timeout},
        )

    except Exception as exc:
        return ToolResult(
            False,
            f"run_bash failed: {exc}",
            {"command": command},
        )


# ── public entry point ────────────────────────────────────────────────────────


def run_bash(
    command: str,
    policy: SandboxPolicy | None = None,
    cwd: str | None = None,
    timeout: float = 10.0,
    confirmed: bool = False,
) -> ToolResult:
    """
    Execute a bash *command* subject to *policy*.

    Parameters
    ----------
    command:
        The shell command to run.
    policy:
        :class:`~policy.SandboxPolicy` that controls which commands are
        permitted, which require confirmation, and which are hard-blocked.
        Defaults to :data:`~policy.DEFAULT_POLICY`.
    cwd:
        Relative working directory (must stay inside the project root).
    timeout:
        Maximum wall-clock seconds (capped at 60).
    confirmed:
        Set to ``True`` to proceed past a ``confirmation_required`` result.
        The typical flow is::

            result = run_bash("rm -rf build/", policy)
            if result.data.get("confirmation_required"):
                # … surface prompt to user …
                if user_says_yes:
                    result = run_bash("rm -rf build/", policy, confirmed=True)
    """
    if policy is None:
        policy = DEFAULT_POLICY

    return _run_bash_sync(
        command=command,
        policy=policy,
        cwd=cwd,
        timeout=timeout,
        confirmed=confirmed,
    )
