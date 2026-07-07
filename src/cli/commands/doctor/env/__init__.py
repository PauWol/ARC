from src.cli.types import CheckResult
from src.util.heal import ensure_dot_env, check_missing_dot_env_entrys, repair_dot_env
from src.util.constants import ENV_PATH, DEFAULT_DOT_ENV


def env_check() -> CheckResult:
    if not ensure_dot_env():
        return CheckResult(
            name=".env",
            ok=False,
            issues=["missing .env file"],
            fixable=True,
            fix_data={
                "create_file": True,
                "wrong": [],
                "missing": list(DEFAULT_DOT_ENV),
            },
        )

    wrong, missing = check_missing_dot_env_entrys()

    issues = []

    if wrong:
        issues = [f"wrong: {line}" for line in wrong]

    if missing:
        issues.extend([f"missing: {item}" for item in missing])

    return CheckResult(
        name=".env",
        ok=not issues,
        issues=issues,
        fixable=True,
        fix_data={
            "create_file": False,
            "wrong": wrong,
            "missing": missing,
        },
    )


def fix_env(result: CheckResult):
    result = result.fix_data
    file = result.get("create_file", False)
    wrong = result.get("wrong", [])
    missing = result.get("missing", [])

    if file:
        ENV_PATH.write_text("")

    repair_dot_env(missing, wrong)
