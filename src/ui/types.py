from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CheckResult:
    name: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    fixable: bool = False
    fix_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorCheck:
    run: Callable[[], CheckResult]
    fix: Callable[[CheckResult], None] | None = None
