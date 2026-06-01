import typer
import questionary

from rich.console import Console
from rich.table import Table

from src.ui.types import DoctorCheck
from src.ui.commands.doctor.env import env_check, fix_env

console = Console()


def confirm(message: str, default: bool = False) -> bool:
    return questionary.confirm(
        message,
        default=default,
    ).ask()


def print_result(result):
    status = "[green]OK[/green]" if result.ok else "[yellow]ISSUES[/yellow]"
    console.print(f"\n[bold]{result.name}[/bold] — {status}")

    for issue in result.issues:
        console.print(f"  • {issue}")


def ask_fix(result, auto_fix: bool) -> bool:
    if result.ok or not result.fixable:
        return False

    if auto_fix:
        return True

    return confirm(f"Fix {result.name}?", default=False)


def doctor(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues"),
):
    console.rule("[bold]Doctor[/bold]")

    checks: list[DoctorCheck] = [
        DoctorCheck(run=env_check, fix=fix_env),
        # DoctorCheck(run=config_check, fix=fix_config),
        # DoctorCheck(run=api_check, fix=fix_api),
    ]

    results = []
    for check in checks:
        result = check.run()
        results.append((check, result))
        print_result(result)

    failed = [(check, result) for check, result in results if not result.ok]

    if not failed:
        console.print("\n[green]All checks passed.[/green]")
        raise typer.Exit()

    console.print("\n[bold]Summary[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Issues")

    for _, result in results:
        table.add_row(
            result.name,
            "OK" if result.ok else "FAILED",
            str(len(result.issues)),
        )

    console.print(table)

    for check, result in failed:
        if not check.fix:
            continue

        if ask_fix(result, fix):
            check.fix(result)
            console.print(f"[green]Fixed {result.name}[/green]")
        else:
            console.print(f"[yellow]Skipped {result.name}[/yellow]")
