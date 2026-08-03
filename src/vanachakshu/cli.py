"""Command-line entry point.

Kept deliberately thin: the CLI parses arguments, calls into the library, and
renders results. All logic lives in the modules it calls, so nothing here needs
testing beyond the exit codes.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from vanachakshu import __version__
from vanachakshu.config import WESTERN_GHATS_CLEAR_SEASON, YELLAPUR_TALUK
from vanachakshu.diagnostics import all_passed, run_diagnostics

app = typer.Typer(
    name="vanachakshu",
    help="Near-real-time forest disturbance alerts for the Western Ghats.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"vanachakshu {__version__}")


@app.command()
def doctor() -> None:
    """Verify that Earth Engine access and data sources are working.

    Runs setup checks in dependency order and stops at the first failure, so
    the report names the actual problem rather than everything downstream of it.

    Exits non-zero on failure, so the scheduled job can use it as a precondition.
    """
    console.print(
        f"\n[bold]Checking setup[/bold] for {YELLAPUR_TALUK.name}, "
        f"season {WESTERN_GHATS_CLEAR_SEASON.start_month:02d}-"
        f"{WESTERN_GHATS_CLEAR_SEASON.end_month:02d}\n"
    )

    results = run_diagnostics()

    for result in results:
        if result.ok:
            console.print(f"  [green]PASS[/green]  {result.name}")
            console.print(f"        [dim]{result.detail}[/dim]")
        else:
            console.print(f"  [red]FAIL[/red]  {result.name}")
            console.print(f"        [dim]{result.detail}[/dim]")

    console.print()

    if all_passed(results):
        console.print("[bold green]All checks passed.[/bold green] Setup is working.\n")
        raise typer.Exit(0)

    for result in results:
        if not result.ok and result.remediation:
            console.print(
                Panel(
                    result.remediation,
                    title=f"How to fix: {result.name}",
                    border_style="yellow",
                )
            )

    console.print("[bold red]Setup is not yet working.[/bold red]\n")
    raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
