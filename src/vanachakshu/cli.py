"""Command-line entry point.

Kept deliberately thin: the CLI parses arguments, calls into the library, and
renders results. All logic lives in the modules it calls, so nothing here needs
testing beyond the exit codes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from vanachakshu import __version__
from vanachakshu.alerts import AlertStore
from vanachakshu.config import WESTERN_GHATS_CLEAR_SEASON, YELLAPUR_TALUK, AlertConfig
from vanachakshu.diagnostics import all_passed, run_diagnostics
from vanachakshu.gee import EarthEngineSetupError, initialize
from vanachakshu.pipeline import default_comparison_years, run_cycle, store_path_for
from vanachakshu.report import format_digest, write_reports

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


@app.command()
def run(
    baseline: Annotated[
        int | None,
        typer.Option(help="Earlier year. Defaults to one year before --recent."),
    ] = None,
    recent: Annotated[
        int | None,
        typer.Option(
            help="Later year. Defaults to the most recent *complete* season, "
            "since compositing a season still in progress reads as vegetation loss."
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Detect and report without writing the alert store. "
            "Use when trying a threshold: confirmation state is not recoverable "
            "once an alert has been marked announced.",
        ),
    ] = False,
    store_dir: Annotated[
        Path | None,
        typer.Option(help="Directory holding alert stores. Defaults to data/alerts."),
    ] = None,
    report_dir: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the digest and GeoJSON. Defaults to data/output "
            "(gitignored — reports are derivable from the store, so they are "
            "regenerated rather than versioned)."
        ),
    ] = None,
) -> None:
    """Run one detection cycle and record any newly confirmed alerts.

    Exits non-zero if Earth Engine is not usable, so the scheduled job fails
    loudly rather than silently reporting nothing found.
    """
    aoi = YELLAPUR_TALUK
    today = date.today()

    # The scheduled job passes neither, so it works them out itself. Baseline is
    # derived from the *resolved* recent year, so `--recent 2023` alone still
    # gives a sensible 2022 baseline rather than one anchored to today.
    _, default_recent = default_comparison_years(WESTERN_GHATS_CLEAR_SEASON, today)
    recent = recent if recent is not None else default_recent
    baseline = baseline if baseline is not None else recent - 1

    try:
        initialize()
    except EarthEngineSetupError as exc:
        console.print("[bold red]Earth Engine is not usable.[/bold red]")
        console.print(Panel(str(exc), border_style="yellow"))
        raise typer.Exit(1) from exc

    store = AlertStore(store_path_for(aoi, store_dir), AlertConfig())

    console.print(f"\n[bold]{aoi.name}[/bold] — {baseline} vs {recent}\n")
    result = run_cycle(
        aoi=aoi,
        season=WESTERN_GHATS_CLEAR_SEASON,
        baseline_year=baseline,
        recent_year=recent,
        today=today,
        store=store,
        dry_run=dry_run,
    )

    for line in result.summary_lines():
        console.print(f"  {line}" if line.startswith(" ") else line)

    files = write_reports(
        new_alerts=result.new_alerts,
        all_alerts=store.alerts,
        aoi=aoi,
        out_dir=report_dir if report_dir is not None else Path("data") / "output",
        issued_on=today,
    )

    if result.new_alerts:
        # The full digest, not a summary of it: this is the text a person acts
        # on, and it should be visible wherever the run is visible.
        console.print()
        console.print(format_digest(result.new_alerts, aoi, today))
    else:
        console.print("\n[green]No newly confirmed disturbances.[/green]")

    console.print(f"\n[dim]digest  : {files.digest}[/dim]")
    console.print(f"[dim]geojson : {files.geojson}  ({len(store.alerts)} tracked)[/dim]")
    console.print()


if __name__ == "__main__":  # pragma: no cover
    app()
