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
from vanachakshu.chips import download_chips, write_contact_sheet
from vanachakshu.config import WESTERN_GHATS_CLEAR_SEASON, YELLAPUR_TALUK, AlertConfig
from vanachakshu.diagnostics import all_passed, run_diagnostics
from vanachakshu.gee import EarthEngineSetupError, initialize
from vanachakshu.pipeline import default_comparison_years, run_cycle, store_path_for
from vanachakshu.report import format_digest, write_reports
from vanachakshu.validation import (
    SIZE_STRATA,
    load_verdicts,
    precision_report,
    size_stratum,
    stratified_sample,
    worksheet_alert_ids,
    write_worksheet,
)

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


@app.command("validate-sample")
def validate_sample(
    per_stratum: Annotated[int, typer.Option(help="Detections to draw from each size band.")] = 25,
    seed: Annotated[
        int,
        typer.Option(
            help="Random seed. Recorded in the output filename so the sample "
            "can be redrawn and audited by someone else."
        ),
    ] = 42,
    store_dir: Annotated[Path | None, typer.Option(help="Alert store directory.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Where to write the worksheet.")] = None,
) -> None:
    """Draw a stratified sample of detections to check against imagery.

    Hansen cannot settle whether these detections are real: it is a 30 m annual
    product and most detections are now sub-hectare. Looking at them is the only
    way, and this builds the worksheet for it.
    """
    aoi = YELLAPUR_TALUK
    store = AlertStore(store_path_for(aoi, store_dir), AlertConfig())
    store.load()

    if not store.alerts:
        console.print("[yellow]The alert store is empty — nothing to validate yet.[/yellow]")
        raise typer.Exit(1)

    sample = stratified_sample(store.alerts, per_stratum=per_stratum, seed=seed)
    target = (out_dir if out_dir is not None else Path("data") / "validation") / (
        f"{aoi.slug}-sample-seed{seed}.csv"
    )
    path = write_worksheet(sample, target)

    console.print(
        f"\n[bold]{len(sample)}[/bold] detections sampled from {len(store.alerts)} tracked"
    )
    counts: dict[str, int] = {}
    for alert in sample:
        stratum = size_stratum(alert.area_ha)
        counts[stratum] = counts.get(stratum, 0) + 1
    for name, _, _ in SIZE_STRATA:
        console.print(f"  {name:>12}: {counts.get(name, 0)}")

    console.print(f"\nworksheet: {path}")
    console.print(
        "\n[dim]Open each satellite_view link, decide whether a clearing is visible,\n"
        "and write true_positive, false_positive or unclear in the verdict column.\n"
        "Leave genuinely ambiguous ones as 'unclear' rather than guessing — a forced\n"
        "call invents certainty, and how often it is unclear is itself a finding.[/dim]\n"
    )


@app.command("validate-chips")
def validate_chips(
    worksheet: Annotated[Path, typer.Argument(help="Worksheet CSV from validate-sample.")],
    baseline: Annotated[int | None, typer.Option(help="Earlier year to show.")] = None,
    recent: Annotated[int | None, typer.Option(help="Later year to show.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Where to write chips and HTML.")] = None,
) -> None:
    """Fetch before/after image chips and build a review page.

    The worksheet's Google Maps links are very high resolution but undated —
    usually one to three years stale — so they answer "is there a clearing
    here?" when the question is "did something change between these years?".
    These chips are dated, cloud-masked, and shown side by side.
    """
    if not worksheet.is_file():
        console.print(f"[red]No such worksheet:[/red] {worksheet}")
        raise typer.Exit(1)

    aoi = YELLAPUR_TALUK
    _, default_recent = default_comparison_years(WESTERN_GHATS_CLEAR_SEASON, date.today())
    recent = recent if recent is not None else default_recent
    baseline = baseline if baseline is not None else recent - 1

    store = AlertStore(store_path_for(aoi), AlertConfig())
    store.load()
    # Every row, not only reviewed ones — imagery is fetched before anyone
    # reviews anything.
    wanted = set(worksheet_alert_ids(worksheet))
    alerts = [a for a in store.alerts if a.alert_id in wanted]

    if not alerts:
        console.print("[yellow]No matching detections found in the alert store.[/yellow]")
        raise typer.Exit(1)

    try:
        initialize()
    except EarthEngineSetupError as exc:
        console.print(Panel(str(exc), border_style="yellow"))
        raise typer.Exit(1) from exc

    target = out_dir if out_dir is not None else Path("data") / "validation" / "chips"
    console.print(f"\nFetching chips for {len(alerts)} detections ({baseline} vs {recent})...")

    chipsets = download_chips(
        alerts=alerts,
        baseline_year=baseline,
        recent_year=recent,
        season=WESTERN_GHATS_CLEAR_SEASON,
        out_dir=target,
    )
    page = write_contact_sheet(chipsets, target / "index.html")

    complete = sum(c.is_complete for c in chipsets)
    with_nicfi = sum(c.nicfi_path is not None for c in chipsets)
    console.print(f"  both dated chips : {complete}/{len(chipsets)}")
    console.print(f"  with NICFI <5 m  : {with_nicfi}/{len(chipsets)}")
    console.print(f"\nreview page: {page}")
    console.print(f"\n[dim]Open it with:  start {page}[/dim]\n")


@app.command("validate-report")
def validate_report(
    worksheet: Annotated[Path, typer.Argument(help="Filled-in worksheet CSV.")],
) -> None:
    """Report precision per size band from a completed worksheet."""
    if not worksheet.is_file():
        console.print(f"[red]No such worksheet:[/red] {worksheet}")
        raise typer.Exit(1)

    records = load_verdicts(worksheet)
    if not records:
        console.print("[yellow]No verdicts filled in yet.[/yellow]")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]Validation against high-resolution imagery[/bold] — {len(records)} judged\n"
    )
    for result in precision_report(records):
        console.print(f"  {result.as_row()}")

    judged = sum(r.judged for r in precision_report(records))
    console.print(
        f"\n[dim]Intervals are 95% Wilson score. Precision on n={judged} is a range,\n"
        "not a point — quote it with the interval or not at all.[/dim]\n"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
