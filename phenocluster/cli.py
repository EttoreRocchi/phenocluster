"""
PhenoCluster Command-Line Interface
====================================

CLI for clinical phenotype discovery using latent class analysis.
"""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from . import __version__
from .config import PhenoClusterConfig
from .pipeline import PhenoClusterPipeline
from .profiles import create_config_yaml, list_profiles

app = typer.Typer(
    name="phenocluster",
    help="PhenoCluster - Clinical Phenotype Discovery Pipeline",
    add_completion=False,
)
console = Console()


def show_banner():
    """Display application banner."""
    from rich.text import Text

    content = Text(justify="center")
    content.append("\n")
    content.append("P H E N O", style="bold cyan")
    content.append("  |  ", style="dim cyan")
    content.append("C L U S T E R", style="bold white")
    content.append("\n\n")
    content.append("Clinical Phenotype Discovery Pipeline", style="dim")
    content.append("\n")
    content.append(f"v{__version__}", style="dim cyan")
    content.append("\n")
    console.print(Panel(content, border_style="cyan", padding=(0, 4), width=min(80, console.width)))


@app.command("run")
def run(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to input CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    force_rerun: bool = typer.Option(
        False,
        "--force-rerun",
        help="Ignore cached artifacts and re-run all steps",
    ),
):
    """
    Run the phenotype discovery pipeline.

    All parameters are controlled via the configuration YAML file.
    Use 'create-config' to generate a config from a profile.

    Example:

        phenocluster run -d data.csv -c config.yaml
        phenocluster run -d data.csv -c config.yaml --force-rerun
    """
    show_banner()

    try:
        # Load data
        with console.status("[bold cyan]Loading data...", spinner="dots"):
            df = pd.read_csv(data)

        console.print(
            f"[green]OK[/green] Loaded data: {len(df):,} rows x {len(df.columns)} columns\n"
        )

        # Load configuration
        with console.status("[bold cyan]Loading configuration...", spinner="dots"):
            cfg = PhenoClusterConfig.from_yaml(config)

        # Show configuration summary
        config_table = Table(
            title="Configuration Summary", show_header=True, header_style="bold cyan"
        )
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Project", cfg.project_name)
        config_table.add_row("Output", cfg.output_dir)
        config_table.add_row("Continuous features", str(len(cfg.continuous_columns)))
        config_table.add_row("Categorical features", str(len(cfg.categorical_columns)))
        config_table.add_row(
            "Outcomes",
            f"{len(cfg.outcome_columns)} ({'on' if cfg.outcome.enabled else 'off'})",
        )

        if cfg.model_selection.enabled:
            config_table.add_row(
                "Model selection",
                (
                    f"{cfg.model_selection.min_clusters}-"
                    f"{cfg.model_selection.max_clusters} clusters"
                    f" ({cfg.model_selection.criterion})"
                ),
            )
        else:
            config_table.add_row("Clusters (fixed)", str(cfg.n_clusters))

        config_table.add_row(
            "Train/test split",
            (
                f"{int((1 - cfg.data_split.test_size) * 100)}%"
                f" / {int(cfg.data_split.test_size * 100)}%"
            ),
        )
        config_table.add_row(
            "Stability analysis", "Enabled" if cfg.stability.enabled else "Disabled"
        )
        if cfg.survival.enabled:
            n_targets = len(cfg.survival.targets) if cfg.survival.targets else 0
            config_table.add_row("Survival analysis", f"Enabled ({n_targets} targets)")
        else:
            config_table.add_row("Survival analysis", "Disabled")

        if cfg.multistate.enabled:
            n_states = len(cfg.multistate.states) if cfg.multistate.states else 0
            config_table.add_row("Multistate analysis", f"Enabled ({n_states} states)")
        else:
            config_table.add_row("Multistate analysis", "Disabled")

        config_table.add_row(
            "Inference",
            "Enabled" if cfg.inference.enabled else "Descriptive only",
        )
        config_table.add_row("Random seed", str(cfg.random_state))

        console.print(config_table)
        console.print()

        # Create and run pipeline
        console.print(
            Panel.fit(
                "[bold cyan]Starting Pipeline Execution[/bold cyan]",
                border_style="cyan",
            )
        )

        pipeline = PhenoClusterPipeline(cfg)
        results = pipeline.fit(df, force_rerun=force_rerun)
        pipeline.save_results()

        # Display results
        console.print("\n")
        results_table = Table(
            title="[bold green]Pipeline Completed Successfully",
            show_header=True,
            header_style="bold green",
        )
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green", justify="right")

        results_table.add_row("Clusters identified", str(results["n_clusters"]))
        results_table.add_row("Total samples", f"{results.get('n_samples', 'N/A'):,}")

        if results.get("split_info"):
            split = results["split_info"]
            results_table.add_row("Training samples", f"{split.get('train_size', 'N/A'):,}")
            results_table.add_row("Test samples", f"{split.get('test_size', 'N/A'):,}")

        if results.get("model_selection"):
            ms = results["model_selection"]
            criterion = ms.get("criterion_used", ms.get("criterion", "BIC"))
            # Use the CV-averaged score (consistent with comparison table)
            best_k = ms.get("best_n_clusters")
            criterion_val = None
            for r in ms.get("all_results", []):
                if r.get("n_clusters") == best_k:
                    criterion_val = r.get(criterion)
                    break
            if criterion_val is not None:
                results_table.add_row(f"Best {criterion}", f"{criterion_val:.2f}")

        if results.get("stability_results"):
            stability = results["stability_results"]
            results_table.add_row("Stability score", f"{stability.get('mean_consensus', 0):.3f}")

        console.print(results_table)
        console.print(f"\n[bold green]Results saved to:[/bold green] {cfg.output_dir}")

        # Show output files
        output_path = Path(cfg.output_dir)
        if output_path.exists():
            tree = Tree(f"[bold]{output_path.name}/")
            for item in sorted(output_path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size / 1024  # KB
                    tree.add(f"[cyan]{item.name}[/cyan] [dim]({size:.1f} KB)")
            console.print(tree)

        console.print("\n[bold green]Done.[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        raise typer.Exit(code=130)
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print("[dim]Run with logging.level: DEBUG in config for full traceback")
        raise typer.Exit(code=1)
    except Exception as e:
        import traceback

        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        console.print(f"[dim]{traceback.format_exc(limit=3)}")
        raise typer.Exit(code=1)


@app.command("create-config")
def create_config(
    output: Path = typer.Option(
        "config.yaml",
        "--output",
        "-o",
        help="Output YAML path",
        file_okay=True,
        dir_okay=False,
    ),
    profile: str = typer.Option(
        "complete",
        "--profile",
        "-p",
        help="Profile: descriptive, complete, quick",
    ),
):
    """
    Generate a configuration YAML file from a profile template.

    Profiles set sensible defaults for common use-cases.
    Data-specific parameters (column names, survival targets) are left
    as placeholders that you fill in.

    Profiles:

        descriptive  - Phenotype discovery only, no statistical inference
        complete     - All analyses enabled (inference + survival + multistate)
        quick        - Fast iteration for development (reduced runs)

    Example:

        phenocluster create-config -p complete -o config.yaml
        phenocluster create-config -p quick -o quick_config.yaml
    """
    show_banner()

    available = list_profiles()
    if profile not in available:
        console.print(
            f"[red]Error: Unknown profile '{profile}'. Available: {', '.join(available)}[/red]"
        )
        raise typer.Exit(code=1)

    try:
        create_config_yaml(profile, output)
        console.print(f"[bold green]Config created:[/bold green] {output}")
        console.print(f"[dim]Profile: {profile}[/dim]")
        console.print("\n[yellow]Next steps:[/yellow]")
        console.print("  1. Edit the config to add your column names and data paths")
        console.print("  2. Validate: [green]phenocluster validate-config -c config.yaml[/green]")
        console.print("  3. Run: [green]phenocluster run -d data.csv -c config.yaml[/green]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def _validate_structure(cfg, errors):
    """Check config structural validity (no data file needed)."""
    if not cfg.continuous_columns and not cfg.categorical_columns:
        errors.append("At least one of continuous_columns or categorical_columns must be non-empty")

    if cfg.outcome.enabled and not cfg.outcome.outcome_columns:
        errors.append("outcome.enabled=true but no outcome_columns specified")

    if not (0.0 < cfg.data_split.test_size < 1.0):
        errors.append(f"data.split.test_size must be in (0, 1), got {cfg.data_split.test_size}")

    if cfg.model_selection.enabled:
        if cfg.model_selection.min_clusters < 2:
            errors.append("model.selection.min_clusters must be >= 2")
        if cfg.model_selection.max_clusters < cfg.model_selection.min_clusters:
            errors.append("model.selection.max_clusters must be >= min_clusters")

    if cfg.survival.enabled and cfg.survival.targets:
        for t in cfg.survival.targets:
            if not t.time_column or not t.event_column:
                errors.append(
                    f"Survival target '{t.name}' must have both time_column and event_column"
                )

    _validate_multistate_structure(cfg, errors)

    if cfg.reference_phenotype.strategy not in (
        "largest",
        "healthiest",
        "specific",
    ):
        errors.append(
            f"reference_phenotype.strategy must be 'largest', 'healthiest', "
            f"or 'specific', got '{cfg.reference_phenotype.strategy}'"
        )

    if cfg.feature_selection.enabled:
        if cfg.feature_selection.require_target and not cfg.feature_selection.target_column:
            errors.append(
                f"feature_selection.method='{cfg.feature_selection.method}' "
                f"requires a target_column"
            )


def _validate_multistate_structure(cfg, errors):
    """Validate multistate config structure."""
    if not cfg.multistate.enabled:
        return
    if not cfg.multistate.states:
        errors.append("multistate.states must be non-empty when enabled")
    if not cfg.multistate.transitions:
        errors.append("multistate.transitions must be non-empty when enabled")
    state_ids = {s.id for s in cfg.multistate.states} if cfg.multistate.states else set()
    if cfg.multistate.transitions:
        for tr in cfg.multistate.transitions:
            if tr.from_state not in state_ids:
                errors.append(
                    f"Transition '{tr.name}' references unknown from_state {tr.from_state}"
                )
            if tr.to_state not in state_ids:
                errors.append(f"Transition '{tr.name}' references unknown to_state {tr.to_state}")


def _check_columns(col_list, section_name, csv_columns, errors):
    """Flag config columns missing from the CSV."""
    for col in col_list:
        if col not in csv_columns:
            errors.append(f"{section_name}: column '{col}' not found in data")


def _validate_against_data(cfg, csv_columns, errors, warnings_list):
    """Cross-reference config column names against CSV header."""
    _check_columns(
        cfg.continuous_columns,
        "data.continuous_columns",
        csv_columns,
        errors,
    )
    _check_columns(
        cfg.categorical_columns,
        "data.categorical_columns",
        csv_columns,
        errors,
    )
    if cfg.outcome.enabled:
        _check_columns(
            cfg.outcome.outcome_columns,
            "outcome.outcome_columns",
            csv_columns,
            errors,
        )

    if cfg.data_split.stratify_by:
        _check_columns(
            [cfg.data_split.stratify_by],
            "data.split.stratify_by",
            csv_columns,
            errors,
        )

    if cfg.feature_selection.enabled and cfg.feature_selection.target_column:
        _check_columns(
            [cfg.feature_selection.target_column],
            "preprocessing.feature_selection.target_column",
            csv_columns,
            errors,
        )

    if cfg.survival.enabled and cfg.survival.targets:
        for t in cfg.survival.targets:
            _check_columns(
                [t.time_column],
                f"survival.targets[{t.name}].time_column",
                csv_columns,
                errors,
            )
            _check_columns(
                [t.event_column],
                f"survival.targets[{t.name}].event_column",
                csv_columns,
                errors,
            )

    if cfg.multistate.enabled and cfg.multistate.states:
        for s in cfg.multistate.states:
            if s.event_column:
                _check_columns(
                    [s.event_column],
                    f"multistate.states[{s.name}].event_column",
                    csv_columns,
                    errors,
                )
            if s.time_column:
                _check_columns(
                    [s.time_column],
                    f"multistate.states[{s.name}].time_column",
                    csv_columns,
                    errors,
                )
        if cfg.multistate.baseline_confounders:
            _check_columns(
                cfg.multistate.baseline_confounders,
                "multistate.baseline_confounders",
                csv_columns,
                errors,
            )

    if cfg.reference_phenotype.strategy == "healthiest" and cfg.reference_phenotype.health_outcome:
        _check_columns(
            [cfg.reference_phenotype.health_outcome],
            "reference_phenotype.health_outcome",
            csv_columns,
            errors,
        )

    overlap = set(cfg.continuous_columns) & set(cfg.categorical_columns)
    if overlap:
        warnings_list.append(f"Columns in both continuous and categorical: {sorted(overlap)}")


def _display_validation_results(cfg, config, errors, warnings_list, csv_columns):
    """Display validation results to console."""
    if errors:
        console.print("[bold red]INVALID[/bold red] - config has errors:\n")
        for err in errors:
            console.print(f"  [red]x[/red] {err}")
    if warnings_list:
        if not errors:
            console.print("[bold yellow]WARNINGS:[/bold yellow]\n")
        else:
            console.print()
        for warn in warnings_list:
            console.print(f"  [yellow]![/yellow] {warn}")

    if not errors:
        console.print(f"\n[bold green]VALID[/bold green] - {config}")
        console.print(f"  Project: {cfg.project_name}")
        console.print(
            f"  Features: {len(cfg.continuous_columns)} continuous, "
            f"{len(cfg.categorical_columns)} categorical"
        )
        console.print(
            f"  Outcomes: {len(cfg.outcome.outcome_columns)}"
            f" ({'on' if cfg.outcome.enabled else 'off'})"
        )
        n_targets = len(cfg.survival.targets) if cfg.survival.targets else 0
        console.print(
            f"  Analyses: outcome={'on' if cfg.outcome.enabled else 'off'}, "
            f"survival={'on' if cfg.survival.enabled else 'off'} "
            f"({n_targets} targets), "
            f"multistate={'on' if cfg.multistate.enabled else 'off'}, "
            f"inference={'on' if cfg.inference.enabled else 'off'}"
        )
        if csv_columns is not None:
            console.print(f"  Data columns matched: all OK ({len(csv_columns)} in CSV)")
    else:
        raise typer.Exit(code=1)


@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    data: Path = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to CSV file - cross-checks column names against actual data",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
):
    """
    Validate a configuration YAML file.

    Checks YAML structure, required sections, value ranges, and
    internal consistency (e.g. survival targets reference valid columns,
    multistate transitions reference valid states).

    When --data is supplied, also cross-references every column
    name in the config against the actual CSV header, catching typos and
    missing columns before a long pipeline run.

    Examples:

        phenocluster validate-config -c config.yaml
        phenocluster validate-config -c config.yaml -d data.csv
    """
    show_banner()

    errors = []
    warnings_list = []

    try:
        cfg = PhenoClusterConfig.from_yaml(config)
    except Exception as e:
        console.print(f"[bold red]INVALID[/bold red] - failed to parse config: {e}")
        raise typer.Exit(code=1)

    _validate_structure(cfg, errors)

    csv_columns = None
    if data is not None:
        try:
            csv_columns = set(pd.read_csv(data, nrows=0).columns)
        except Exception as e:
            console.print(f"[bold red]Error reading data file:[/bold red] {e}")
            raise typer.Exit(code=1)

        console.print(f"[green]OK[/green] Data header: {len(csv_columns)} columns in {data.name}\n")
        _validate_against_data(cfg, csv_columns, errors, warnings_list)

    _display_validation_results(cfg, config, errors, warnings_list, csv_columns)


@app.command("version")
def version():
    """Show version information."""
    from rich.text import Text

    content = Text(justify="center")
    content.append("\n")
    content.append("P H E N O", style="bold cyan")
    content.append("  |  ", style="dim cyan")
    content.append("C L U S T E R", style="bold white")
    content.append("\n\n")
    content.append("Clinical Phenotype Discovery Pipeline", style="dim")
    content.append("\n\n")
    content.append(f"v{__version__}", style="bold cyan")
    content.append("\n")
    content.append(
        "Click here for the GitHub repository",
        style="dim cyan link https://github.com/EttoreRocchi/phenocluster",
    )
    content.append("\n")
    content.append(
        "Click here for the documentation",
        style="dim cyan link https://ettorerocchi.github.io/phenocluster",
    )
    content.append("\n")
    console.print(Panel(content, border_style="cyan", padding=(0, 4), width=min(80, console.width)))


def main():
    """Main entry point."""
    app()


# sphinx-click bridge: expose Typer app as a Click group for documentation generation
typer_click_object = typer.main.get_command(app)


if __name__ == "__main__":
    main()
