"""`phenocluster run` — execute the phenotype discovery pipeline."""

from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree

from ...config import PhenoClusterConfig
from ..banner import show_banner
from ..console import console
from ..errors import handle_cli_errors

HELP_PANEL_PIPELINE = "Pipeline"


def _render_config_summary(cfg: PhenoClusterConfig) -> Table:
    table = Table(title="Configuration Summary", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Project", cfg.project_name)
    table.add_row("Output", cfg.output_dir)
    table.add_row("Continuous features", str(len(cfg.continuous_columns)))
    table.add_row("Categorical features", str(len(cfg.categorical_columns)))
    table.add_row(
        "Outcomes",
        f"{len(cfg.outcome_columns)} ({'on' if cfg.outcome.enabled else 'off'})",
    )

    if cfg.model_selection.enabled:
        table.add_row(
            "Model selection",
            (
                f"{cfg.model_selection.min_clusters}-"
                f"{cfg.model_selection.max_clusters} clusters"
                f" ({cfg.model_selection.criterion})"
            ),
        )
    else:
        table.add_row("Clusters (fixed)", str(cfg.n_clusters))

    table.add_row(
        "Train/test split",
        (f"{int((1 - cfg.data_split.test_size) * 100)}% / {int(cfg.data_split.test_size * 100)}%"),
    )
    table.add_row("Stability analysis", "Enabled" if cfg.stability.enabled else "Disabled")
    if cfg.survival.enabled:
        n_targets = len(cfg.survival.targets) if cfg.survival.targets else 0
        table.add_row("Survival analysis", f"Enabled ({n_targets} targets)")
    else:
        table.add_row("Survival analysis", "Disabled")

    if cfg.multistate.enabled:
        n_states = len(cfg.multistate.states) if cfg.multistate.states else 0
        table.add_row("Multistate analysis", f"Enabled ({n_states} states)")
    else:
        table.add_row("Multistate analysis", "Disabled")

    table.add_row("Inference", "Enabled" if cfg.inference.enabled else "Descriptive only")
    table.add_row("Random seed", str(cfg.random_state))
    return table


def _render_results(results: dict) -> Table:
    table = Table(
        title="[bold green]Pipeline Completed Successfully",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Clusters identified", str(results["n_clusters"]))
    table.add_row("Total samples", f"{results.get('n_samples', 'N/A'):,}")

    if results.get("split_info"):
        split = results["split_info"]
        table.add_row("Training samples", f"{split.get('train_size', 'N/A'):,}")
        table.add_row("Test samples", f"{split.get('test_size', 'N/A'):,}")

    if results.get("model_selection"):
        ms = results["model_selection"]
        criterion = ms.get("criterion_used", ms.get("criterion", "BIC"))
        best_k = ms.get("best_n_clusters")
        criterion_val = None
        for r in ms.get("all_results", []):
            if r.get("n_clusters") == best_k:
                criterion_val = r.get(criterion)
                break
        if criterion_val is not None:
            table.add_row(f"Best {criterion}", f"{criterion_val:.2f}")

    if results.get("stability_results"):
        stability = results["stability_results"]
        table.add_row("Stability score", f"{stability.get('mean_consensus', 0):.3f}")
    return table


def register(app: typer.Typer) -> None:
    @app.command("run", rich_help_panel=HELP_PANEL_PIPELINE)
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
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable DEBUG logging and full tracebacks on error.",
        ),
        quiet: bool = typer.Option(
            False,
            "--quiet",
            "-q",
            help="Suppress the banner and non-essential output.",
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
        if not quiet:
            show_banner()

        # Upgrade to rich tracebacks with locals when verbose, and route
        # pipeline logging through the shared Rich Console.
        from ...utils.logging import PhenoClusterLogger
        from ..errors import install_tracebacks

        install_tracebacks(verbose=verbose)
        PhenoClusterLogger.configure_cli(console=console, verbose=verbose, quiet=quiet)

        # Resolve pandas / pipeline / config through the `phenocluster.cli`
        # package namespace so tests that patch `phenocluster.cli.pd`,
        # `phenocluster.cli.PhenoClusterPipeline`, and
        # `phenocluster.cli.PhenoClusterConfig.from_yaml` continue to work.
        from phenocluster import cli as _cli_pkg

        with handle_cli_errors(verbose=verbose):
            with console.status("[bold cyan]Loading data...", spinner="dots"):
                df = _cli_pkg.pd.read_csv(data)

            console.print(
                f"[green]OK[/green] Loaded data: {len(df):,} rows x {len(df.columns)} columns\n"
            )

            with console.status("[bold cyan]Loading configuration...", spinner="dots"):
                cfg = _cli_pkg.PhenoClusterConfig.from_yaml(config)

            console.print(_render_config_summary(cfg))
            console.print()

            console.print(
                Panel.fit(
                    "[bold cyan]Starting Pipeline Execution[/bold cyan]",
                    border_style="cyan",
                )
            )

            pipeline = _cli_pkg.PhenoClusterPipeline(cfg)

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )

            with progress:
                task_id = progress.add_task("Running pipeline", total=1.0)

                def progress_callback(stage: str, fraction: float) -> None:
                    progress.update(
                        task_id,
                        description=f"[bold cyan]{stage}",
                        completed=max(0.0, min(fraction, 1.0)),
                    )

                try:
                    results = pipeline.fit(
                        df, force_rerun=force_rerun, progress_callback=progress_callback
                    )
                except TypeError:
                    results = pipeline.fit(df, force_rerun=force_rerun)
                progress.update(task_id, completed=1.0, description="[bold green]Done")

            pipeline.save_results()

            console.print("\n")
            console.print(_render_results(results))
            console.print(f"\n[bold green]Results saved to:[/bold green] {cfg.output_dir}")

            output_path = Path(cfg.output_dir)
            if output_path.exists():
                tree = Tree(f"[bold]{output_path.name}/")
                for item in sorted(output_path.iterdir()):
                    if item.is_file():
                        size = item.stat().st_size / 1024  # KB
                        tree.add(f"[cyan]{item.name}[/cyan] [dim]({size:.1f} KB)[/dim]")
                console.print(tree)

            console.print("\n[bold green]Done.[/bold green]")
