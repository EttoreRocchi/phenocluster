"""`create-config`, `validate-config`, `list-profiles`, `show-profile`."""

import tempfile
from enum import Enum
from pathlib import Path

import pandas as pd
import typer
from rich.syntax import Syntax
from rich.table import Table

from ...config import PhenoClusterConfig
from ...profiles import PROFILES, create_config_yaml, list_profiles
from ..banner import show_banner
from ..console import console
from ..errors import handle_cli_errors
from ..validation import (
    _display_validation_results,
    _validate_against_data,
    _validate_structure,
)

HELP_PANEL_CONFIG = "Configuration"


def _build_profile_enum() -> type[Enum]:
    """Build a dynamic Enum of available profile names for Typer completion."""
    return Enum("ProfileName", {name: name for name in list_profiles()})


ProfileName = _build_profile_enum()


def register(app: typer.Typer) -> None:
    @app.command("create-config", rich_help_panel=HELP_PANEL_CONFIG)
    def create_config(
        output: Path = typer.Option(
            "config.yaml",
            "--output",
            "-o",
            help="Output YAML path",
            file_okay=True,
            dir_okay=False,
        ),
        profile: ProfileName = typer.Option(
            ProfileName("complete"),
            "--profile",
            "-p",
            help="Profile template to use",
            case_sensitive=False,
        ),
    ):
        """
        Generate a configuration YAML file from a profile template.

        Profiles set sensible defaults for common use-cases.
        Data-specific parameters (column names, survival targets) are left
        as placeholders that you fill in.

        Example:

            phenocluster create-config -p complete -o config.yaml
            phenocluster create-config -p quick -o quick_config.yaml
        """
        show_banner()

        profile_name = profile.value if isinstance(profile, Enum) else str(profile)

        with handle_cli_errors(verbose=False):
            create_config_yaml(profile_name, output)
            console.print(f"[bold green]Config created:[/bold green] {output}")
            console.print(f"[dim]Profile: {profile_name}[/dim]")
            console.print("\n[yellow]Next steps:[/yellow]")
            console.print("  1. Edit the config to add your column names and data paths")
            console.print(
                "  2. Validate: [green]phenocluster validate-config -c config.yaml[/green]"
            )
            console.print("  3. Run: [green]phenocluster run -d data.csv -c config.yaml[/green]")

    @app.command("validate-config", rich_help_panel=HELP_PANEL_CONFIG)
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
        internal consistency. When --data is supplied, also cross-references
        every column name in the config against the actual CSV header.

        Examples:

            phenocluster validate-config -c config.yaml
            phenocluster validate-config -c config.yaml -d data.csv
        """
        show_banner()

        errors: list[str] = []
        warnings_list: list[str] = []

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

            console.print(
                f"[green]OK[/green] Data header: {len(csv_columns)} columns in {data.name}\n"
            )
            _validate_against_data(cfg, csv_columns, errors, warnings_list)

        _display_validation_results(cfg, config, errors, warnings_list, csv_columns)

    @app.command("list-profiles", rich_help_panel=HELP_PANEL_CONFIG)
    def list_profiles_cmd():
        """List available configuration profile templates."""
        table = Table(
            title="Available Profiles",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        for name in list_profiles():
            desc = PROFILES[name].get("_description", "").strip()
            table.add_row(name, desc)
        console.print(table)
        console.print(
            "\n[dim]Use [cyan]phenocluster show-profile <name>[/cyan] to preview a profile.[/dim]"
        )

    @app.command("show-profile", rich_help_panel=HELP_PANEL_CONFIG)
    def show_profile(
        profile: ProfileName = typer.Argument(
            ...,
            help="Profile name",
            case_sensitive=False,
        ),
    ):
        """Print the resolved YAML for a profile with syntax highlighting."""
        profile_name = profile.value if isinstance(profile, Enum) else str(profile)

        with handle_cli_errors(verbose=False):
            with tempfile.NamedTemporaryFile(mode="r", suffix=".yaml", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                create_config_yaml(profile_name, tmp_path)
                yaml_text = tmp_path.read_text()
            finally:
                tmp_path.unlink(missing_ok=True)

            syntax = Syntax(yaml_text, "yaml", theme="ansi_dark", line_numbers=False)
            console.print(
                f"[bold cyan]Profile:[/bold cyan] {profile_name}  "
                f"[dim]({PROFILES[profile_name].get('_description', '').strip()})[/dim]\n"
            )
            console.print(syntax)
