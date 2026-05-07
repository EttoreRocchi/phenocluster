"""`phenocluster dashboard` - launch the optional Streamlit dashboard."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ..console import console

HELP_PANEL_DASHBOARD = "Dashboard"


def _streamlit_executable() -> Optional[str]:
    """Resolve the streamlit CLI binary, preferring the active venv."""
    venv_streamlit = Path(sys.executable).parent / "streamlit"
    if venv_streamlit.exists():
        return str(venv_streamlit)
    return shutil.which("streamlit")


def register(app: typer.Typer) -> None:
    @app.command("dashboard", rich_help_panel=HELP_PANEL_DASHBOARD)
    def dashboard(
        results_dir: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a previously saved PhenoCluster results directory.",
        ),
        port: int = typer.Option(8501, "--port", help="Local port to bind the dashboard."),
        host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind."),
        headless: bool = typer.Option(
            True, "--headless/--browser", help="Run in headless mode (do not auto-open a browser)."
        ),
    ):
        """Launch an interactive Streamlit dashboard over saved pipeline outputs.

        Requires the optional 'dashboard' extras:
        pip install 'phenocluster\\[dashboard]'

        Example: phenocluster dashboard ./results/
        """
        from ...dashboard._imports import require_streamlit

        try:
            require_streamlit()
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc

        streamlit_bin = _streamlit_executable()
        if streamlit_bin is None:
            console.print(
                "[red]Could not locate the 'streamlit' CLI on PATH.[/red] "
                "Install it via:\n    pip install 'phenocluster[dashboard]'"
            )
            raise typer.Exit(code=2)

        from ... import dashboard as dashboard_pkg

        app_path = Path(dashboard_pkg.__file__).parent / "app.py"
        if not app_path.exists():
            console.print(f"[red]Dashboard entry point not found: {app_path}[/red]")
            raise typer.Exit(code=2)

        cmd = [
            streamlit_bin,
            "run",
            str(app_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true" if headless else "false",
            "--",
            str(results_dir),
        ]
        console.print(f"[bold cyan]Launching dashboard:[/bold cyan] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard interrupted by user.[/yellow]")
