"""Unified CLI error handling."""

from contextlib import contextmanager

import pandas as pd
import typer
from rich.traceback import install as install_rich_traceback

from .console import console

FRIENDLY_EXCEPTIONS = (
    ValueError,
    FileNotFoundError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)


def install_tracebacks(verbose: bool) -> None:
    """Install Rich tracebacks with locals when verbose is set."""
    install_rich_traceback(console=console, show_locals=verbose, suppress=[typer])


@contextmanager
def handle_cli_errors(verbose: bool = False):
    """Map known exceptions to friendly messages + consistent exit codes.

    - KeyboardInterrupt -> 130
    - Known user errors (value, file, CSV parse) -> 1 with short message
    - Unknown exceptions -> 1; Rich traceback in verbose mode, short message otherwise
    """
    try:
        yield
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except FRIENDLY_EXCEPTIONS as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print("[dim]Re-run with --verbose for full traceback.[/dim]")
        raise typer.Exit(code=1)
    except Exception as e:
        if verbose:
            console.print_exception(show_locals=True)
        else:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
            console.print("[dim]Re-run with --verbose for full traceback.[/dim]")
        raise typer.Exit(code=1)
