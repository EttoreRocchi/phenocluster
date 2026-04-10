"""Command registration helpers for the PhenoCluster CLI."""

import typer

from . import config_cmd, run, version


def register_all(app: typer.Typer) -> None:
    """Attach all command modules to the given Typer app.

    Registration order controls the order of panels in `--help`:
    Configuration first, Pipeline second, Info last.
    """
    config_cmd.register(app)
    run.register(app)
    version.register(app)
