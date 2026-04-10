"""Typer application and entry point."""

import typer

from .commands import register_all
from .errors import install_tracebacks

app = typer.Typer(
    name="phenocluster",
    help="PhenoCluster - Clinical Phenotype Discovery Pipeline",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Install Rich tracebacks once at import time. `run` upgrades to show_locals
# when `--verbose` is passed.
install_tracebacks(verbose=False)

register_all(app)


def main() -> None:
    """Main entry point for the `phenocluster` console script."""
    app()


# sphinx-click bridge: expose Typer app as a Click group for documentation generation
typer_click_object = typer.main.get_command(app)


if __name__ == "__main__":
    main()
