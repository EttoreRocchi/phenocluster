"""Shared banner rendering for the CLI."""

from rich.panel import Panel
from rich.text import Text

from .. import __version__
from .console import console

GITHUB_URL = "https://github.com/EttoreRocchi/phenocluster"
DOCS_URL = "https://ettorerocchi.github.io/phenocluster"


def show_banner(include_links: bool = False) -> None:
    """Render the PhenoCluster banner panel.

    Parameters
    ----------
    include_links : bool
        When True, append GitHub and documentation links (used by `version`).
    """
    content = Text(justify="center")
    content.append("\n")
    content.append("P H E N O", style="bold cyan")
    content.append("  |  ", style="dim cyan")
    content.append("C L U S T E R", style="bold white")
    content.append("\n\n")
    content.append("Clinical Phenotype Discovery Pipeline", style="dim")
    content.append("\n")
    content.append(f"v{__version__}", style="bold cyan" if include_links else "dim cyan")
    content.append("\n")
    if include_links:
        content.append(
            "Click here for the GitHub repository",
            style=f"dim cyan link {GITHUB_URL}",
        )
        content.append("\n")
        content.append(
            "Click here for the documentation",
            style=f"dim cyan link {DOCS_URL}",
        )
        content.append("\n")
    console.print(Panel(content, border_style="cyan", padding=(0, 4), width=min(80, console.width)))
