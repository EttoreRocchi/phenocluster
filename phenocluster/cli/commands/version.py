"""`phenocluster version` — show version banner with links."""

import typer

from ..banner import show_banner

HELP_PANEL_INFO = "Info"


def register(app: typer.Typer) -> None:
    @app.command("version", rich_help_panel=HELP_PANEL_INFO)
    def version():
        """Show version information."""
        show_banner(include_links=True)
