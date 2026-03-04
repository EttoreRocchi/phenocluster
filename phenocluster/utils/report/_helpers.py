"""Shared HTML report formatting helpers."""

import re
from html import escape as _html_escape
from pathlib import Path
from typing import Any, Optional


def safe(value: object) -> str:
    """Escape a value for safe HTML interpolation."""
    return _html_escape(str(value))


def encode_plot_html(html_path: Path) -> Optional[str]:
    """Read an HTML plot file and return its content."""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def format_value(value: Any, precision: int = 3) -> str:
    """Format a value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{precision}f}"
    return str(value)


def format_ci(ci_lower, ci_upper, bracket: str = "[]") -> str:
    """Format a confidence interval, returning 'N/A' if either bound is missing."""
    if not isinstance(ci_lower, (int, float)) or not isinstance(ci_upper, (int, float)):
        return "N/A"
    lo, hi = format_value(ci_lower), format_value(ci_upper)
    if bracket == "()":
        return f"({lo}-{hi})"
    return f"[{lo}, {hi}]"


def fmt_p(p: float) -> str:
    """Format a single p-value, using '<0.0001' for very small values."""
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def empty_section(section_id: str, title: str, message: str) -> str:
    """Return an empty HTML section with a placeholder message."""
    return f"""
    <section id="{section_id}">
        <h2>{title}</h2>
        <p>{message}</p>
    </section>
"""


def extract_plot_height(html_content: str) -> int:
    """Extract figure height from Plotly HTML. Returns height + padding, default 650."""
    match = re.search(r'"height"\s*:\s*(\d+)', html_content)
    if match:
        height = min(max(int(match.group(1)), 200), 2000)
        return height + 80
    return 650


def escape_html(content: str) -> str:
    """Escape HTML for embedding in srcdoc attribute."""
    return (
        content.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def embed_plot(results_dir: Path, plot_name: str) -> str:
    """Embed a single plot by name pattern."""
    plots_dir = results_dir / "plots"
    html_files = list(plots_dir.glob(f"*{plot_name}*.html")) if plots_dir.exists() else []

    if html_files:
        plot_path = html_files[0]
        content = encode_plot_html(plot_path)
        if content:
            height = extract_plot_height(content)
            return f'''
            <div class="plot-container">
                <h4>{plot_path.stem.replace("_", " ").title()}</h4>
                <iframe class="plot-frame" style="height:{height}px"
                        srcdoc="{escape_html(content)}"></iframe>
            </div>
            '''

    return ""


def embed_plots_matching(results_dir: Path, pattern: str) -> str:
    """Embed all plots matching a pattern."""
    plots_dir = results_dir / "plots"
    html_files = sorted(plots_dir.glob(f"*{pattern}*.html")) if plots_dir.exists() else []

    plots = []
    for plot_path in html_files:
        content = encode_plot_html(plot_path)
        if content:
            height = extract_plot_height(content)
            plots.append(f'''
            <div class="plot-container">
                <h4>{plot_path.stem.replace("_", " ").title()}</h4>
                <iframe class="plot-frame" style="height:{height}px"
                        srcdoc="{escape_html(content)}"></iframe>
            </div>
            ''')

    return "\n".join(plots)
