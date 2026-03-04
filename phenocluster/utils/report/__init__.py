"""PhenoCluster HTML Report Generator."""

from .generator import generate_html_report
from .sections.external_validation import (
    generate_external_validation_section as _generate_external_validation_section,  # noqa: F401
)

__all__ = ["generate_html_report"]
