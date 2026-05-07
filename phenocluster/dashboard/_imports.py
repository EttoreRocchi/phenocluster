"""Lazy-import helpers for optional dashboard dependencies."""


def require_streamlit():
    """Return the streamlit module or raise a friendly install hint.

    Streamlit is an optional extra; loading the dashboard subpackage must
    not fail when it is absent. Calling this from the CLI command (or
    from :mod:`phenocluster.dashboard.app`) gives the user an actionable
    error pointing to the right ``pip install`` invocation.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "The PhenoCluster dashboard requires extra dependencies.\n"
            "Install them with:\n"
            "    pip install 'phenocluster[dashboard]'"
        ) from exc
    return st
