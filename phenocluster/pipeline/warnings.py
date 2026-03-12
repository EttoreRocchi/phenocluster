"""StepMix warning suppression context manager."""

import warnings
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def suppress_stepmix_warnings() -> Iterator[None]:
    """Suppress expected numerical warnings from StepMix EM optimization.

    These occur when random initializations produce degenerate clusters
    (empty clusters, zero variance, 0/0 in categorical probabilities).
    StepMix handles these internally and selects the best valid solution.
    Scoped to a context manager to avoid masking warnings in user code.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"stepmix\.emission\.gaussian",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"stepmix\.emission\.categorical",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"scipy\.special\._logsumexp",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"numpy\.core\._methods",
        )
        warnings.filterwarnings(
            "ignore",
            module=r"stepmix",
            message=r"Initializations did not converge",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"One or more of the test scores are non-finite",
            module=r"sklearn",
        )
        yield
