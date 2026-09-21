"""
PhenoCluster Phenotype Ordering
===============================

After the final full-cohort fit the pipeline renumbers the phenotypes by size,
so that phenotype 0 is always the largest and reports read consistently across
runs. That renumbering used to be applied to the label array alone, leaving the
fitted model on its original component order. Any later call to ``predict`` on a
new cohort, in the generalizability stage or in external validation, then
returned ids from the model's own ordering while the derivation labels used the
size ordering: the two label spaces silently disagreed whenever the model's
components did not already come out sorted by size.

:class:`PhenotypeOrderedModel` closes that gap by wrapping the fitted estimator
so every prediction it makes is expressed in the size ordering.
"""

from typing import Any

import numpy as np


def size_order(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return the component ids sorted from the largest phenotype to the smallest.

    Parameters
    ----------
    labels : np.ndarray
        Hard assignments produced by the fitted model.
    n_clusters : int
        Number of components in the model.

    Returns
    -------
    np.ndarray
        ``order`` such that ``order[new_id]`` is the model's own id for the
        phenotype that should be numbered ``new_id``.
    """
    sizes = np.bincount(np.asarray(labels), minlength=n_clusters)
    return np.argsort(-sizes)


def is_identity(order: np.ndarray) -> bool:
    """Return True when ``order`` leaves the component ids unchanged."""
    return bool(np.array_equal(order, np.arange(len(order))))


class PhenotypeOrderedModel:
    """Wrap a fitted mixture model so its predictions use the size ordering.

    The wrapper is transparent: every attribute it does not define itself is
    read from the wrapped estimator, so downstream consumers that inspect
    hyperparameters or call ``score`` keep working. Use :attr:`base_estimator`
    to reach the unwrapped model, for instance to build a fresh estimator of
    the same type.

    Parameters
    ----------
    model
        Fitted StepMix-like estimator exposing ``predict`` and
        ``predict_proba``.
    order : np.ndarray
        Output of :func:`size_order`: ``order[new_id]`` is the wrapped model's
        id for the phenotype numbered ``new_id``.
    """

    def __init__(self, model: Any, order: np.ndarray):
        order = np.asarray(order, dtype=int)
        mapping = np.empty(len(order), dtype=int)
        for new, old in enumerate(order):
            mapping[old] = new
        self.base_estimator = model
        self.phenotype_order = order
        self.phenotype_mapping = mapping

    def predict(self, X) -> np.ndarray:
        """Predict phenotypes, renumbered by size."""
        return self.phenotype_mapping[np.asarray(self.base_estimator.predict(X))]

    def predict_proba(self, X) -> np.ndarray:
        """Predict posterior probabilities, with columns renumbered by size."""
        return np.asarray(self.base_estimator.predict_proba(X))[:, self.phenotype_order]

    def score(self, X, y=None) -> float:
        """Delegate the log-likelihood to the wrapped estimator."""
        return self.base_estimator.score(X)

    def __getattr__(self, name: str) -> Any:
        if name in {"base_estimator", "phenotype_order", "phenotype_mapping"}:
            raise AttributeError(name)
        return getattr(self.base_estimator, name)

    def __repr__(self) -> str:
        return f"PhenotypeOrderedModel({self.base_estimator!r}, order={list(self.phenotype_order)})"


def unwrap_model(model: Any) -> Any:
    """Return the underlying estimator when ``model`` is wrapped, else ``model``."""
    return getattr(model, "base_estimator", model)
