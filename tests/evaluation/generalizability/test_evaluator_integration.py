"""End-to-end regression test for `GeneralizabilityEvaluator.evaluate_cohort`.

Pre-fix, ``_compute_outcome_concordance`` called non-existent
``.analyze(...)`` methods on ``OutcomeAnalyzer`` and ``SurvivalAnalyzer``;
the resulting ``AttributeError`` was masked by a generic ``except
Exception`` so the cohort report silently lacked the ``outcomes`` and
``survival`` blocks. Unit tests for the helper functions
(``compare_outcomes`` / ``compare_survival``) passed but the bug shipped.

These tests exercise the public ``evaluate_cohort`` path with mocked
analyzers and assert the concordance keys are populated.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@dataclass
class _SurvivalTargetStub:
    """Stand-in for ``SurvivalTarget`` (with the same field names)."""

    name: str
    time_column: str
    event_column: str


def _outcome_block(point: float = 1.4) -> dict:
    log_p = float(np.log(point))
    se = 0.2
    return {
        "OR": float(point),
        "CI_lower": float(np.exp(log_p - 1.96 * se)),
        "CI_upper": float(np.exp(log_p + 1.96 * se)),
        "p_value": 0.01,
    }


def _hr_block(point: float = 1.5) -> dict:
    log_p = float(np.log(point))
    se = 0.18
    return {
        "HR": float(point),
        "CI_lower": float(np.exp(log_p - 1.96 * se)),
        "CI_upper": float(np.exp(log_p + 1.96 * se)),
        "p_value": 0.02,
    }


def _make_config():
    """Minimal stub config exposing the attributes the evaluator reads."""
    cfg = MagicMock()
    cfg.continuous_columns = ["c1", "c2"]
    cfg.categorical_columns = []
    cfg.outcome_columns = ["death"]
    cfg.outcome.enabled = True
    cfg.survival.enabled = True
    cfg.survival.targets = [_SurvivalTargetStub(name="os", time_column="t", event_column="e")]
    cfg.reference_phenotype = MagicMock(id=0)
    cfg.random_state = 0
    cfg.generalizability.refit = False
    cfg.generalizability.calibration.enabled = False
    cfg.generalizability.drift.enabled = False
    cfg.generalizability.outcome_concordance.enabled = True
    # Logger setup reads real strings from these fields.
    cfg.logging.level = "WARNING"
    cfg.logging.log_format = "simple"
    cfg.logging.log_to_file = False
    cfg.logging.log_file = None
    return cfg


def _make_model(n_classes: int = 2):
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.zeros(len(X), dtype=int))
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.tile(np.array([0.6, 0.4][:n_classes]), (len(X), 1))
    )
    model.score = MagicMock(return_value=-100.0)
    return model


def _make_preprocessor(df: pd.DataFrame) -> MagicMock:
    """Identity preprocessor: returns the cohort frame and a feature matrix."""

    def _transform_preprocess(frame):
        feat = frame[["c1", "c2"]].to_numpy(dtype=float)
        return frame, feat

    pre = MagicMock()
    pre.transform_impute = MagicMock(side_effect=lambda x: x)
    pre.transform_outliers = MagicMock(side_effect=lambda x: x)
    pre.transform_preprocess = MagicMock(side_effect=_transform_preprocess)
    pre.get_feature_matrix = MagicMock(
        side_effect=lambda frame, *_args, **_kw: frame[["c1", "c2"]].to_numpy(dtype=float)
    )
    return pre


@pytest.fixture
def patched_analyzers(monkeypatch):
    """Replace OutcomeAnalyzer / SurvivalAnalyzer with mocks exposing the
    methods the post-fix evaluator calls. If the evaluator regresses to
    ``.analyze(...)``, the mocks raise AttributeError which the regression
    test asserts must NOT be silently swallowed.
    """
    import logging as _logging

    # Stub out the project logger factory so it doesn't try to read real
    # paths from the MagicMock config.
    eval_mod = importlib.import_module("phenocluster.evaluation.generalizability.evaluator")
    monkeypatch.setattr(
        eval_mod, "get_logger", lambda *a, **kw: _logging.getLogger("test.evaluator")
    )

    outcome_mod = importlib.import_module("phenocluster.evaluation.outcome_analysis")
    survival_mod = importlib.import_module("phenocluster.evaluation.survival")

    outcome_instance = MagicMock()
    outcome_instance.analyze_outcomes = MagicMock(
        return_value={"death": {0: _outcome_block(1.4), 1: _outcome_block(0.7)}}
    )
    # Sentinel: any call to `.analyze(...)` (the broken pre-fix path)
    # must blow up so the test catches a regression.
    del outcome_instance.analyze
    OutcomeStub = MagicMock(return_value=outcome_instance)
    monkeypatch.setattr(outcome_mod, "OutcomeAnalyzer", OutcomeStub)

    survival_instance = MagicMock()
    survival_instance.analyze_survival = MagicMock(
        return_value={
            0: _hr_block(1.5),
            1: _hr_block(0.8),
        }
    )
    del survival_instance.analyze
    SurvivalStub = MagicMock(return_value=survival_instance)
    monkeypatch.setattr(survival_mod, "SurvivalAnalyzer", SurvivalStub)

    return {
        "outcome": outcome_instance,
        "outcome_cls": OutcomeStub,
        "survival": survival_instance,
        "survival_cls": SurvivalStub,
    }


def test_evaluate_cohort_invokes_correct_analyzer_methods(patched_analyzers):
    """Critical regression test for finding #1.

    Before the fix, ``_compute_outcome_concordance`` called
    ``OutcomeAnalyzer.analyze`` and ``SurvivalAnalyzer.analyze``, methods
    that don't exist. The bare ``except Exception`` swallowed the
    AttributeError and produced an empty concordance dict.
    """
    from phenocluster.evaluation.generalizability.evaluator import (
        GeneralizabilityEvaluator,
    )

    cfg = _make_config()
    derivation_df = pd.DataFrame(
        {"c1": np.linspace(0, 1, 20), "c2": np.linspace(1, 2, 20), "t": 5.0, "e": 1}
    )
    derivation_labels = np.array([0, 1] * 10)
    deriv_outcomes = {"full_cohort": {"death": {0: _outcome_block(1.5), 1: _outcome_block(0.7)}}}
    deriv_survival = {"os": {0: _hr_block(1.6), 1: _hr_block(0.9)}}

    val_df = pd.DataFrame(
        {"c1": np.linspace(0, 1, 30), "c2": np.linspace(1, 2, 30), "t": 5.0, "e": 1}
    )

    evaluator = GeneralizabilityEvaluator(
        cfg,
        derivation_labels=derivation_labels,
        derivation_outcomes=deriv_outcomes,
        derivation_survival=deriv_survival,
        derivation_df=derivation_df,
        model=_make_model(),
        preprocessor=_make_preprocessor(val_df),
        feature_selector=None,
        n_clusters=2,
    )

    report = evaluator.evaluate_cohort(val_df, label="t1", kind="temporal")

    assert report is not None
    # The two correct methods were invoked at least once.
    patched_analyzers["outcome"].analyze_outcomes.assert_called()
    patched_analyzers["survival"].analyze_survival.assert_called()
    # And concordance keys are populated, not silently empty.
    assert "outcomes" in report.outcome_concordance
    assert "survival" in report.outcome_concordance


def test_evaluate_cohort_passes_target_columns_to_survival(patched_analyzers):
    """The fix routes ``target.time_column`` / ``target.event_column`` directly
    rather than passing the whole target object; confirm the kwargs match.
    """
    from phenocluster.evaluation.generalizability.evaluator import (
        GeneralizabilityEvaluator,
    )

    cfg = _make_config()
    val_df = pd.DataFrame(
        {"c1": np.linspace(0, 1, 30), "c2": np.linspace(1, 2, 30), "t": 5.0, "e": 1}
    )
    evaluator = GeneralizabilityEvaluator(
        cfg,
        derivation_labels=np.array([0, 1] * 10),
        derivation_outcomes={"full_cohort": {"death": {0: _outcome_block(1.5)}}},
        derivation_survival={"os": {0: _hr_block(1.6)}},
        derivation_df=pd.DataFrame(
            {"c1": np.linspace(0, 1, 20), "c2": np.linspace(1, 2, 20), "t": 5.0, "e": 1}
        ),
        model=_make_model(),
        preprocessor=_make_preprocessor(val_df),
        feature_selector=None,
        n_clusters=2,
    )
    evaluator.evaluate_cohort(val_df, label="t1", kind="temporal")

    call = patched_analyzers["survival"].analyze_survival.call_args
    # `analyze_survival` must be called with explicit time_column / event_column kwargs.
    assert call.kwargs["time_column"] == "t"
    assert call.kwargs["event_column"] == "e"


def test_evaluate_cohorts_method_is_gone():
    """`evaluate_cohorts` and `build_report` were deleted (finding #6/#18).
    Their removal is a public-API contract change; this guards it.
    """
    from phenocluster.evaluation.generalizability.evaluator import (
        GeneralizabilityEvaluator,
    )

    assert not hasattr(GeneralizabilityEvaluator, "evaluate_cohorts")
    assert not hasattr(GeneralizabilityEvaluator, "build_report")
