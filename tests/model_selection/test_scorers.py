"""Tests for model-selection scorer factories."""

import numpy as np
import pytest

from phenocluster.model_selection.scorers import (
    aic_score,
    bic_score,
    caic_score,
    create_scorer,
    get_all_criteria,
    icl_score,
    relative_entropy_score,
    sabic_score,
)


class _GoodModel:
    """Fake estimator returning fixed criterion values."""

    n_parameters = 12

    def bic(self, X):
        return 1000.0

    def aic(self, X):
        return 800.0

    def caic(self, X):
        return 1100.0

    def sabic(self, X):
        return 950.0

    def score(self, X):
        return -400.0

    def entropy(self, X):
        return 25.0

    def relative_entropy(self, X):
        return 0.7


class _RaisingModel:
    """Fake estimator whose every method raises."""

    def __getattr__(self, item):
        def raiser(*_a, **_k):
            raise RuntimeError("boom")

        if item == "n_parameters":
            raise RuntimeError("no params")
        return raiser


class _NaNModel:
    def bic(self, X):
        return float("nan")

    def aic(self, X):
        return float("inf")

    def caic(self, X):
        return float("nan")

    def sabic(self, X):
        return float("nan")

    def entropy(self, X):
        return float("nan")

    def relative_entropy(self, X):
        return float("nan")


X = np.zeros((5, 2))


class TestICScorerFactory:
    def test_bic_negates(self):
        assert bic_score(_GoodModel(), X) == -1000.0

    def test_aic_negates(self):
        assert aic_score(_GoodModel(), X) == -800.0

    def test_caic_negates(self):
        assert caic_score(_GoodModel(), X) == -1100.0

    def test_sabic_negates(self):
        assert sabic_score(_GoodModel(), X) == -950.0

    def test_non_finite_returns_neg_inf(self):
        assert bic_score(_NaNModel(), X) == -np.inf

    def test_exception_returns_neg_inf(self):
        assert bic_score(_RaisingModel(), X) == -np.inf

    def test_scorer_has_metadata(self):
        assert bic_score.__name__ == "bic_score"
        assert "BIC" in bic_score.__doc__


class TestICLScore:
    def test_formula(self):
        assert icl_score(_GoodModel(), X) == -(1000.0 + 2 * 25.0)

    def test_exception_returns_neg_inf(self):
        assert icl_score(_RaisingModel(), X) == -np.inf

    def test_non_finite_returns_neg_inf(self):
        assert icl_score(_NaNModel(), X) == -np.inf


class TestRelativeEntropyScore:
    def test_passthrough(self):
        assert relative_entropy_score(_GoodModel(), X) == 0.7

    def test_exception_returns_zero(self):
        assert relative_entropy_score(_RaisingModel(), X) == 0.0

    def test_non_finite_returns_zero(self):
        assert relative_entropy_score(_NaNModel(), X) == 0.0


class TestCreateScorer:
    @pytest.mark.parametrize(
        "criterion, fn",
        [
            ("BIC", bic_score),
            ("AIC", aic_score),
            ("CAIC", caic_score),
            ("SABIC", sabic_score),
            ("ICL", icl_score),
            ("ENTROPY", relative_entropy_score),
        ],
    )
    def test_dispatch(self, criterion, fn):
        assert create_scorer(criterion) is fn

    def test_case_insensitive(self):
        assert create_scorer("bic") is bic_score

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            create_scorer("Z")


class TestGetAllCriteria:
    def test_full_success(self):
        out = get_all_criteria(_GoodModel(), X)
        assert out["BIC"] == 1000.0
        assert out["AIC"] == 800.0
        assert out["CAIC"] == 1100.0
        assert out["SABIC"] == 950.0
        assert out["entropy"] == 25.0
        assert out["ICL"] == 1000.0 + 2 * 25.0
        assert out["relative_entropy"] == 0.7
        assert out["ENTROPY"] == 0.7
        assert out["log_likelihood"] == -400.0
        assert out["n_parameters"] == 12

    def test_failure_returns_none_entries(self):
        out = get_all_criteria(_RaisingModel(), X)
        for key in (
            "BIC",
            "AIC",
            "CAIC",
            "SABIC",
            "entropy",
            "ICL",
            "relative_entropy",
            "ENTROPY",
            "n_parameters",
            "log_likelihood",
        ):
            assert out[key] is None

    def test_missing_bic_yields_icl_none(self):
        class PartialModel:
            def entropy(self, X):
                return 10.0

            def bic(self, X):
                raise RuntimeError("nope")

        out = get_all_criteria(PartialModel(), X)
        assert out["ICL"] is None
        assert out["entropy"] == 10.0
