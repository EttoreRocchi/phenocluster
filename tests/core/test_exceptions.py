"""Tests for the custom exception hierarchy."""

import pytest

from phenocluster.core.exceptions import (
    DataSplitError,
    FeatureSelectionError,
    InsufficientDataError,
    ModelNotFittedError,
    PhenoClusterError,
)


class TestPhenoClusterError:
    def test_message_only(self):
        err = PhenoClusterError("boom")
        assert str(err) == "boom"
        assert err.message == "boom"
        assert err.details == {}

    def test_with_details(self):
        err = PhenoClusterError("boom", details={"k": "v"})
        rendered = str(err)
        assert "boom" in rendered
        assert "k" in rendered


class TestModelNotFittedError:
    def test_message_includes_model_name(self):
        err = ModelNotFittedError("LassoSelector")
        assert "LassoSelector" in str(err)
        assert err.model_name == "LassoSelector"

    def test_default_name(self):
        err = ModelNotFittedError()
        assert "Model" in str(err)


class TestFeatureSelectionError:
    def test_method_recorded(self):
        err = FeatureSelectionError("oops", method="lasso")
        assert err.method == "lasso"
        assert "lasso" in str(err)

    def test_features_recorded(self):
        err = FeatureSelectionError("missing cols", features=["a", "b"])
        assert err.features == ["a", "b"]

    def test_no_extras(self):
        err = FeatureSelectionError("plain")
        assert err.method is None
        assert err.features is None
        assert str(err) == "plain"


class TestDataSplitError:
    def test_n_samples_recorded(self):
        err = DataSplitError("bad split", n_samples=10)
        assert err.n_samples == 10
        assert "10" in str(err)

    def test_split_sizes_recorded(self):
        err = DataSplitError("bad split", split_sizes={"train": 8, "test": 2})
        assert err.split_sizes == {"train": 8, "test": 2}

    def test_no_extras(self):
        err = DataSplitError("plain")
        assert err.n_samples is None
        assert err.split_sizes is None


class TestInsufficientDataError:
    def test_records_n_and_min(self):
        err = InsufficientDataError("too few", n_samples=3, min_required=10)
        rendered = str(err)
        assert "3" in rendered
        assert "10" in rendered

    def test_no_extras(self):
        err = InsufficientDataError("plain")
        assert str(err) == "plain"


class TestRaising:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(PhenoClusterError):
            raise FeatureSelectionError("subclass")
