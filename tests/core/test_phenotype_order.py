"""Tests for the size-ordered phenotype wrapper."""

import numpy as np
import pytest

from phenocluster.core.phenotype_order import (
    PhenotypeOrderedModel,
    is_identity,
    size_order,
    unwrap_model,
)


class FakeModel:
    """Minimal mixture-model stand-in with a fixed posterior per row."""

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)
        self.n_components = self.proba.shape[1]

    def predict(self, X):
        return self.proba.argmax(axis=1)

    def predict_proba(self, X):
        return self.proba

    def score(self, X):
        return -1.5

    def get_params(self, deep=False):
        return {"n_components": self.n_components}


#: component 0 holds one row, component 1 holds three, so the size order is [1, 0]
PROBA = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
X = np.zeros((4, 2))


class TestSizeOrder:
    def test_order_puts_the_largest_component_first(self):
        labels = np.array([0, 1, 1, 1])
        assert list(size_order(labels, 2)) == [1, 0]

    def test_identity_is_recognised(self):
        assert is_identity(np.array([0, 1, 2]))
        assert not is_identity(np.array([1, 0, 2]))


class TestPhenotypeOrderedModel:
    def test_predictions_use_the_size_ordering(self):
        model = FakeModel(PROBA)
        wrapped = PhenotypeOrderedModel(model, size_order(model.predict(X), 2))
        assert list(wrapped.predict(X)) == [1, 0, 0, 0]

    def test_posterior_columns_follow_the_labels(self):
        model = FakeModel(PROBA)
        wrapped = PhenotypeOrderedModel(model, size_order(model.predict(X), 2))
        proba = wrapped.predict_proba(X)
        assert proba[0].tolist() == [0.1, 0.9]
        assert proba.argmax(axis=1).tolist() == wrapped.predict(X).tolist()

    def test_same_label_space_on_a_second_cohort(self):
        """The bug this guards: derivation labels reordered, new cohort not."""
        model = FakeModel(PROBA)
        wrapped = PhenotypeOrderedModel(model, size_order(model.predict(X), 2))
        derivation_labels = wrapped.predict(X)
        validation = FakeModel(np.array([[0.95, 0.05], [0.05, 0.95]]))
        wrapped_validation = PhenotypeOrderedModel(validation, wrapped.phenotype_order)
        # a row that the model puts in its component 0 must land on the same
        # phenotype id in both cohorts
        assert wrapped_validation.predict(X)[0] == derivation_labels[0]

    def test_score_and_attributes_pass_through(self):
        model = FakeModel(PROBA)
        wrapped = PhenotypeOrderedModel(model, np.array([1, 0]))
        assert wrapped.score(X) == pytest.approx(-1.5)
        assert wrapped.n_components == 2

    def test_unwrap_returns_the_base_estimator(self):
        model = FakeModel(PROBA)
        wrapped = PhenotypeOrderedModel(model, np.array([1, 0]))
        assert unwrap_model(wrapped) is model
        assert unwrap_model(model) is model

    def test_identity_order_leaves_predictions_untouched(self):
        model = FakeModel(PROBA)
        wrapped = PhenotypeOrderedModel(model, np.array([0, 1]))
        assert wrapped.predict(X).tolist() == model.predict(X).tolist()
        assert wrapped.predict_proba(X).tolist() == model.predict_proba(X).tolist()
