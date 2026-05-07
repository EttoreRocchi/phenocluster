"""Smoke tests for individual report section renderers."""

from phenocluster.utils.report.sections.classification_quality import (
    generate_classification_quality_section,
)
from phenocluster.utils.report.sections.data_quality import generate_data_quality_section
from phenocluster.utils.report.sections.feature_importance import (
    generate_feature_importance_section,
)
from phenocluster.utils.report.sections.model_selection import generate_model_selection_section
from phenocluster.utils.report.sections.multistate import generate_multistate_section
from phenocluster.utils.report.sections.outcomes import generate_outcomes_section
from phenocluster.utils.report.sections.stability import generate_stability_section
from phenocluster.utils.report.sections.survival import generate_survival_section
from phenocluster.utils.report.sections.validation import generate_validation_section


def _outcome_results():
    return {
        "full_cohort": {
            "outcome1": {
                "0": {
                    "OR": 1.0,
                    "CI_lower": 1.0,
                    "CI_upper": 1.0,
                    "p_value": None,
                    "n_positive": 5,
                    "n_total": 20,
                },
                "1": {
                    "OR": 2.4,
                    "CI_lower": 1.2,
                    "CI_upper": 4.5,
                    "p_value": 0.01,
                    "n_positive": 12,
                    "n_total": 20,
                },
            }
        },
        "train": {
            "outcome1": {
                "0": {"OR": 1.0, "CI_lower": 1.0, "CI_upper": 1.0, "p_value": None},
                "1": {"OR": 2.0, "CI_lower": 1.0, "CI_upper": 4.0, "p_value": 0.05},
            }
        },
        "test": {
            "outcome1": {
                "0": {"OR": 1.0, "CI_lower": 1.0, "CI_upper": 1.0, "p_value": None},
                "1": {"OR": 2.6, "CI_lower": 1.0, "CI_upper": 5.0, "p_value": 0.04},
            }
        },
    }


def _survival_results():
    return {
        "mortality": {
            "median_survival": {0: 24.0, 1: 12.0},
            "survival_data": {
                0: {"n_patients": 30, "n_events": 10},
                1: {"n_patients": 20, "n_events": 12},
            },
            "comparison": {
                "1_vs_0": {
                    "HR": 1.8,
                    "CI_lower": 0.9,
                    "CI_upper": 3.5,
                    "p_value": 0.04,
                    "q_value": 0.05,
                },
            },
            "logrank_p_value": 0.03,
        }
    }


def _multistate_results():
    return {
        "transition_results": {
            "alive_to_dead": {
                "from_state": 0,
                "to_state": 1,
                "n_events": 12,
                "n_at_risk": 50,
                "phenotype_effects": {
                    "0": {"HR": 1.0, "p_value": None},
                    "1": {
                        "HR": 2.0,
                        "CI_lower": 1.1,
                        "CI_upper": 3.4,
                        "p_value": 0.03,
                        "q_value": 0.04,
                    },
                },
            }
        },
        "pathway_results": [
            {
                "pathway": (0, 1),
                "state_names": ["alive", "dead"],
                "counts_by_phenotype": {"0": 5, "1": 7},
                "total_count": 12,
            }
        ],
    }


def _validation_payload():
    return {
        "validation_report": {
            "train_log_likelihood": -2.0,
            "test_log_likelihood": -2.1,
            "n_train": 80,
            "n_test": 20,
            "train_cluster_pcts": {"0": 60.0, "1": 40.0},
            "test_cluster_pcts": {"0": 55.0, "1": 45.0},
            "train_cluster_sizes": {"0": 48, "1": 32},
            "test_cluster_sizes": {"0": 11, "1": 9},
        },
        "outcome_results": _outcome_results(),
    }


class TestValidationSection:
    def test_full_payload(self):
        html = generate_validation_section(_validation_payload())
        assert "Internal Validation" in html
        assert "Phenotype" in html

    def test_empty_returns_placeholder(self):
        html = generate_validation_section({})
        assert "Validation metrics not available" in html


class TestSurvivalSection:
    def test_full_payload(self, tmp_path):
        html = generate_survival_section({"survival_results": _survival_results()}, tmp_path)
        assert "Survival Analysis" in html
        assert "Mortality" in html

    def test_empty_returns_placeholder(self, tmp_path):
        html = generate_survival_section({}, tmp_path)
        assert "Survival" in html


class TestMultistateSection:
    def test_full_payload(self, tmp_path):
        html = generate_multistate_section({"multistate_results": _multistate_results()}, tmp_path)
        assert "Multistate" in html or "transition" in html.lower()

    def test_empty_returns_placeholder(self, tmp_path):
        html = generate_multistate_section({}, tmp_path)
        assert isinstance(html, str)


class TestOutcomesSection:
    def test_full_payload(self, tmp_path):
        html = generate_outcomes_section({"outcome_results": _outcome_results()}, tmp_path)
        assert "Outcome" in html or "Phenotype" in html

    def test_empty_returns_placeholder(self, tmp_path):
        html = generate_outcomes_section({}, tmp_path)
        assert isinstance(html, str)


class TestFeatureImportanceSection:
    def test_full_payload(self):
        data = {
            "feature_importance": {
                "top_features_per_cluster": {
                    "0": [
                        {
                            "feature": "x1",
                            "importance": 0.8,
                            "type": "continuous",
                            "direction": "higher",
                            "effect_size": 0.6,
                        },
                        {
                            "feature": "cat1",
                            "importance": 0.5,
                            "type": "categorical",
                            "dominant_category": "A",
                            "dominant_ratio": 2.5,
                        },
                    ],
                    "1": [
                        {
                            "feature": "x2",
                            "importance": 0.4,
                            "type": "continuous",
                            "direction": "lower",
                            "effect_size": 0.3,
                        }
                    ],
                }
            }
        }
        html = generate_feature_importance_section(data)
        assert "Phenotype" in html
        assert "x1" in html
        assert "cat1" in html

    def test_grouped_features(self):
        data = {
            "feature_importance": {
                "top_features_per_cluster": {
                    "0": [
                        {
                            "feature": "x1",
                            "group": "vitals",
                            "importance": 0.7,
                            "type": "continuous",
                            "direction": "higher",
                            "effect_size": 0.5,
                        }
                    ]
                }
            }
        }
        html = generate_feature_importance_section(data)
        assert "vitals" in html

    def test_empty_returns_placeholder(self):
        html = generate_feature_importance_section({})
        assert "not available" in html

    def test_no_top_features_renders_message(self):
        data = {"feature_importance": {"top_features_per_cluster": {}}}
        html = generate_feature_importance_section(data)
        assert isinstance(html, str)


class TestClassificationQualitySection:
    def test_full_payload(self):
        data = {
            "classification_quality": {
                "AvePP": {0: 0.95, 1: 0.90},
                "n_per_cluster": {0: 30, 1: 20},
                "min_AvePP": 0.90,
                "mean_AvePP": 0.93,
            }
        }
        html = generate_classification_quality_section(data)
        assert isinstance(html, str)

    def test_empty_returns_placeholder(self):
        html = generate_classification_quality_section({})
        assert isinstance(html, str)


class TestStabilitySection:
    def test_full_payload(self, tmp_path):
        data = {
            "stability_results": {
                "mean_consensus": 0.85,
                "ci_95_lower": 0.80,
                "ci_95_upper": 0.90,
                "n_runs": 10,
                "n_valid": 10,
            }
        }
        html = generate_stability_section(data, tmp_path)
        assert isinstance(html, str)

    def test_empty_returns_placeholder(self, tmp_path):
        html = generate_stability_section({}, tmp_path)
        assert isinstance(html, str)


class TestModelSelectionSection:
    def test_full_payload(self, tmp_path):
        data = {
            "model_selection_summary": {
                "best_n_clusters": 3,
                "criterion_used": "BIC",
                "best_criterion_value": -120.5,
            }
        }
        html = generate_model_selection_section(data, tmp_path)
        assert isinstance(html, str)


class TestDataQualitySection:
    def test_full_payload(self):
        data = {
            "data_quality": {
                "n_samples": 100,
                "n_features": 5,
                "n_missing": 3,
                "missing_pct": 0.6,
            }
        }
        html = generate_data_quality_section(data)
        assert isinstance(html, str)

    def test_empty_returns_placeholder(self):
        html = generate_data_quality_section({})
        assert isinstance(html, str)
