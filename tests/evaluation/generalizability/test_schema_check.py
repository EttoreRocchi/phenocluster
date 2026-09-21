"""Tests for the validation cohort schema audit."""

import numpy as np
import pandas as pd

from phenocluster.config import PhenoClusterConfig
from phenocluster.data.preprocessor import DataPreprocessor
from phenocluster.evaluation.generalizability.schema_check import check_cohort_schema


def _config(tmp_path):
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "schema", "output_dir": str(tmp_path), "random_state": 0},
            "data": {
                "continuous_columns": ["age"],
                "categorical_columns": ["grade", "flag"],
                "split": {},
            },
            "preprocessing": {"imputation": {"enabled": False}, "outlier": {"enabled": False}},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": True, "outcome_columns": ["death"]},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


def _fitted_preprocessor(tmp_path, derivation):
    cfg = _config(tmp_path)
    preprocessor = DataPreprocessor(cfg)
    preprocessor.fit_preprocessor(derivation)
    return cfg, preprocessor


DERIVATION = pd.DataFrame(
    {
        "age": [50.0, 60.0, 70.0, 80.0],
        "grade": ["I", "II", "I", "II"],
        "flag": [0.0, 1.0, 1.0, np.nan],
        "death": [0, 1, 0, 1],
    }
)


def _check(tmp_path, cohort):
    cfg, preprocessor = _fitted_preprocessor(tmp_path, DERIVATION)
    return check_cohort_schema(
        cohort,
        preprocessor=preprocessor,
        continuous_columns=cfg.continuous_columns,
        categorical_columns=cfg.categorical_columns,
        outcome_columns=cfg.outcome_columns,
    )


class TestSchemaCheck:
    def test_aligned_cohort_has_no_findings(self, tmp_path):
        cohort = pd.DataFrame(
            {"age": [55.0], "grade": ["I"], "flag": [1], "death": [0]},
        )
        result = _check(tmp_path, cohort)
        assert result["warnings"] == []
        assert result["unseen_categories"] == {}

    def test_missing_feature_column_is_reported(self, tmp_path):
        cohort = pd.DataFrame({"age": [55.0], "flag": [1], "death": [0]})
        result = _check(tmp_path, cohort)
        assert result["missing_features"] == ["grade"]
        assert any("feature column" in w for w in result["warnings"])

    def test_missing_outcome_column_is_reported(self, tmp_path):
        cohort = pd.DataFrame({"age": [55.0], "grade": ["I"], "flag": [1]})
        result = _check(tmp_path, cohort)
        assert result["missing_outcomes"] == ["death"]
        assert any("outcome column" in w for w in result["warnings"])

    def test_entirely_missing_column_is_reported(self, tmp_path):
        cohort = pd.DataFrame(
            {"age": [55.0], "grade": [np.nan], "flag": [1], "death": [0]},
        )
        result = _check(tmp_path, cohort)
        assert result["all_missing_features"] == ["grade"]

    def test_unseen_category_is_reported_with_share(self, tmp_path):
        cohort = pd.DataFrame(
            {
                "age": [55.0, 65.0, 75.0, 85.0],
                "grade": ["III", "III", "III", "I"],
                "flag": [1, 0, 1, 0],
                "death": [0, 1, 0, 1],
            }
        )
        result = _check(tmp_path, cohort)
        assert result["unseen_categories"]["grade"]["labels"] == ["III"]
        assert result["unseen_categories"]["grade"]["n_rows"] == 3
        assert result["unseen_categories"]["grade"]["fraction"] == 0.75
        assert any("same coding" in w for w in result["warnings"])

    def test_integer_codes_are_not_flagged_against_float_training(self, tmp_path):
        """The 0/1 column is float in the derivation CSV and integer here."""
        cohort = pd.DataFrame(
            {"age": [55.0, 65.0], "grade": ["I", "II"], "flag": [0, 1], "death": [0, 1]},
        )
        result = _check(tmp_path, cohort)
        assert "flag" not in result["unseen_categories"]
