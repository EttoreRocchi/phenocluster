"""Tests for training_scope, derivation-only fit and external cohort support."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import (
    ExternalCohortSpec,
    GeneralizabilityConfig,
    PhenoClusterConfig,
    TemporalSpec,
)


def _base_config_dict():
    return {
        "global": {"project_name": "scope", "random_state": 0},
        "data": {
            "continuous_columns": ["c1", "c2"],
            "categorical_columns": [],
            "split": {"test_size": 0.2},
        },
        "preprocessing": {
            "categorical_encoding": {"method": "label"},
            "imputation": {"enabled": False},
            "outlier": {"enabled": False},
        },
        "model": {"n_clusters": 2},
        "outcome": {"enabled": False},
        "logging": {"level": "WARNING", "log_to_file": False},
    }


class TestExternalCohortSpec:
    def test_round_trip_via_yaml_dict(self):
        payload = _base_config_dict()
        payload["generalizability"] = {
            "enabled": True,
            "training_scope": "per_split",
            "external_cohorts": [
                {"path": "/tmp/cohort_B.csv", "label": "hospital_X", "kind": "site"},
                {"path": "/tmp/cohort_2024.csv", "label": "era_2024", "kind": "temporal"},
            ],
        }
        cfg = PhenoClusterConfig.from_dict(payload)
        assert len(cfg.generalizability.external_cohorts) == 2
        assert all(isinstance(c, ExternalCohortSpec) for c in cfg.generalizability.external_cohorts)
        assert cfg.generalizability.external_cohorts[0].label == "hospital_X"
        assert cfg.generalizability.external_cohorts[1].kind == "temporal"

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            ExternalCohortSpec(path="x.csv", label="x", kind="bogus")

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError, match="path"):
            ExternalCohortSpec(path="", label="x")

    def test_enabled_with_only_external_cohorts_ok(self):
        cfg = GeneralizabilityConfig(
            enabled=True,
            external_cohorts=[ExternalCohortSpec(path="x.csv", label="x")],
        )
        assert cfg.enabled is True

    def test_enabled_without_any_source_rejected(self):
        with pytest.raises(ValueError, match="temporal.*multisite.*external_cohorts"):
            GeneralizabilityConfig(enabled=True)


class TestTrainingScopeDefault:
    def test_default_is_per_split(self):
        cfg = GeneralizabilityConfig(
            enabled=True,
            temporal=TemporalSpec(time_column="t", time_cutoff="2020-01-01"),
        )
        assert cfg.training_scope == "per_split"

    def test_global_scope_accepted(self):
        cfg = GeneralizabilityConfig(
            enabled=True,
            training_scope="global",
            temporal=TemporalSpec(time_column="t", time_cutoff="2020-01-01"),
        )
        assert cfg.training_scope == "global"

    def test_unknown_scope_rejected(self):
        with pytest.raises(ValueError, match="training_scope"):
            GeneralizabilityConfig(
                enabled=True,
                training_scope="bogus",
                temporal=TemporalSpec(time_column="t", time_cutoff="2020-01-01"),
            )


class TestStageRoutes:
    """Smoke tests that exercise the GeneralizationStage routing without StepMix."""

    def test_stage_skips_when_disabled(self, tmp_path):
        cfg = PhenoClusterConfig.from_dict(_base_config_dict())
        from phenocluster.pipeline.context import PipelineContext
        from phenocluster.pipeline.stages import GeneralizationStage

        stage = GeneralizationStage(cfg, logging.getLogger("t"))
        ctx = PipelineContext()
        stage.run(ctx, preprocessor=None, feature_selector=None)
        assert ctx.generalizability_results == {}

    def test_external_cohorts_load_path_does_not_crash(self, tmp_path):
        """Stage gracefully reports failure when global preprocessor is missing."""
        ext_path = tmp_path / "external.csv"
        rng = np.random.default_rng(0)
        ext = pd.DataFrame({"c1": rng.normal(size=80), "c2": rng.normal(size=80)})
        ext.to_csv(ext_path, index=False)

        payload = _base_config_dict()
        payload["global"]["output_dir"] = str(tmp_path)
        payload["generalizability"] = {
            "enabled": True,
            "training_scope": "per_split",
            "external_cohorts": [
                {"path": str(ext_path), "label": "site_B", "kind": "external"},
            ],
        }
        cfg = PhenoClusterConfig.from_dict(payload)

        from phenocluster.pipeline.context import PipelineContext
        from phenocluster.pipeline.stages import GeneralizationStage

        stage = GeneralizationStage(cfg, logging.getLogger("t"))
        ctx = PipelineContext()
        ctx.model = MagicMock()
        ctx.labels = np.zeros(10, dtype=int)
        ctx.n_clusters = 2
        stage.run(ctx, preprocessor=None, feature_selector=None)
        # No crash; no reports because the (mock) preprocessor cannot transform.
        assert ctx.generalizability_results == {}
