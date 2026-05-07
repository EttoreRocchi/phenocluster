"""Tests for GeneralizabilityConfig YAML round-tripping and pipeline gating."""

import pytest

from phenocluster.config import (
    GeneralizabilityConfig,
    MultiSiteSpec,
    PhenoClusterConfig,
)


def _base_config_dict():
    return {
        "global": {"project_name": "t", "random_state": 42},
        "data": {
            "continuous_columns": ["c1"],
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


def test_multisite_logo_round_trip():
    payload = _base_config_dict()
    payload["generalizability"] = {
        "enabled": True,
        "multisite": {"site_column": "center", "scheme": "logo", "min_site_size": 50},
    }
    cfg = PhenoClusterConfig.from_dict(payload)
    assert isinstance(cfg.generalizability.multisite, MultiSiteSpec)
    assert cfg.generalizability.multisite.site_column == "center"
    assert cfg.generalizability.multisite.scheme == "logo"
    assert cfg.generalizability.multisite.min_site_size == 50


def test_generalizability_enabled_requires_one_of_temporal_or_multisite():
    with pytest.raises(ValueError, match="temporal.*multisite"):
        GeneralizabilityConfig(enabled=True)


def test_generalization_stage_no_op_when_disabled():
    """Stage should silently return when generalizability.enabled is False."""
    cfg = PhenoClusterConfig.from_dict(_base_config_dict())
    import logging

    from phenocluster.pipeline.context import PipelineContext
    from phenocluster.pipeline.stages import GeneralizationStage

    stage = GeneralizationStage(cfg, logging.getLogger("test"))
    ctx = PipelineContext()
    stage.run(ctx, preprocessor=None, feature_selector=None)
    assert ctx.generalizability_results == {}
