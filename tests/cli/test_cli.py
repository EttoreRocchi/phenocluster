"""Tests for the CLI module."""

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from phenocluster.cli import (
    _check_columns,
    _validate_against_data,
    _validate_multistate_structure,
    _validate_structure,
    app,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


runner = CliRunner()


def _make_cfg(**overrides):
    """Create a PhenoClusterConfig with defaults suitable for CLI tests."""
    from phenocluster.config import PhenoClusterConfig

    base = {
        "global": {"project_name": "test", "random_state": 42},
        "data": {
            "continuous_columns": ["x1", "x2"],
            "categorical_columns": ["cat1"],
            "split": {"test_size": 0.2},
        },
        "preprocessing": {},
        "model": {
            "n_clusters": 3,
            "selection": {"enabled": True, "min_clusters": 2, "max_clusters": 5},
        },
        "outcome": {"enabled": False},
        "inference": {"enabled": True},
        "logging": {"level": "WARNING", "log_to_file": False},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base:
            base[k].update(v)
        else:
            base[k] = v
    return PhenoClusterConfig.from_dict(base)


class TestValidateStructure:
    def test_valid_config_no_errors(self):
        cfg = _make_cfg()
        errors = []
        _validate_structure(cfg, errors)
        assert errors == []

    def test_no_columns(self):
        cfg = _make_cfg(
            data={"continuous_columns": [], "categorical_columns": [], "split": {"test_size": 0.2}}
        )
        errors = []
        _validate_structure(cfg, errors)
        assert any("continuous_columns or categorical_columns" in e for e in errors)

    def test_bad_test_size(self):
        # DataSplitConfig.__post_init__ rejects invalid test_size,
        # so mock the attribute to test _validate_structure logic
        cfg = _make_cfg()
        cfg.data_split = MagicMock()
        cfg.data_split.test_size = 1.5
        errors = []
        _validate_structure(cfg, errors)
        assert any("test_size" in e for e in errors)

    def test_min_clusters_lt_2(self):
        cfg = _make_cfg(
            model={
                "n_clusters": 3,
                "selection": {"enabled": True, "min_clusters": 1, "max_clusters": 5},
            }
        )
        errors = []
        _validate_structure(cfg, errors)
        assert any("min_clusters" in e for e in errors)

    def test_max_lt_min_clusters(self):
        cfg = _make_cfg(
            model={
                "n_clusters": 3,
                "selection": {"enabled": True, "min_clusters": 5, "max_clusters": 3},
            }
        )
        errors = []
        _validate_structure(cfg, errors)
        assert any("max_clusters" in e for e in errors)

    def test_survival_missing_cols(self):
        cfg = _make_cfg(
            survival={
                "enabled": True,
                "targets": [{"name": "os", "time_column": "", "event_column": "evt"}],
            }
        )
        errors = []
        _validate_structure(cfg, errors)
        assert any("time_column" in e or "event_column" in e for e in errors)

    def test_outcome_enabled_no_columns(self):
        # OutcomeConfig.__post_init__ rejects enabled=True with no columns,
        # so mock the outcome attribute
        cfg = _make_cfg()
        cfg.outcome = MagicMock()
        cfg.outcome.enabled = True
        cfg.outcome.outcome_columns = []
        errors = []
        _validate_structure(cfg, errors)
        assert any("outcome_columns" in e for e in errors)


class TestValidateMultistateStructure:
    def test_disabled_no_errors(self):
        cfg = _make_cfg(multistate={"enabled": False})
        errors = []
        _validate_multistate_structure(cfg, errors)
        assert errors == []

    def test_enabled_no_states(self):
        cfg = _make_cfg()
        cfg.multistate = MagicMock()
        cfg.multistate.enabled = True
        cfg.multistate.states = []
        cfg.multistate.transitions = []
        errors = []
        _validate_multistate_structure(cfg, errors)
        assert any("states" in e for e in errors)

    def test_bad_transition_ref(self):
        # MultistateConfig validates internally, so mock the multistate attribute
        from phenocluster.config import MultistateState, MultistateTransition

        cfg = _make_cfg()
        cfg.multistate = MagicMock()
        cfg.multistate.enabled = True
        cfg.multistate.states = [MultistateState(id=0, name="init", state_type="initial")]
        cfg.multistate.transitions = [MultistateTransition(name="bad", from_state=0, to_state=99)]
        errors = []
        _validate_multistate_structure(cfg, errors)
        assert any("unknown" in e.lower() or "to_state" in e for e in errors)


class TestCheckColumns:
    def test_missing_column_flagged(self):
        errors = []
        _check_columns(["a", "b", "missing"], "section", {"a", "b", "c"}, errors)
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_all_present_no_errors(self):
        errors = []
        _check_columns(["a", "b"], "section", {"a", "b", "c"}, errors)
        assert errors == []


class TestValidateAgainstData:
    def test_missing_continuous_column(self):
        cfg = _make_cfg()
        errors = []
        warnings_list = []
        _validate_against_data(cfg, {"cat1"}, errors, warnings_list)
        assert any("x1" in e or "x2" in e for e in errors)

    def test_overlap_warning(self):
        cfg = _make_cfg(
            data={
                "continuous_columns": ["x1", "overlap"],
                "categorical_columns": ["overlap"],
                "split": {"test_size": 0.2},
            }
        )
        errors = []
        warnings_list = []
        csv_columns = {"x1", "overlap"}
        _validate_against_data(cfg, csv_columns, errors, warnings_list)
        assert any("overlap" in w for w in warnings_list)


class TestRunCommand:
    @patch("phenocluster.cli.PhenoClusterPipeline")
    @patch("phenocluster.cli.PhenoClusterConfig.from_yaml")
    @patch("phenocluster.cli.pd.read_csv")
    def test_happy_path(self, mock_csv, mock_yaml, mock_pipeline_cls, tmp_path):
        import pandas as pd

        # Setup mocks
        mock_csv.return_value = pd.DataFrame({"x": [1, 2, 3]})
        cfg = MagicMock()
        cfg.project_name = "test"
        cfg.output_dir = str(tmp_path)
        cfg.continuous_columns = ["x"]
        cfg.categorical_columns = []
        cfg.outcome_columns = []
        cfg.outcome.enabled = False
        cfg.model_selection.enabled = False
        cfg.n_clusters = 3
        cfg.data_split.test_size = 0.2
        cfg.stability.enabled = False
        cfg.survival.enabled = False
        cfg.multistate.enabled = False
        cfg.inference.enabled = False
        cfg.random_state = 42
        mock_yaml.return_value = cfg

        pipeline_inst = MagicMock()
        pipeline_inst.fit.return_value = {"n_clusters": 3, "n_samples": 100}
        mock_pipeline_cls.return_value = pipeline_inst

        # Create dummy files for --data and --config
        data_file = tmp_path / "data.csv"
        data_file.write_text("x\n1\n2\n3\n")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("global:\n  project_name: test\n")

        result = runner.invoke(app, ["run", "-d", str(data_file), "-c", str(config_file)])
        assert result.exit_code == 0

    @patch("phenocluster.cli.PhenoClusterConfig.from_yaml")
    @patch("phenocluster.cli.pd.read_csv")
    def test_value_error_exits_1(self, mock_csv, mock_yaml, tmp_path):
        mock_csv.side_effect = ValueError("bad data")
        data_file = tmp_path / "data.csv"
        data_file.write_text("x\n1\n")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("x: 1\n")
        result = runner.invoke(app, ["run", "-d", str(data_file), "-c", str(config_file)])
        assert result.exit_code == 1


class TestCreateConfigCommand:
    def test_valid_profile(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        result = runner.invoke(app, ["create-config", "-p", "complete", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_invalid_profile(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        result = runner.invoke(app, ["create-config", "-p", "nonexistent", "-o", str(out)])
        # Typer rejects the unknown enum value with a usage error (exit 2).
        assert result.exit_code != 0


class TestValidateConfigCommand:
    def test_valid_config(self, tmp_path):
        """Create a config YAML with actual columns and validate it."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "global:\n"
            "  project_name: test\n"
            "  random_state: 42\n"
            "data:\n"
            "  continuous_columns: [x1, x2]\n"
            "  categorical_columns: []\n"
            "  split:\n"
            "    test_size: 0.2\n"
            "model:\n"
            "  n_clusters: 3\n"
            "outcome:\n"
            "  enabled: false\n"
            "logging:\n"
            "  level: WARNING\n"
            "  log_to_file: false\n"
        )
        result = runner.invoke(app, ["validate-config", "-c", str(config_file)])
        assert result.exit_code == 0


class TestVersionCommand:
    def test_version_runs(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0


class TestRunFlags:
    @patch("phenocluster.cli.PhenoClusterPipeline")
    @patch("phenocluster.cli.PhenoClusterConfig.from_yaml")
    @patch("phenocluster.cli.pd.read_csv")
    def test_run_quiet_suppresses_banner(self, mock_csv, mock_yaml, mock_pipeline_cls, tmp_path):
        import pandas as pd

        mock_csv.return_value = pd.DataFrame({"x": [1, 2, 3]})
        cfg = MagicMock()
        cfg.project_name = "test"
        cfg.output_dir = str(tmp_path)
        cfg.continuous_columns = ["x"]
        cfg.categorical_columns = []
        cfg.outcome_columns = []
        cfg.outcome.enabled = False
        cfg.model_selection.enabled = False
        cfg.n_clusters = 3
        cfg.data_split.test_size = 0.2
        cfg.stability.enabled = False
        cfg.survival.enabled = False
        cfg.multistate.enabled = False
        cfg.inference.enabled = False
        cfg.random_state = 42
        mock_yaml.return_value = cfg

        pipeline_inst = MagicMock()
        pipeline_inst.fit.return_value = {"n_clusters": 3, "n_samples": 100}
        mock_pipeline_cls.return_value = pipeline_inst

        data_file = tmp_path / "data.csv"
        data_file.write_text("x\n1\n2\n3\n")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("global:\n  project_name: test\n")

        result = runner.invoke(
            app, ["run", "-d", str(data_file), "-c", str(config_file), "--quiet"]
        )
        assert result.exit_code == 0
        # Banner contains the spaced "P H E N O" text; --quiet should suppress it.
        assert "P H E N O" not in result.output

    def test_help_panels_present(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for panel in ("Configuration", "Pipeline", "Info"):
            assert panel in result.output

    def test_run_help_advertises_verbose_quiet(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "--verbose" in output
        assert "--quiet" in output


class TestListProfilesCommand:
    def test_list_profiles_runs(self):
        result = runner.invoke(app, ["list-profiles"])
        assert result.exit_code == 0
        for name in ("descriptive", "complete", "quick"):
            assert name in result.output


class TestShowProfileCommand:
    def test_show_profile_complete(self):
        result = runner.invoke(app, ["show-profile", "complete"])
        assert result.exit_code == 0
        assert "complete" in result.output
        assert "global" in result.output or "project_name" in result.output

    def test_show_profile_unknown(self):
        result = runner.invoke(app, ["show-profile", "nonexistent"])
        assert result.exit_code != 0


class TestBackwardsCompatImports:
    def test_package_level_exports(self):
        """Legacy symbols must still resolve from `phenocluster.cli`."""
        from phenocluster.cli import (  # noqa: F401
            PhenoClusterConfig,
            PhenoClusterPipeline,
            _check_columns,
            _validate_against_data,
            _validate_multistate_structure,
            _validate_structure,
            app,
            main,
            pd,
            typer_click_object,
        )
