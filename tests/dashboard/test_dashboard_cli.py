"""Tests for the `phenocluster dashboard` CLI command."""

import builtins
import sys
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from phenocluster.cli.app import app
from phenocluster.dashboard._imports import require_streamlit

runner = CliRunner()


def test_require_streamlit_raises_when_missing(monkeypatch):
    """Removing streamlit from sys.modules and the import path must surface a friendly error."""
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError("No module named 'streamlit'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as exc_info:
        require_streamlit()
    assert "phenocluster[dashboard]" in str(exc_info.value)


def test_dashboard_command_exits_when_streamlit_missing(monkeypatch, tmp_path):
    """`phenocluster dashboard` should exit with code 2 and a helpful message."""

    def _missing_streamlit():
        raise ImportError("The PhenoCluster dashboard requires extra dependencies.")

    monkeypatch.setattr("phenocluster.dashboard._imports.require_streamlit", _missing_streamlit)
    result = runner.invoke(app, ["dashboard", str(tmp_path)])
    assert result.exit_code != 0
    output = result.stdout + (result.stderr or "")
    assert "phenocluster[dashboard]" in output or "extra dependencies" in output


def test_dashboard_command_invokes_subprocess(monkeypatch, tmp_path):
    """When streamlit is present, the command builds the expected subprocess call."""

    monkeypatch.setattr("phenocluster.dashboard._imports.require_streamlit", lambda: MagicMock())
    monkeypatch.setattr(
        "phenocluster.cli.commands.dashboard._streamlit_executable", lambda: "/fake/streamlit"
    )

    captured = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = list(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("phenocluster.cli.commands.dashboard.subprocess.run", fake_run)
    result = runner.invoke(app, ["dashboard", str(tmp_path), "--port", "9999"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert captured["cmd"][0] == "/fake/streamlit"
    assert "run" in captured["cmd"]
    assert "9999" in captured["cmd"]
    assert str(tmp_path) in captured["cmd"]
