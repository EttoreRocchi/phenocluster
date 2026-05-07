"""Tests for the safe CSV reader."""

import os
from pathlib import Path

import pandas as pd
import pytest

from phenocluster.core.exceptions import DataSplitError
from phenocluster.utils.io import safe_read_csv


def test_reads_regular_csv(tmp_path: Path):
    csv = tmp_path / "ok.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv, index=False)
    df = safe_read_csv(csv)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(DataSplitError, match="not found"):
        safe_read_csv(tmp_path / "missing.csv")


def test_oversize_rejected(tmp_path: Path):
    csv = tmp_path / "big.csv"
    pd.DataFrame({"a": list(range(100))}).to_csv(csv, index=False)
    cap = csv.stat().st_size - 1
    with pytest.raises(DataSplitError, match="larger than"):
        safe_read_csv(csv, max_bytes=cap)


def test_directory_rejected(tmp_path: Path):
    with pytest.raises(DataSplitError, match="not a regular file"):
        safe_read_csv(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX FIFO support")
def test_pseudo_file_rejected(tmp_path: Path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    try:
        with pytest.raises(DataSplitError, match="not a regular file"):
            safe_read_csv(fifo)
    finally:
        fifo.unlink()


def test_symlink_to_regular_file_ok(tmp_path: Path):
    src = tmp_path / "real.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)
    link = tmp_path / "link.csv"
    link.symlink_to(src)
    df = safe_read_csv(link)
    assert df["a"].iloc[0] == 1
