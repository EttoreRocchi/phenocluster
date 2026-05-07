"""Tests for the v0.3.0 splitting strategies (temporal, holdout, LOGO)."""

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import DataSplitConfig
from phenocluster.core.exceptions import DataSplitError, InsufficientDataError
from phenocluster.data.splitting import (
    HoldoutGroupSplitter,
    LeaveOneGroupOutSplitter,
    RandomSplitter,
    TemporalSplitter,
    make_splitter,
)


def _frame_with_dates(n=200, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "admission_date": dates,
            "site": rng.choice(["A", "B", "C", "D"], size=n, p=[0.4, 0.3, 0.2, 0.1]),
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        }
    )


class TestTemporalCutoff:
    def test_cutoff_produces_chronological_split(self):
        df = _frame_with_dates(n=400)
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_cutoff="2018-06-30",
        )
        result = TemporalSplitter(cfg).split(df)
        assert result.partition_kind == "temporal"
        assert (result.train["admission_date"] <= pd.Timestamp("2018-06-30")).all()
        assert (result.test["admission_date"] > pd.Timestamp("2018-06-30")).all()
        assert result.derivation_window == "<=2018-06-30"
        assert result.validation_window == ">2018-06-30"

    def test_cutoff_accepts_pandas_timestamp(self):
        df = _frame_with_dates(n=400)
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_cutoff=pd.Timestamp("2018-06-30"),
        )
        result = TemporalSplitter(cfg).split(df)
        assert result.n_train + result.n_test == 400

    def test_invalid_cutoff_string_rejected_eagerly(self):
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_cutoff="not-a-date",
            min_validation_size=1,
        )
        with pytest.raises(ValueError, match="time_cutoff"):
            TemporalSplitter(cfg)

    def test_empty_validation_raises(self):
        df = _frame_with_dates(n=200)
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_cutoff="2099-01-01",
            min_validation_size=5,
        )
        with pytest.raises(InsufficientDataError):
            TemporalSplitter(cfg).split(df)


class TestTemporalFraction:
    def test_fraction_holds_out_latest_rows(self):
        df = _frame_with_dates(n=200)
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_scheme="fraction",
            time_test_fraction=0.2,
        )
        result = TemporalSplitter(cfg).split(df)
        assert result.n_test == 40
        assert result.test["admission_date"].min() > result.train["admission_date"].max()


class TestTemporalWindows:
    def test_sliding_yields_n_windows(self):
        df = _frame_with_dates(n=300)
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_scheme="sliding",
            n_windows=3,
        )
        results = list(TemporalSplitter(cfg).iter_splits(df))
        assert len(results) == 3
        labels = [r.partition_label for r in results]
        assert labels == [
            "sliding_window_1_of_3",
            "sliding_window_2_of_3",
            "sliding_window_3_of_3",
        ]
        for r in results:
            assert r.test["admission_date"].min() > r.train["admission_date"].max()

    def test_expanding_grows_training_set(self):
        df = _frame_with_dates(n=300)
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="admission_date",
            time_scheme="expanding",
            n_windows=3,
        )
        results = list(TemporalSplitter(cfg).iter_splits(df))
        assert len(results) == 3
        train_sizes = [r.n_train for r in results]
        assert train_sizes == sorted(train_sizes)


class TestHoldoutGroup:
    def test_holdout_one_site(self):
        df = _frame_with_dates(n=200)
        cfg = DataSplitConfig(
            strategy="holdout_group",
            group_column="site",
            holdout_values=["D"],
            min_validation_size=5,
        )
        result = HoldoutGroupSplitter(cfg).split(df)
        assert result.partition_kind == "site"
        assert result.partition_label == "site=D"
        assert (result.test["site"] == "D").all()
        assert "D" not in set(result.train["site"].unique())

    def test_missing_group_values_excluded(self):
        df = _frame_with_dates(n=120)
        df.loc[:9, "site"] = np.nan
        cfg = DataSplitConfig(
            strategy="holdout_group",
            group_column="site",
            holdout_values=["A"],
            min_validation_size=5,
        )
        result = HoldoutGroupSplitter(cfg).split(df)
        assert result.n_train + result.n_test < 120
        assert result.stratification_fallback_reason is not None
        assert "missing" in result.stratification_fallback_reason

    def test_unknown_holdout_value_raises_insufficient(self):
        df = _frame_with_dates(n=100)
        cfg = DataSplitConfig(
            strategy="holdout_group",
            group_column="site",
            holdout_values=["NOPE"],
            min_validation_size=5,
        )
        with pytest.raises(InsufficientDataError):
            HoldoutGroupSplitter(cfg).split(df)


class TestLeaveOneGroupOut:
    def test_yields_one_split_per_group(self):
        df = _frame_with_dates(n=400)
        cfg = DataSplitConfig(
            strategy="leave_one_group_out",
            group_column="site",
            min_validation_size=5,
        )
        results = list(LeaveOneGroupOutSplitter(cfg).iter_splits(df))
        held_out = [r.partition_label for r in results]
        assert held_out == ["site=A", "site=B", "site=C", "site=D"]
        for r in results:
            unique_test = set(r.test["site"].unique())
            assert len(unique_test) == 1

    def test_split_returns_first_cohort(self):
        """`split()` defaults to ``next(iter_splits(df))``: no LSP raise."""
        df = _frame_with_dates(n=400)
        cfg = DataSplitConfig(
            strategy="leave_one_group_out",
            group_column="site",
            min_validation_size=5,
        )
        result = LeaveOneGroupOutSplitter(cfg).split(df)
        assert result.partition_label == "site=A"
        assert set(result.test["site"].unique()) == {"A"}

    def test_skips_groups_below_minimum(self):
        df = _frame_with_dates(n=100)
        cfg = DataSplitConfig(
            strategy="leave_one_group_out",
            group_column="site",
            min_validation_size=200,
        )
        results = list(LeaveOneGroupOutSplitter(cfg).iter_splits(df))
        assert results == []

    def test_single_group_raises(self):
        df = _frame_with_dates(n=50)
        df["site"] = "A"
        cfg = DataSplitConfig(
            strategy="leave_one_group_out",
            group_column="site",
            min_validation_size=1,
        )
        with pytest.raises(DataSplitError):
            list(LeaveOneGroupOutSplitter(cfg).iter_splits(df))


class TestFactory:
    @pytest.mark.parametrize(
        "strategy,extra,expected_cls",
        [
            ("random", {}, RandomSplitter),
            (
                "temporal",
                {"time_column": "admission_date", "time_cutoff": "2018-06-30"},
                TemporalSplitter,
            ),
            (
                "holdout_group",
                {"group_column": "site", "holdout_values": ["A"]},
                HoldoutGroupSplitter,
            ),
            (
                "leave_one_group_out",
                {"group_column": "site"},
                LeaveOneGroupOutSplitter,
            ),
        ],
    )
    def test_factory_dispatches(self, strategy, extra, expected_cls):
        cfg = DataSplitConfig(strategy=strategy, **extra)
        assert isinstance(make_splitter(cfg), expected_cls)


class TestConfigValidation:
    def test_temporal_requires_time_column(self):
        with pytest.raises(ValueError, match="time_column"):
            DataSplitConfig(strategy="temporal")

    def test_holdout_requires_group_and_values(self):
        with pytest.raises(ValueError, match="group_column"):
            DataSplitConfig(strategy="holdout_group")
        with pytest.raises(ValueError, match="holdout_values"):
            DataSplitConfig(strategy="holdout_group", group_column="site")

    def test_logo_requires_group_column(self):
        with pytest.raises(ValueError, match="group_column"):
            DataSplitConfig(strategy="leave_one_group_out")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            DataSplitConfig(strategy="bogus")

    def test_temporal_fraction_bounds(self):
        with pytest.raises(ValueError):
            DataSplitConfig(
                strategy="temporal",
                time_column="t",
                time_scheme="fraction",
                time_test_fraction=1.5,
            )

    def test_temporal_windows_require_n_windows(self):
        with pytest.raises(ValueError, match="n_windows"):
            DataSplitConfig(
                strategy="temporal",
                time_column="t",
                time_scheme="sliding",
            )


class TestTemporalBoundaryTies:
    def test_fraction_split_no_shared_date(self):
        df = pd.DataFrame(
            {
                "t": pd.to_datetime(["2020-01-01"] * 5 + ["2020-06-01"] * 5 + ["2020-12-31"] * 5),
                "x": np.arange(15),
            }
        )
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="t",
            time_scheme="fraction",
            time_test_fraction=0.4,
            min_validation_size=1,
        )
        result = TemporalSplitter(cfg).split(df)
        train_dates = set(df.iloc[result.train_indices]["t"].dt.date)
        test_dates = set(df.iloc[result.test_indices]["t"].dt.date)
        assert train_dates.isdisjoint(test_dates)

    def test_sliding_window_no_shared_date(self):
        df = pd.DataFrame(
            {
                "t": pd.to_datetime(["2018-01-01"] * 10)
                .append(pd.to_datetime(["2019-01-01"] * 10))
                .append(pd.to_datetime(["2020-01-01"] * 10)),
                "x": np.arange(30),
            }
        )
        cfg = DataSplitConfig(
            strategy="temporal",
            time_column="t",
            time_scheme="sliding",
            n_windows=2,
            min_validation_size=1,
        )
        for split in TemporalSplitter(cfg).iter_splits(df):
            train_dates = set(df.iloc[split.train_indices]["t"].dt.date)
            test_dates = set(df.iloc[split.test_indices]["t"].dt.date)
            assert train_dates.isdisjoint(test_dates)
