"""Tests for statistical inference utilities -- FDR correction."""

import numpy as np
import pytest


class TestFDRCorrection:
    """Verify custom Benjamini-Hochberg FDR correction."""

    def _fdr(self, p_values):
        from phenocluster.evaluation.stats_utils import apply_fdr_correction

        return apply_fdr_correction(p_values)

    def test_known_bh_result(self):
        """Manual BH calculation for 4 p-values."""
        # p-values sorted: 0.01, 0.03, 0.04, 0.20
        # BH adjusted (rank i, n=4):
        #   rank 1: 0.01 * 4/1 = 0.04
        #   rank 2: 0.03 * 4/2 = 0.06
        #   rank 3: 0.04 * 4/3 = 0.0533
        #   rank 4: 0.20 * 4/4 = 0.20
        # Enforce monotonicity (backward pass): [0.04, 0.0533, 0.0533, 0.20]
        p_values = [0.01, 0.04, 0.03, 0.20]
        q = self._fdr(p_values)

        assert q[0] == pytest.approx(0.04, abs=1e-10)  # 0.01 -> 0.04
        assert q[2] == pytest.approx(4 / 3 * 0.04, abs=1e-10)  # 0.03 -> 0.0533
        assert q[1] == pytest.approx(4 / 3 * 0.04, abs=1e-10)  # 0.04 -> min(0.08, 0.0533) = 0.0533
        assert q[3] == pytest.approx(0.20, abs=1e-10)  # 0.20 -> 0.20

    def test_monotonicity(self):
        """Corrected q-values must preserve rank order of original p-values."""
        p_values = [0.005, 0.01, 0.03, 0.05, 0.10]
        q = self._fdr(p_values)

        for i in range(len(q) - 1):
            assert q[i] <= q[i + 1] + 1e-12

    def test_q_geq_p(self):
        """No q-value should be smaller than the original p-value."""
        p_values = [0.001, 0.01, 0.05, 0.10, 0.50]
        q = self._fdr(p_values)

        for p, qv in zip(p_values, q):
            assert qv >= p - 1e-12

    def test_all_significant(self):
        """All tiny p-values should remain significant after correction."""
        p_values = [0.001, 0.002, 0.003]
        q = self._fdr(p_values)

        for qv in q:
            assert qv < 0.05

    def test_none_nan_preserved(self):
        """None and NaN entries must be preserved unchanged."""
        p_values = [0.01, None, 0.05, np.nan]
        q = self._fdr(p_values)

        assert q[1] is None
        assert q[3] is None or np.isnan(q[3])
        # Valid entries should be corrected
        assert isinstance(q[0], float)
        assert isinstance(q[2], float)

    def test_single_p_value(self):
        """Single p-value -> BH correction leaves it unchanged."""
        q = self._fdr([0.05])
        assert q[0] == pytest.approx(0.05)

    def test_empty_list(self):
        """Empty input -> empty output."""
        q = self._fdr([])
        assert q == []
