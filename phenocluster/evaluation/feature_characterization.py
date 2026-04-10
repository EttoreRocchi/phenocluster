"""Descriptive feature characterization using effect sizes."""

import fnmatch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..config import FeatureCharacterizationConfig, PhenoClusterConfig
from ..evaluation.stats_utils import apply_fdr_correction
from ..utils.logging import get_logger


def _assign_feature_group(name: str, config: FeatureCharacterizationConfig) -> str:
    """Assign a feature to a group based on custom_groups or prefix."""
    if config.custom_groups is None:
        if config.prefix_separator in name:
            return name.split(config.prefix_separator)[0]
        return "Other"
    for group_name, patterns in config.custom_groups.items():
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return group_name
    if config.prefix_separator in name:
        return name.split(config.prefix_separator)[0]
    return "Other"


class FeatureCharacterizer:
    """Compute effect sizes (Hedges' g*, Cramer's V) per cluster."""

    def __init__(self, config: PhenoClusterConfig, n_clusters: int):
        self.config = config
        self.n_clusters = n_clusters
        self.logger = get_logger("evaluation", config)

    def compute_feature_importance(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Dict]:
        """
        Compute feature importance / effect sizes for each cluster.

        Continuous features are characterised with Hedges' g\\* (Welch-
        consistent) against the pooled complement; categorical features
        use Cramer's V. When ``config.inference.fdr_correction`` is
        enabled, Benjamini-Hochberg FDR is applied across all tests.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe containing the columns declared in
            ``config.continuous_columns`` and ``config.categorical_columns``.
        labels : np.ndarray
            Hard cluster assignments aligned with the rows of ``df``.

        Returns
        -------
        Dict[str, Dict]
            Dictionary with keys ``continuous`` (per-feature per-cluster
            Hedges' g\\* entries), ``categorical`` (per-feature Cramer's V
            entries), and ``summary`` (aggregated top features per cluster).
        """
        self.logger.info("DESCRIPTIVE FEATURE CHARACTERIZATION")
        results = {"continuous": {}, "categorical": {}, "summary": {}}

        if self.config.continuous_columns:
            results["continuous"] = self._continuous_effects(df, labels)

        if self.config.categorical_columns:
            results["categorical"] = self._categorical_effects(df, labels)

        # Collect all p-values and apply BH FDR correction (if enabled)
        if self.config.inference.fdr_correction:
            self._apply_fdr(results)

        results["summary"] = self._build_feature_summary(results)
        return results

    def _continuous_effects(self, df, labels):
        """Compute Hedges' g* for each continuous feature per cluster."""
        self.logger.info("Continuous Features (Hedges' g* effect sizes):")
        result = {}
        for col in self.config.continuous_columns:
            if col not in df.columns:
                continue
            feature_results = {}
            for cid in range(self.n_clusters):
                entry = self._hedges_g_star(df, labels, col, cid)
                if entry is not None:
                    feature_results[cid] = entry
            result[col] = feature_results

            if feature_results:
                best = max(feature_results.items(), key=lambda x: abs(x[1]["effect_size"]))
                if abs(best[1]["effect_size"]) >= 0.2:
                    self.logger.info(
                        f"  {col}: Cluster {best[0]} {best[1]['magnitude']} "
                        f"(g*={best[1]['effect_size']:.2f}, {best[1]['direction']})"
                    )
        return result

    def _hedges_g_star(self, df, labels, col, cluster_id):
        """Compute Hedges' g* (small-sample-corrected) effect size with Welch variance
        for a cluster vs. rest comparison on a single continuous feature.
        """
        mask = labels == cluster_id
        c_data = df.loc[mask, col].dropna()
        r_data = df.loc[~mask, col].dropna()
        if len(c_data) < 2 or len(r_data) < 2:
            return None

        n1, n2 = len(c_data), len(r_data)
        s1 = float(c_data.std(ddof=1))
        s2 = float(r_data.std(ddof=1))
        s_avg = np.sqrt((s1**2 + s2**2) / 2.0)

        scale = max(abs(float(c_data.mean())), abs(float(r_data.mean())), 1.0)
        if s_avg < 1e-12 * scale:
            return None

        mean_diff = float(c_data.mean()) - float(r_data.mean())
        d_welch = mean_diff / s_avg

        # Welch–Satterthwaite degrees of freedom
        denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        if denom > 0:
            df_w = ((s1**2 / n1) + (s2**2 / n2)) ** 2 / denom
        else:
            df_w = float(n1 + n2 - 2)

        # Hedges' small-sample correction J(df) ~= 1 - 3 / (4*df - 1)
        if df_w > 0.25:
            j_correction = 1.0 - 3.0 / (4.0 * df_w - 1.0)
        else:
            j_correction = 1.0
        g_star = j_correction * d_welch

        abs_g = abs(g_star)
        if abs_g >= 0.8:
            mag = "large"
        elif abs_g >= 0.5:
            mag = "medium"
        elif abs_g >= 0.2:
            mag = "small"
        else:
            mag = "negligible"

        # Welch's t-test p-value
        _, p_value = stats.ttest_ind(c_data, r_data, equal_var=False)

        return {
            "effect_size": float(g_star),
            "effect_size_metric": "hedges_g_star_welch",
            "direction": "higher" if g_star > 0 else "lower",
            "magnitude": mag,
            "cluster_mean": float(c_data.mean()),
            "rest_mean": float(r_data.mean()),
            "cluster_std": s1,
            "rest_std": s2,
            "average_std": float(s_avg),
            "welch_df": float(df_w),
            "n_samples": n1,
            "p_value": float(p_value),
        }

    def _categorical_effects(self, df, labels):
        """Compute Cramer's V for each categorical feature per cluster."""
        self.logger.info("Categorical Features (Cramer's V association):")
        result = {}
        for col in self.config.categorical_columns:
            if col not in df.columns:
                continue
            overall_dist = df[col].value_counts(normalize=True, dropna=False)
            feature_results = {}
            for cid in range(self.n_clusters):
                entry = self._cramers_v_entry(df, labels, col, cid, overall_dist)
                if entry is not None:
                    feature_results[cid] = entry
            result[col] = feature_results

            if feature_results:
                best = max(feature_results.items(), key=lambda x: x[1]["cramers_v"])
                if best[1]["cramers_v"] >= 0.1:
                    self.logger.info(
                        f"  {col}: Cluster {best[0]} (V={best[1]['cramers_v']:.2f}, "
                        f"dominant: {best[1]['dominant_category']})"
                    )
        return result

    def _apply_fdr(self, results):
        """Collect all p-values from continuous and categorical results and apply BH FDR."""
        keys = []
        pvals = []
        for col, fr in results.get("continuous", {}).items():
            for cid, entry in fr.items():
                pv = entry.get("p_value")
                if pv is not None and not np.isnan(pv):
                    keys.append(("continuous", col, cid))
                    pvals.append(pv)
        for col, fr in results.get("categorical", {}).items():
            for cid, entry in fr.items():
                pv = entry.get("p_value")
                if pv is not None and not np.isnan(pv):
                    keys.append(("categorical", col, cid))
                    pvals.append(pv)

        if not pvals:
            return

        q_values = apply_fdr_correction(pvals)
        for (kind, col, cid), q_val in zip(keys, q_values):
            results[kind][col][cid]["q_value"] = q_val

    def _cramers_v_entry(self, df, labels, col, cluster_id, overall_dist):
        """Compute Cramer's V entry for one categorical feature in one cluster."""
        mask = labels == cluster_id
        c_data = df.loc[mask, col].dropna()
        if len(c_data) == 0:
            return None

        cluster_dist = c_data.value_counts(normalize=True)
        dominant_cat, dominant_ratio = self._find_dominant(cluster_dist, overall_dist)
        cramers_v, chi2_p = self._compute_cramers_v(df[col].values, mask)

        return {
            "cramers_v": float(cramers_v),
            "dominant_category": dominant_cat,
            "dominant_ratio": float(dominant_ratio),
            "cluster_distribution": cluster_dist.to_dict(),
            "n_samples": len(c_data),
            "p_value": float(chi2_p) if chi2_p is not None else None,
        }

    @staticmethod
    def _find_dominant(cluster_dist, overall_dist):
        max_diff, dominant_cat, dominant_ratio = -np.inf, None, 1.0
        for cat in cluster_dist.index:
            c_prop = cluster_dist.get(cat, 0)
            o_prop = overall_dist.get(cat, 0)
            if o_prop > 0:
                diff = c_prop - o_prop
                if diff > max_diff:
                    max_diff = diff
                    dominant_cat = cat
                    dominant_ratio = c_prop / o_prop
        return dominant_cat, dominant_ratio

    @staticmethod
    def _compute_cramers_v(
        categorical_values: np.ndarray, cluster_mask: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Compute Cramer's V between a categorical variable and cluster membership."""
        valid_mask = pd.notna(categorical_values)
        cat_valid = categorical_values[valid_mask]
        cluster_valid = cluster_mask[valid_mask].astype(int)

        if len(cat_valid) == 0:
            return (0.0, None)

        contingency = pd.crosstab(cat_valid, cluster_valid)
        try:
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        except ValueError:
            return (0.0, None)

        n = len(cat_valid)
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return (0.0, None)

        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return (min(cramers_v, 1.0), float(p_val))

    def _build_feature_summary(self, results):
        """Build summary of top discriminating features per cluster."""
        cfg = self.config.feature_characterization
        use_grouping = cfg.group_by_prefix
        self.logger.info("Top Discriminating Features per Cluster:")
        summary = {}

        for cid in range(self.n_clusters):
            features = self._collect_cluster_features(results, cid)

            if use_grouping:
                summary_feats = self._grouped_top_features(features, cfg, cid)
            else:
                features.sort(key=lambda x: x[1], reverse=True)
                summary_feats = features[:5]
                self.logger.info(f"  Cluster {cid}:")
                for f in summary_feats[:3]:
                    self.logger.info(f"    - {f[0]}: {f[1]:.2f} ({f[3]})")

            summary[cid] = [
                {"feature": f[0], "importance": float(f[1]), "effect": float(f[2]), "type": f[3]}
                for f in summary_feats
            ]
        return summary

    @staticmethod
    def _collect_cluster_features(results, cid):
        features = []
        for col, fr in results["continuous"].items():
            if cid in fr:
                d = fr[cid]["effect_size"]
                features.append((col, abs(d), d, "continuous"))
        for col, fr in results["categorical"].items():
            if cid in fr:
                v = fr[cid]["cramers_v"]
                features.append((col, v, v, "categorical"))
        return features

    def _grouped_top_features(self, features, cfg, cid):
        groups = defaultdict(list)
        for f in features:
            groups[_assign_feature_group(f[0], cfg)].append(f)

        summary = []
        self.logger.info(f"  Cluster {cid}:")
        for gn in sorted(groups.keys()):
            gf = sorted(groups[gn], key=lambda x: x[1], reverse=True)
            top = gf[: cfg.n_top_per_group]
            summary.extend(top)
            self.logger.info(f"    {gn} (top {cfg.n_top_per_group}):")
            for f in top[:3]:
                self.logger.info(f"      - {f[0]}: {f[1]:.2f} ({f[3]})")

        summary.sort(key=lambda x: x[1], reverse=True)
        return summary

    def get_top_features_per_cluster(
        self,
        feature_importance: Dict,
        n_top: int = 10,
        feature_char_config: Optional[FeatureCharacterizationConfig] = None,
    ) -> Dict[int, List[Dict]]:
        """Get top discriminating features per cluster."""
        use_grouping = feature_char_config is not None and feature_char_config.group_by_prefix
        result = {}

        for cid in range(self.n_clusters):
            cluster_features = self._build_feature_dicts(
                feature_importance, cid, use_grouping, feature_char_config
            )

            if use_grouping:
                result[cid] = self._select_grouped(cluster_features, feature_char_config)
            else:
                n = feature_char_config.n_top_overall if feature_char_config else n_top
                cluster_features.sort(key=lambda x: x["importance"], reverse=True)
                result[cid] = cluster_features[:n]

        return result

    @staticmethod
    def _build_feature_dicts(fi, cid, use_grouping, cfg):
        features = []
        for col, fr in fi.get("continuous", {}).items():
            if cid in fr:
                d = {
                    "feature": col,
                    "type": "continuous",
                    "importance": abs(fr[cid]["effect_size"]),
                    "effect_size": fr[cid]["effect_size"],
                    "direction": fr[cid]["direction"],
                    "magnitude": fr[cid]["magnitude"],
                }
                if use_grouping:
                    d["group"] = _assign_feature_group(col, cfg)
                features.append(d)
        for col, fr in fi.get("categorical", {}).items():
            if cid in fr:
                d = {
                    "feature": col,
                    "type": "categorical",
                    "importance": fr[cid]["cramers_v"],
                    "cramers_v": fr[cid]["cramers_v"],
                    "dominant_category": fr[cid]["dominant_category"],
                    "dominant_ratio": fr[cid]["dominant_ratio"],
                }
                if use_grouping:
                    d["group"] = _assign_feature_group(col, cfg)
                features.append(d)
        return features

    @staticmethod
    def _select_grouped(features, cfg):
        groups = defaultdict(list)
        for f in features:
            groups[f["group"]].append(f)
        selected = []
        for gn in sorted(groups.keys()):
            gf = sorted(groups[gn], key=lambda x: x["importance"], reverse=True)
            selected.extend(gf[: cfg.n_top_per_group])
        selected.sort(key=lambda x: x["importance"], reverse=True)
        return selected
