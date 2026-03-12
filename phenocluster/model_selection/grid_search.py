"""
PhenoCluster Model Selection
==============================

StepMix model selection using information criteria (BIC, AIC, ICL, etc.).

Each candidate number of clusters is fitted on the full training set and
scored directly. Supports both measurement-only models and measurement +
structural models for distal outcome analysis using StepMix's built-in
capabilities.
"""

import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from stepmix.stepmix import StepMix

from ..core.exceptions import ModelNotFittedError
from ..core.types import ModelSelectionResult
from .scorers import get_all_criteria

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..config import ModelSelectionConfig

# IC criteria where lower values indicate better fit
_LOWER_IS_BETTER = {"BIC", "AIC", "CAIC", "SABIC", "ICL"}


class StepMixModelSelector:
    """
    Model selection for StepMix using information criteria.

    Fits each candidate number of clusters on the full training set and
    selects the best by the chosen information criterion (BIC, AIC, etc.).

    Supports both measurement-only models and models with structural
    components for distal outcome analysis (using StepMix's 3-step
    estimation with BCH/ML correction).

    Parameters
    ----------
    config : ModelSelectionConfig
        Configuration for model selection
    measurement : dict or str
        StepMix measurement model specification
    structural : dict or str, optional
        StepMix structural model specification for distal outcomes
    stepmix_params : dict, optional
        Additional StepMix parameters (n_steps, correction, etc.)

    Examples
    --------
    >>> # Measurement-only model
    >>> config = ModelSelectionConfig(min_clusters=2, max_clusters=6, criterion='BIC')
    >>> selector = StepMixModelSelector(config, measurement='continuous')
    >>> best_model, results = selector.select_best_model(X)

    >>> # Model with structural (distal outcomes)
    >>> structural = {'outcomes': {'model': 'bernoulli', 'n_columns': 6}}
    >>> selector = StepMixModelSelector(
    ...     config, measurement='continuous', structural=structural,
    ...     stepmix_params={'n_steps': 3, 'correction': 'BCH'}
    ... )
    >>> best_model, results = selector.select_best_model(X, Y=outcomes)
    """

    def __init__(
        self,
        config: "ModelSelectionConfig",
        measurement: Union[Dict, str] = "continuous",
        structural: Optional[Union[Dict, str]] = None,
        stepmix_params: Optional[Dict] = None,
    ):
        self.config = config
        self.measurement = measurement
        self.structural = structural
        self.stepmix_params = stepmix_params or {}

        self.best_model_ = None
        self.cv_results_ = None
        self.all_criteria_ = None
        self.structural_params_ = None
        self._is_fitted = False
        self._has_structural = structural is not None

    def _build_param_grid(self) -> Dict[str, List]:
        """Build parameter grid for model selection.

        Only n_components is searched; n_init is a computational
        parameter (number of random restarts) and is held fixed.
        """
        return {
            "n_components": list(range(self.config.min_clusters, self.config.max_clusters + 1)),
        }

    def _create_base_model(self, include_structural: bool = False) -> StepMix:
        """Create base StepMix model with configured parameters."""
        params = {
            "measurement": self.measurement,
            "random_state": self.config.random_state,
            "verbose": 0,
            "progress_bar": 0,
        }

        if include_structural and self._has_structural:
            params["structural"] = self.structural
            params["n_steps"] = self.stepmix_params.get("n_steps", 3)
            params["correction"] = self.stepmix_params.get("correction", "BCH")
            params["assignment"] = self.stepmix_params.get("assignment", "modal")

        for key, value in self.stepmix_params.items():
            if key not in ["n_steps", "correction", "assignment", "structural"]:
                params[key] = value
            elif include_structural and key not in params:
                params[key] = value

        return StepMix(**params)

    def _validate_cluster_sizes(self, model: StepMix, X: np.ndarray, min_size: int) -> bool:
        """Check if all clusters meet minimum size requirement."""
        try:
            labels = model.predict(X)
            unique, counts = np.unique(labels, return_counts=True)

            if len(unique) != model.n_components:
                logger.debug(
                    f"k={model.n_components}: expected {model.n_components} "
                    f"clusters, got {len(unique)}"
                )
                return False

            if np.min(counts) < min_size:
                logger.debug(
                    f"k={model.n_components}: cluster sizes "
                    f"{dict(zip(unique, counts))}, min required {min_size}"
                )
                return False

            return True
        except (ValueError, AttributeError):
            return False

    def _fit_candidates(
        self, X: np.ndarray, k_range: List[int], n_init: int, ic_method: str
    ) -> Tuple[Dict[int, StepMix], Dict[int, float]]:
        """Fit models for each candidate k and collect IC scores."""
        fitted_models: Dict[int, StepMix] = {}
        ic_scores: Dict[int, float] = {}

        for k in k_range:
            model = self._create_base_model(include_structural=False)
            model.set_params(n_components=k, n_init=n_init)
            try:
                model.fit(X)
                criteria = get_all_criteria(model, X)
                ic_val = criteria.get(ic_method)
                if ic_val is not None and np.isfinite(ic_val):
                    fitted_models[k] = model
                    ic_scores[k] = ic_val
                else:
                    logger.debug(f"k={k}: {ic_method} is non-finite, skipping")
            except (ValueError, RuntimeError) as e:
                logger.debug(f"k={k}: fitting failed: {e}")

        if not ic_scores:
            raise ValueError(
                "All model fits failed. "
                "This usually indicates a data issue "
                "(e.g., too many missing values, "
                "constant features, or incompatible data types). "
                "Try enabling imputation or checking your data quality."
            )

        return fitted_models, ic_scores

    def _select_best_k(self, ic_scores: Dict[int, float], ic_method: str) -> int:
        """Select the best k based on IC scores."""
        if ic_method in _LOWER_IS_BETTER:
            return min(ic_scores, key=ic_scores.get)
        return max(ic_scores, key=ic_scores.get)

    def _build_results_table(
        self, ic_scores: Dict[int, float], n_init: int, ic_method: str
    ) -> pd.DataFrame:
        """Build CV-style results table for reporting."""
        rows = []
        for rank, k in enumerate(
            sorted(
                ic_scores,
                key=ic_scores.get,
                reverse=(ic_method not in _LOWER_IS_BETTER),
            ),
            start=1,
        ):
            rows.append(
                {
                    "param_n_components": k,
                    "param_n_init": n_init,
                    "mean_test_score": (
                        -ic_scores[k] if ic_method in _LOWER_IS_BETTER else ic_scores[k]
                    ),
                    "rank_test_score": rank,
                }
            )
        return pd.DataFrame(rows)

    def _apply_cluster_size_validation(
        self,
        best_model: StepMix,
        X: np.ndarray,
        min_cluster_size: int,
        n_init: int,
    ) -> Tuple[StepMix, Dict]:
        """Validate cluster sizes and find alternative if needed."""
        best_params = {
            "n_components": best_model.n_components,
            "n_init": n_init,
        }

        if self._validate_cluster_sizes(best_model, X, min_cluster_size):
            return best_model, best_params

        valid_model = self._find_valid_model(X, min_cluster_size)
        if valid_model is not None:
            best_params["n_components"] = valid_model.n_components
            return valid_model, best_params

        labels = best_model.predict(X)
        _, counts = np.unique(labels, return_counts=True)
        logger.warning(
            f"No model meets min_cluster_size={min_cluster_size}. "
            f"Proceeding with best k={best_model.n_components} "
            f"(cluster sizes: {sorted(counts)}). "
            f"Consider reducing min_cluster_size in config."
        )
        return best_model, best_params

    def select_best_model(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
    ) -> Tuple[StepMix, ModelSelectionResult]:
        """
        Select best model using information criteria.

        Each candidate number of clusters is fitted on the full training
        set and scored by the configured information criterion. Model
        selection is performed on the measurement model only (X). If Y
        is provided and a structural model is configured, the structural
        model is fitted in a second step using 3-step estimation with
        BCH/ML correction.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for measurement model
        Y : np.ndarray, optional
            Outcome matrix for structural model (distal outcomes).
            Required if structural model was specified in constructor.
        param_grid : dict, optional
            Custom parameter grid (overrides config)

        Returns
        -------
        Tuple[StepMix, ModelSelectionResult]
            Best fitted model and selection results
        """
        if param_grid is None:
            param_grid = self._build_param_grid()

        n_samples = X.shape[0]
        min_cluster_size = self.config.get_min_cluster_size(n_samples)
        n_init = self.config.n_init[0] if self.config.n_init else 20
        ic_method = self.config.criterion.upper()
        k_range = param_grid["n_components"]

        from sklearn.exceptions import ConvergenceWarning

        old_np_settings = np.seterr(divide="ignore", invalid="ignore", over="ignore")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                fitted_models, ic_scores = self._fit_candidates(X, k_range, n_init, ic_method)

                best_k = self._select_best_k(ic_scores, ic_method)
                best_measurement_model = fitted_models[best_k]

                self.cv_results_ = self._build_results_table(ic_scores, n_init, ic_method)

                best_measurement_model, best_params = self._apply_cluster_size_validation(
                    best_measurement_model, X, min_cluster_size, n_init
                )

                if self._has_structural and Y is not None:
                    self.best_model_ = self._fit_with_structural(X, Y, best_params)
                else:
                    self.best_model_ = best_measurement_model

                self.all_criteria_ = get_all_criteria(self.best_model_, X)
        finally:
            np.seterr(**old_np_settings)

        self._is_fitted = True

        best_score = -ic_scores[best_k] if ic_method in _LOWER_IS_BETTER else ic_scores[best_k]
        result_dict = {
            "best_model": self.best_model_,
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": self.cv_results_.to_dict(),
        }

        if self.structural_params_ is not None:
            result_dict["structural_params"] = self.structural_params_

        result = ModelSelectionResult(**result_dict)
        return self.best_model_, result

    def _fit_with_structural(self, X: np.ndarray, Y: np.ndarray, best_params: Dict) -> StepMix:
        """Fit model with structural component using 3-step estimation."""
        model = self._create_base_model(include_structural=True)
        model.set_params(
            **{k: v for k, v in best_params.items() if k in ["n_components", "n_init"]}
        )

        model.fit(X, Y)

        try:
            params = model.get_parameters()
            if "structural" in params:
                self.structural_params_ = params["structural"]
        except AttributeError:
            pass

        return model

    def _find_valid_model(self, X: np.ndarray, min_cluster_size: int) -> Optional[StepMix]:
        """Find first model meeting cluster size constraints, by score order."""
        scores = self.cv_results_["mean_test_score"].values
        finite_mask = np.isfinite(scores)
        sorted_idx = np.argsort(-np.where(finite_mask, scores, -np.inf))

        n_init = self.config.n_init[0] if self.config.n_init else 20

        for idx in sorted_idx:
            if not finite_mask[idx]:
                continue

            params = {
                "n_components": self.cv_results_.loc[idx, "param_n_components"],
                "n_init": n_init,
            }

            model = self._create_base_model(include_structural=False)
            model.set_params(**params)

            try:
                model.fit(X)
                if self._validate_cluster_sizes(model, X, min_cluster_size):
                    return model
            except (ValueError, RuntimeError):
                continue

        return None

    def get_comparison_table(self) -> pd.DataFrame:
        """Get a formatted comparison table of all evaluated models."""
        if self.cv_results_ is None:
            raise ModelNotFittedError("StepMixModelSelector")

        df = self.cv_results_[
            [
                "param_n_components",
                "param_n_init",
                "mean_test_score",
                "rank_test_score",
            ]
        ].copy()

        df.columns = ["n_clusters", "n_init", "mean_score", "rank"]
        df = df.sort_values("rank")

        return df

    def get_selection_results(self) -> Dict:
        """Get model selection results as dictionary."""
        if not self._is_fitted:
            raise ModelNotFittedError("StepMixModelSelector")

        comparison = self.get_comparison_table()
        criterion_name = self.config.criterion
        sign = -1.0 if criterion_name.upper() in _LOWER_IS_BETTER else 1.0
        all_results = []
        for _, row in comparison.iterrows():
            all_results.append(
                {
                    "n_clusters": int(row["n_clusters"]),
                    criterion_name: float(row["mean_score"]) * sign,
                }
            )

        return {
            "comparison_table": comparison,
            "all_criteria": self.all_criteria_,
            "best_n_clusters": self.best_model_.n_components,
            "criterion": self.config.criterion,
            "cv_results": self.cv_results_,
            "all_results": all_results,
        }
