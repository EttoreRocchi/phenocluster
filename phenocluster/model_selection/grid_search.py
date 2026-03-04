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
        self.structural_params_ = None  # Parameters from structural model
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
        """
        Create base StepMix model for model selection.

        Parameters
        ----------
        include_structural : bool
            Whether to include structural model specification.
            Set to False during model selection since the structural
            model is fitted in a second step.

        Returns
        -------
        StepMix
            Configured StepMix model
        """
        params = {
            "measurement": self.measurement,
            "random_state": self.config.random_state,
            "verbose": 0,
            "progress_bar": 0,
        }

        # Add structural model if requested and available
        if include_structural and self._has_structural:
            params["structural"] = self.structural
            # Add 3-step estimation parameters for structural models
            params["n_steps"] = self.stepmix_params.get("n_steps", 3)
            params["correction"] = self.stepmix_params.get("correction", "BCH")
            params["assignment"] = self.stepmix_params.get("assignment", "modal")

        # Add remaining stepmix_params (excluding structural-specific ones if not used)
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

            # Check if we got the expected number of clusters
            if len(unique) != model.n_components:
                logger.debug(
                    f"k={model.n_components}: expected {model.n_components} clusters, "
                    f"got {len(unique)}"
                )
                return False

            # Check minimum cluster size
            if np.min(counts) < min_size:
                logger.debug(
                    f"k={model.n_components}: cluster sizes {dict(zip(unique, counts))}, "
                    f"min required {min_size}"
                )
                return False

            return True
        except (ValueError, AttributeError):
            # Model may not be fitted or prediction failed
            return False

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

        # Suppress numerical warnings from StepMix Gaussian emission during
        # model selection. Zero-covariance clusters produce divide-by-zero and
        # invalid-value warnings that are harmless (error_score=-inf handles
        # them).
        from sklearn.exceptions import ConvergenceWarning

        old_np_settings = np.seterr(divide="ignore", invalid="ignore", over="ignore")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                # Fit each n_components on the full training set
                # and compute IC directly (not via CV fold scoring,
                # which uses wrong n in the penalty term).
                ic_method = self.config.criterion.upper()
                k_range = param_grid["n_components"]

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
                        "constant features, or incompatible "
                        "data types). "
                        "Try enabling imputation or checking "
                        "your data quality."
                    )

                # For IC criteria, lower is better; for entropy,
                # higher is better
                _lower_is_better = {
                    "BIC",
                    "AIC",
                    "CAIC",
                    "SABIC",
                    "ICL",
                }
                if ic_method in _lower_is_better:
                    best_k = min(ic_scores, key=ic_scores.get)
                else:
                    best_k = max(ic_scores, key=ic_scores.get)

                best_measurement_model = fitted_models[best_k]
                best_params = {
                    "n_components": best_k,
                    "n_init": n_init,
                }

                # Build CV-style results table for reporting
                rows = []
                for rank, k in enumerate(
                    sorted(
                        ic_scores,
                        key=ic_scores.get,
                        reverse=(ic_method not in _lower_is_better),
                    ),
                    start=1,
                ):
                    rows.append(
                        {
                            "param_n_components": k,
                            "param_n_init": n_init,
                            "mean_test_score": (
                                -ic_scores[k] if ic_method in _lower_is_better else ic_scores[k]
                            ),
                            "rank_test_score": rank,
                        }
                    )
                self.cv_results_ = pd.DataFrame(rows)

                # Validate cluster sizes for best model
                if not self._validate_cluster_sizes(best_measurement_model, X, min_cluster_size):
                    valid_model = self._find_valid_model(X, min_cluster_size)
                    if valid_model is not None:
                        best_measurement_model = valid_model
                        best_params = {
                            "n_components": (best_measurement_model.n_components),
                            "n_init": n_init,
                        }
                    else:
                        labels = best_measurement_model.predict(X)
                        _, counts = np.unique(labels, return_counts=True)
                        logger.warning(
                            f"No model meets "
                            f"min_cluster_size="
                            f"{min_cluster_size}. "
                            f"Proceeding with best k="
                            f"{best_measurement_model.n_components}"
                            f" (cluster sizes: "
                            f"{sorted(counts)}). "
                            f"Consider reducing "
                            f"min_cluster_size in config."
                        )

                # If structural model specified and Y provided,
                # fit full model
                if self._has_structural and Y is not None:
                    self.best_model_ = self._fit_with_structural(X, Y, best_params)
                else:
                    self.best_model_ = best_measurement_model

                # Compute all criteria for best model
                self.all_criteria_ = get_all_criteria(self.best_model_, X)
        finally:
            np.seterr(**old_np_settings)

        self._is_fitted = True

        best_score = -ic_scores[best_k] if ic_method in _lower_is_better else ic_scores[best_k]
        result_dict = {
            "best_model": self.best_model_,
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": self.cv_results_.to_dict(),
        }

        # Add structural parameters if available
        if self.structural_params_ is not None:
            result_dict["structural_params"] = self.structural_params_

        result = ModelSelectionResult(**result_dict)

        return self.best_model_, result

    def _fit_with_structural(self, X: np.ndarray, Y: np.ndarray, best_params: Dict) -> StepMix:
        """
        Fit model with structural component using 3-step estimation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for measurement model
        Y : np.ndarray
            Outcome matrix for structural model
        best_params : Dict
            Best parameters from model selection

        Returns
        -------
        StepMix
            Fitted model with structural component
        """
        # Create model with structural specification
        model = self._create_base_model(include_structural=True)
        model.set_params(
            **{k: v for k, v in best_params.items() if k in ["n_components", "n_init"]}
        )

        # Fit with both X and Y
        model.fit(X, Y)

        # Extract structural parameters
        try:
            params = model.get_parameters()
            if "structural" in params:
                self.structural_params_ = params["structural"]
        except AttributeError:
            # Model doesn't support get_parameters()
            pass

        return model

    def _find_valid_model(self, X: np.ndarray, min_cluster_size: int) -> Optional[StepMix]:
        """
        Find a valid model that meets cluster size constraints.

        Iterates through results in order of score to find first valid model.
        Returns None if no model meets the constraint.
        """
        scores = self.cv_results_["mean_test_score"].values
        # Sort by score descending, skipping NaN/non-finite scores
        finite_mask = np.isfinite(scores)
        sorted_idx = np.argsort(-np.where(finite_mask, scores, -np.inf))

        n_init = self.config.n_init[0] if self.config.n_init else 20

        for idx in sorted_idx:
            if not finite_mask[idx]:
                continue  # Skip models with NaN/inf scores

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
                # Model fitting failed (convergence, invalid params)
                continue

        return None

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get a formatted comparison table of all evaluated models.

        Returns
        -------
        pd.DataFrame
            Comparison table with key metrics
        """
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
        """
        Get model selection results as dictionary.

        Returns
        -------
        Dict
            Dictionary containing:
            - comparison_table: DataFrame with model comparison
            - all_criteria: Dict with all information criteria for best model
            - best_n_clusters: int, optimal number of clusters
            - criterion: str, selection criterion used
        """
        if not self._is_fitted:
            raise ModelNotFittedError("StepMixModelSelector")

        comparison = self.get_comparison_table()
        criterion_name = self.config.criterion
        # IC scorers negate values for sklearn (higher=better); un-negate for display
        _ic_criteria = {"BIC", "AIC", "CAIC", "SABIC", "ICL"}
        sign = -1.0 if criterion_name.upper() in _ic_criteria else 1.0
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
