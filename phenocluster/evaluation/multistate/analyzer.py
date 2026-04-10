"""MultistateAnalyzer orchestrator."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ...config import PhenoClusterConfig
from ...utils.logging import get_logger
from .hazard_ratios import extract_hazard_ratios
from .pathways import analyze_pathway_frequencies
from .trajectory import TrajectoryBuilder
from .transition_hazards import TransitionHazardFitter
from .types import MonteCarloResults, MultistateResults


class MultistateAnalyzer:
    """Multi-state survival analysis with phenotype-specific transitions.

    Uses lifelines CoxPHFitter for transition-specific hazard modeling and
    Cox proportional hazards for hazard ratio estimation.
    """

    # See SurvivalAnalyzer.MAX_REASONABLE_HR for the rationale: HRs outside
    # [1/1000, 1000] are flagged as ``unreliable`` (typical signature of
    # separation or an ill-conditioned transition-specific design matrix).
    MAX_REASONABLE_HR = 1000

    def __init__(self, config: PhenoClusterConfig, n_clusters: int, reference_phenotype: int = 0):
        self.config = config
        self.n_clusters = n_clusters
        self.reference_phenotype = reference_phenotype
        self.logger = get_logger("multistate", config)
        self.multistate_config = config.multistate

        self._traj_builder = TrajectoryBuilder(config, n_clusters, reference_phenotype, self.logger)

    def prepare_trajectories(self, data, labels, baseline_confounders=None):
        """Delegate to TrajectoryBuilder."""
        return self._traj_builder.prepare_trajectories(data, labels, baseline_confounders)

    def fit_transition_model(self, trajectories):
        """Fit multistate transition-specific hazard model."""
        self.logger.info("Fitting transition-specific hazard model (lifelines)...")

        if not trajectories:
            raise ValueError("No trajectories provided for transition model fitting")

        terminal_states = self._traj_builder.get_terminal_state_ids()
        covariate_names = trajectories[0].covariates.index.tolist()

        model = TransitionHazardFitter(
            terminal_states=terminal_states,
            covariate_names=covariate_names,
            logger=self.logger,
            random_state=self.config.random_state,
        )
        model.fit(trajectories)

        n_fitted = sum(len(targets) for targets in model.state_models.values())
        self.logger.info(f"Fitted {n_fitted} transition-specific Cox models")
        return model

    def extract_hazard_ratios(self, trajectories):
        """Fit Cox PH models for each transition and extract hazard ratios."""
        return extract_hazard_ratios(
            self.multistate_config.transitions,
            trajectories,
            self.n_clusters,
            self.reference_phenotype,
            self.config.inference,
            self.MAX_REASONABLE_HR,
            self.logger,
            min_events_per_transition=(self.multistate_config.min_events_per_transition),
            baseline_confounders=(self.multistate_config.baseline_confounders),
        )

    def run_monte_carlo_predictions(
        self,
        model,
        trajectories,
        phenotype_indices,
        n_simulations=None,
        time_points=None,
        max_transitions=None,
    ) -> MonteCarloResults:
        """
        Estimate state occupation probabilities per phenotype using MC.

        Simulates patient trajectories through the multistate model for
        each phenotype and aggregates them into state-occupation
        probability curves with bootstrap confidence bands.

        Parameters
        ----------
        model
            Fitted multistate model with per-transition Cox PH fits.
        trajectories
            Observed patient trajectories used to seed the simulation.
        phenotype_indices
            Mapping of phenotype id to the indices of patients assigned
            to that phenotype.
        n_simulations : int, optional
            Number of Monte Carlo simulations per patient. Defaults to
            ``multistate_config.monte_carlo_n_simulations`` when ``None``.
        time_points : sequence of float, optional
            Time points at which to report state occupation probabilities.
            Defaults to ``multistate_config.monte_carlo_time_points`` when
            ``None``.
        max_transitions : int, optional
            Maximum number of transitions to simulate per trajectory.
            Defaults to ``multistate_config.max_transitions_per_path``
            when ``None``.

        Returns
        -------
        MonteCarloResults
            Container with the evaluated time points, per-phenotype
            state-occupation probability curves, lower/upper bootstrap
            bands, the number of simulations, and a simulation summary.
        """
        self.logger.info("Running Monte Carlo predictions...")

        times = time_points or self.multistate_config.monte_carlo_time_points
        max_trans = max_transitions or self.multistate_config.max_transitions_per_path
        n_sims = n_simulations or self.multistate_config.monte_carlo_n_simulations

        self._warn_extrapolation(trajectories, times)

        initial_state = self._traj_builder.get_initial_state().id
        all_state_ids = [s.id for s in self.multistate_config.states]
        traj_by_id = {t.sample_id: t for t in trajectories}
        n_boot = 200

        state_probs: Dict[int, Dict[int, List[float]]] = {}
        state_probs_lower: Dict[int, Dict[int, List[float]]] = {}
        state_probs_upper: Dict[int, Dict[int, List[float]]] = {}

        for phenotype in range(self.n_clusters):
            defaults = {s: [0.0] * len(times) for s in all_state_ids}
            state_probs[phenotype] = dict(defaults)
            state_probs_lower[phenotype] = dict(defaults)
            state_probs_upper[phenotype] = dict(defaults)

            patient_probs = self._simulate_phenotype_patients(
                model,
                phenotype,
                phenotype_indices,
                traj_by_id,
                initial_state,
                all_state_ids,
                times,
                max_trans,
                n_sims,
            )
            if not patient_probs:
                continue

            self._aggregate_patient_probs(
                phenotype,
                patient_probs,
                all_state_ids,
                times,
                state_probs,
                state_probs_lower,
                state_probs_upper,
                n_boot,
            )

        self.logger.info(f"Monte Carlo completed: {len(times)} time points")
        return MonteCarloResults(
            time_points=times,
            state_probabilities=state_probs,
            n_simulations=n_sims,
            simulation_summary={
                "max_transitions": max_trans,
                "initial_state": initial_state,
                "n_phenotypes": self.n_clusters,
                "method": "marginalised_coxph",
                "n_bootstrap": n_boot,
            },
            state_probabilities_lower=state_probs_lower,
            state_probabilities_upper=state_probs_upper,
        )

    def _warn_extrapolation(self, trajectories, times):
        """Warn if MC time points exceed observed follow-up."""
        if not trajectories:
            return
        max_observed = max(
            t.time_at_each_state[-1] if t.time_at_each_state else 0 for t in trajectories
        )
        if max_observed <= 0:
            return
        beyond = [t for t in times if t > max_observed]
        if beyond:
            self.logger.warning(
                f"MC time points {beyond} exceed max observed "
                f"follow-up ({max_observed:.1f}). "
                f"Extrapolated probabilities may be unreliable."
            )

    def _simulate_phenotype_patients(
        self,
        model,
        phenotype,
        phenotype_indices,
        traj_by_id,
        initial_state,
        all_state_ids,
        times,
        max_trans,
        n_sims,
    ) -> List[Dict[int, List[float]]]:
        """Run MC simulations for all patients in a phenotype."""
        patient_idxs = phenotype_indices.get(phenotype, [])
        if not patient_idxs:
            self.logger.warning(f"No patients in phenotype {phenotype}")
            return []

        patient_trajs = [traj_by_id[i] for i in patient_idxs if i in traj_by_id]
        if not patient_trajs:
            self.logger.warning(f"No valid trajectories for phenotype {phenotype}")
            return []

        per_patient_sims = max(1, -(-n_sims // len(patient_trajs)))
        floor_per_patient = 50
        if per_patient_sims < floor_per_patient:
            self.logger.warning(
                f"Phenotype {phenotype}: requested {n_sims} simulations spread "
                f"over {len(patient_trajs)} patients yields "
                f"{per_patient_sims} sims/patient; raising to {floor_per_patient} "
                "for stability. Total simulations will exceed the configured "
                "budget for this phenotype."
            )
            per_patient_sims = floor_per_patient
        patient_probs = []
        for traj in patient_trajs:
            try:
                probs = self._mc_for_covariates(
                    model,
                    traj.covariates.values,
                    initial_state,
                    all_state_ids,
                    times,
                    max_trans,
                    per_patient_sims,
                )
                patient_probs.append(probs)
            except Exception:
                continue

        if not patient_probs:
            self.logger.warning(f"Monte Carlo failed for all patients in phenotype {phenotype}")
        else:
            self.logger.info(
                f"Phenotype {phenotype}: marginalised over "
                f"{len(patient_probs)} patients "
                f"({per_patient_sims} sims each)"
            )
        return patient_probs

    def _aggregate_patient_probs(
        self,
        phenotype,
        patient_probs,
        all_state_ids,
        times,
        state_probs,
        state_probs_lower,
        state_probs_upper,
        n_boot,
    ):
        """Average patient-level probs and compute bootstrap CIs."""
        for s in all_state_ids:
            vals = [p[s] for p in patient_probs]
            state_probs[phenotype][s] = [
                float(np.mean([v[t_idx] for v in vals])) for t_idx in range(len(times))
            ]

        rng = np.random.default_rng(self.config.random_state)
        boot_probs: Dict[int, List[List[float]]] = {s: [] for s in all_state_ids}
        for _ in range(n_boot):
            boot_idx = rng.choice(len(patient_probs), len(patient_probs), replace=True)
            for s in all_state_ids:
                vals = [patient_probs[boot_idx[j]][s] for j in range(len(boot_idx))]
                boot_probs[s].append(
                    [float(np.mean([v[t_idx] for v in vals])) for t_idx in range(len(times))]
                )

        for s in all_state_ids:
            arr = np.array(boot_probs[s])
            state_probs_lower[phenotype][s] = np.percentile(arr, 2.5, axis=0).tolist()
            state_probs_upper[phenotype][s] = np.percentile(arr, 97.5, axis=0).tolist()

    def _mc_for_covariates(
        self,
        model,
        sample_covariates,
        initial_state,
        all_state_ids,
        times,
        max_trans,
        n_sims,
    ) -> Dict[int, List[float]]:
        """Run MC simulation for a single covariate vector.

        Returns a dict mapping state_id -> [prob_at_each_time_point].
        """
        simulated = model.run_monte_carlo_simulation(
            sample_covariates=sample_covariates,
            origin_state=initial_state,
            current_time=0,
            n_random_samples=n_sims,
            max_transitions=max_trans,
            n_jobs=1,
        )
        # Vectorized state counting: collect states into array and use bincount
        sorted_ids = sorted(all_state_ids)
        id_to_idx = {s: i for i, s in enumerate(sorted_ids)}
        n_states = len(sorted_ids)
        counts = np.zeros((len(times), n_states), dtype=int)
        for t_idx, t in enumerate(times):
            states = np.array([sim_traj.state_at_time(t, initial_state) for sim_traj in simulated])
            for s_idx, s in enumerate(sorted_ids):
                counts[t_idx, s_idx] = np.sum(states == s)
        probs = counts / n_sims
        result: Dict[int, List[float]] = {s: probs[:, id_to_idx[s]].tolist() for s in all_state_ids}
        return result

    def analyze_pathway_frequencies(self, data, labels, trajectories=None):
        """Analyze pathway frequencies by phenotype."""
        return analyze_pathway_frequencies(
            data,
            labels,
            self.n_clusters,
            self._traj_builder._state_by_id,
            trajectories=trajectories,
            determine_state_sequence=self._traj_builder._determine_state_sequence,
            logger=self.logger,
        )

    def run_full_analysis(self, data: pd.DataFrame, labels: np.ndarray) -> MultistateResults:
        """Run complete multistate analysis pipeline."""
        self.logger.info("MULTISTATE ANALYSIS (lifelines)")

        missing_cols = self._check_columns(data)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return MultistateResults(
                model_summary={"error": f"Missing columns: {missing_cols}"},
                transition_results={},
                pathway_results=[],
            )

        trajectories, phenotype_indices = self.prepare_trajectories(data, labels)
        if len(trajectories) == 0:
            self.logger.error("No valid trajectories created")
            return MultistateResults(
                model_summary={"error": "No valid patient trajectories"},
                transition_results={},
                pathway_results=[],
            )

        try:
            model = self.fit_transition_model(trajectories)
        except Exception as e:
            self.logger.error(f"Transition model fitting failed: {e}")
            return MultistateResults(
                model_summary={"error": f"Model fitting failed: {e}"},
                transition_results={},
                pathway_results=[],
            )

        transition_results = self.extract_hazard_ratios(trajectories)

        mc_results = None
        if model.state_models:
            try:
                mc_results = self.run_monte_carlo_predictions(
                    model, trajectories, phenotype_indices
                )
            except Exception as e:
                self.logger.warning(f"Monte Carlo predictions failed: {e}")
        else:
            self.logger.warning("No fitted models available for Monte Carlo simulation")

        pathway_results = self.analyze_pathway_frequencies(data, labels, trajectories=trajectories)

        model_summary = {
            "n_patients": len(data),
            "n_trajectories": len(trajectories),
            "n_states": len(self.multistate_config.states),
            "n_allowed_transitions": len(self.multistate_config.transitions),
            "n_transitions_fitted": len(transition_results),
            "n_pathways_observed": len(pathway_results),
            "monte_carlo_n_simulations": self.multistate_config.monte_carlo_n_simulations,
            "transitions_fitted": list(transition_results.keys()),
            "analysis_method": "lifelines_cox_ph",
            "inference": "Cox proportional hazards",
        }

        self.logger.info("MULTISTATE ANALYSIS COMPLETE")
        self.logger.info(f"  Patients: {model_summary['n_patients']}")
        self.logger.info(f"  Transitions fitted: {len(transition_results)}")
        self.logger.info(f"  Unique pathways: {len(pathway_results)}")
        if mc_results:
            self.logger.info(f"  Monte Carlo simulations: {mc_results.n_simulations}")

        return MultistateResults(
            model_summary=model_summary,
            transition_results=transition_results,
            pathway_results=pathway_results,
            state_occupation_probabilities=mc_results,
            fitted_model=model,
        )

    def _check_columns(self, data):
        """Validate that required columns exist."""
        missing = []
        for state in self.multistate_config.states:
            if state.event_column and state.event_column not in data.columns:
                missing.append(state.event_column)
            if state.time_column and state.time_column not in data.columns:
                missing.append(state.time_column)
        return missing

    def results_to_dict(self, results: MultistateResults) -> Dict[str, Any]:
        """Convert MultistateResults to a JSON-serializable dictionary."""
        result_dict = {
            "model_summary": results.model_summary,
            "transition_results": {
                name: {
                    "transition_name": tr.transition_name,
                    "from_state": tr.from_state,
                    "to_state": tr.to_state,
                    "n_events": tr.n_events,
                    "n_at_risk": tr.n_at_risk,
                    "phenotype_effects": tr.phenotype_effects,
                    "covariate_effects": tr.covariate_effects,
                }
                for name, tr in results.transition_results.items()
            },
            "pathway_results": [
                {
                    "pathway": pr.pathway,
                    "state_names": pr.state_names,
                    "counts_by_phenotype": {str(k): v for k, v in pr.counts_by_phenotype.items()},
                    "total_count": pr.total_count,
                    "fraction_by_phenotype": {
                        str(k): v for k, v in pr.fraction_by_phenotype.items()
                    },
                }
                for pr in results.pathway_results
            ],
            "state_occupation_probabilities": (
                self._mc_results_to_dict(results.state_occupation_probabilities)
            ),
        }
        return result_dict

    @staticmethod
    def _mc_results_to_dict(mc) -> Any:
        """Convert MonteCarloResults to a serializable dict."""
        if mc is None:
            return None

        def _stringify(probs_dict):
            return {
                str(phenotype): {str(state): probs for state, probs in sp.items()}
                for phenotype, sp in probs_dict.items()
            }

        mc_dict: Dict[str, Any] = {
            "time_points": mc.time_points,
            "n_simulations": mc.n_simulations,
            "by_phenotype": _stringify(mc.state_probabilities),
            "simulation_summary": mc.simulation_summary,
        }
        if mc.state_probabilities_lower is not None:
            mc_dict["by_phenotype_lower"] = _stringify(mc.state_probabilities_lower)
        if mc.state_probabilities_upper is not None:
            mc_dict["by_phenotype_upper"] = _stringify(mc.state_probabilities_upper)
        return mc_dict
