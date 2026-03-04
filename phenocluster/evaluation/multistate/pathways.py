"""Pathway frequency analysis by phenotype."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import PathwayResult, PatientTrajectory


def analyze_pathway_frequencies(
    data: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
    state_by_id: dict,
    trajectories: Optional[List[PatientTrajectory]] = None,
    determine_state_sequence=None,
    logger=None,
) -> List[PathwayResult]:
    """Analyze pathway frequencies by phenotype."""
    if logger:
        logger.info("Analyzing pathway frequencies...")

    pathway_counts: Dict[str, Dict[int, int]] = {}

    if trajectories is not None:
        for traj in trajectories:
            phenotype = labels[traj.sample_id]
            pathway = " -> ".join(str(s) for s in traj.states)
            if pathway not in pathway_counts:
                pathway_counts[pathway] = {p: 0 for p in range(n_clusters)}
            pathway_counts[pathway][int(phenotype)] += 1
    else:
        for idx, (_, row) in enumerate(data.iterrows()):
            states, _ = determine_state_sequence(row, idx)
            phenotype = int(labels[idx])
            pathway = " -> ".join(str(s) for s in states)
            if pathway not in pathway_counts:
                pathway_counts[pathway] = {p: 0 for p in range(n_clusters)}
            pathway_counts[pathway][phenotype] += 1

    results = []
    for pathway, counts in pathway_counts.items():
        total = sum(counts.values())
        state_ids = [int(s) for s in pathway.replace(" -> ", " ").split()]
        state_names = [
            state_by_id[sid].name if sid in state_by_id else f"State {sid}" for sid in state_ids
        ]
        fractions = {p: counts[p] / total if total > 0 else 0 for p in range(n_clusters)}
        results.append(
            PathwayResult(
                pathway=pathway,
                state_names=state_names,
                counts_by_phenotype=counts,
                total_count=total,
                fraction_by_phenotype=fractions,
            )
        )

    results.sort(key=lambda x: x.total_count, reverse=True)

    if logger:
        logger.info(f"Found {len(results)} unique pathways")
    return results
