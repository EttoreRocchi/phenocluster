"""
PhenoCluster Configuration Profiles
=====================================

Predefined analysis profiles for simplified config generation.
Each profile sets sensible defaults for a specific use case.
"""

from pathlib import Path
from typing import Dict, List

# Base config template (all defaults, data sections left empty)

_BASE_CONFIG: Dict = {
    "global": {
        "project_name": "PhenoCluster",
        "output_dir": "results",
        "random_state": 42,
    },
    "data": {
        "continuous_columns": [],
        "categorical_columns": [],
        "split": {
            "test_size": 0.2,
            "stratify_by": None,
            "shuffle": True,
        },
    },
    "preprocessing": {
        "row_filter": {"enabled": True, "max_missing_pct": 0.30},
        "imputation": {
            "enabled": False,
            "method": "iterative",
            "estimator": "bayesian_ridge",
            "max_iter": 10,
        },
        "categorical_encoding": {"method": "label", "handle_unknown": "ignore"},
        "outlier": {
            "enabled": True,
            "method": "winsorize",
            "contamination": "auto",
            "winsorize_limits": [0.01, 0.01],
        },
        "feature_selection": {
            "enabled": False,
            "method": "lasso",
            "target_column": None,
            "variance_threshold": 0.01,
            "correlation_threshold": 0.9,
            "n_features": None,
        },
    },
    "model": {
        "n_clusters": 3,
        "selection": {
            "enabled": True,
            "min_clusters": 2,
            "max_clusters": 6,
            "criterion": "BIC",
            "min_cluster_size": 0.05,
            "n_init": [100],
            "refit": True,
            "n_jobs": -1,
        },
        "stepmix": {
            "max_iter": 1000,
            "abs_tol": 1.0e-7,
            "rel_tol": 1.0e-5,
        },
    },
    "outcome": {
        "enabled": True,
        "outcome_columns": [],
    },
    "stability": {
        "enabled": True,
        "n_runs": 100,
        "subsample_fraction": 0.8,
        "n_jobs": -1,
    },
    "survival": {
        "enabled": True,
        "use_weighted": False,
        "targets": [],
    },
    "multistate": {
        "enabled": False,
        "states": [],
        "transitions": [],
        "baseline_confounders": [],
        "min_events_per_transition": 3,
        "default_followup_time": 30,
        "monte_carlo": {
            "n_simulations": 1000,
            "time_points": [5, 10, 15, 20, 25, 30],
            "max_transitions_per_path": 10,
        },
    },
    "inference": {
        "enabled": True,
        "confidence_level": 0.95,
        "fdr_correction": True,
        "outcome_test": "auto",
        "cox_penalizer": 0.0,
    },
    "reference_phenotype": {
        "strategy": "largest",
    },
    "external_validation": {
        "enabled": False,
        "external_data_path": None,
    },
    "visualization": {
        "save_plots": True,
        "dpi": 300,
    },
    "logging": {
        "level": "INFO",
        "format": "detailed",
        "log_to_file": True,
        "log_file": "phenocluster.log",
        "quiet_mode": False,
    },
    "cache": {
        "enabled": True,
        "compress_level": 3,
    },
    "data_quality": {
        "enabled": True,
        "missing_threshold": 0.15,
        "correlation_threshold": 0.9,
        "variance_threshold": 0.01,
        "generate_report": True,
    },
    "categorical_flow": {
        "group_by_prefix": True,
        "prefix_separator": "_",
        "custom_groups": {},
        "show_sankey": False,
        "show_proportion_heatmap": True,
        "min_category_pct": 0.03,
    },
    "feature_characterization": {
        "group_by_prefix": True,
        "prefix_separator": "_",
        "custom_groups": {},
        "n_top_per_group": 5,
        "n_top_overall": 20,
    },
}

# Profile overrides (merged on top of _BASE_CONFIG)

PROFILES: Dict[str, Dict] = {
    "descriptive": {
        "_description": "Phenotype discovery only - no statistical inference.",
        "outcome": {"enabled": False, "outcome_columns": []},
        "inference": {"enabled": False},
        "stability": {"enabled": True},
        "survival": {"enabled": True},
        "multistate": {"enabled": False},
    },
    "complete": {
        "_description": "All analyses enabled.",
        "inference": {"enabled": True},
        "stability": {"enabled": True},
        "survival": {"enabled": True},
        "multistate": {"enabled": True},
    },
    "quick": {
        "_description": "Fast iteration for development/debugging.",
        "inference": {"enabled": True},
        "stability": {"enabled": False},
        "survival": {"enabled": True},
        "multistate": {"enabled": False},
    },
}


# Public API


def list_profiles() -> List[str]:
    """Return available profile names."""
    return list(PROFILES.keys())


def get_profile(name: str) -> Dict:
    """Return the config overrides for a profile.

    Parameters
    ----------
    name : str
        Profile name (descriptive, complete, quick).

    Returns
    -------
    dict
        Config overrides for the profile.

    Raises
    ------
    ValueError
        If the profile name is unknown.
    """
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list_profiles()}")
    return PROFILES[name]


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (returns a new dict)."""
    merged = dict(base)
    for key, value in overrides.items():
        if key.startswith("_"):
            continue  # skip metadata keys like _description
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def create_config_yaml(profile: str, output_path: Path) -> None:
    """Write a complete commented YAML config for a given profile.

    Parameters
    ----------
    profile : str
        Profile name.
    output_path : Path
        Destination file path.
    """
    overrides = get_profile(profile)
    config = _deep_merge(_BASE_CONFIG, overrides)
    description = overrides.get("_description", "")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# PhenoCluster configuration - profile: {profile}")
    if description:
        lines.append(f"# {description}")
    lines.append("#")
    lines.append("# Generated by: phenocluster create-config")
    lines.append("")

    g = config["global"]
    lines.append("global:")
    lines.append(f'  project_name: "{g["project_name"]}"')
    lines.append(f'  output_dir: "{g["output_dir"]}"')
    lines.append(f"  random_state: {g['random_state']}")
    lines.append("")

    lines.append("data:")
    lines.append("  continuous_columns: []   # TODO: add your continuous feature columns")
    lines.append("  categorical_columns: []  # TODO: add your categorical feature columns")
    lines.append("  split:")
    d = config["data"]["split"]
    lines.append(f"    test_size: {d['test_size']}")
    lines.append("    stratify_by: null      # TODO: set stratification column or null")
    lines.append(f"    shuffle: {str(d['shuffle']).lower()}")
    lines.append("")

    pp = config["preprocessing"]
    lines.append("preprocessing:")
    lines.append("  row_filter:")
    lines.append(f"    enabled: {str(pp['row_filter']['enabled']).lower()}")
    lines.append(f"    max_missing_pct: {pp['row_filter']['max_missing_pct']}")
    lines.append("  imputation:")
    lines.append(f"    enabled: {str(pp['imputation']['enabled']).lower()}")
    lines.append(f'    method: "{pp["imputation"]["method"]}"')
    lines.append(f'    estimator: "{pp["imputation"]["estimator"]}"')
    lines.append(f"    max_iter: {pp['imputation']['max_iter']}")
    lines.append("  categorical_encoding:")
    lines.append(f'    method: "{pp["categorical_encoding"]["method"]}"')
    lines.append(f'    handle_unknown: "{pp["categorical_encoding"]["handle_unknown"]}"')
    lines.append("  outlier:")
    lines.append(f"    enabled: {str(pp['outlier']['enabled']).lower()}")
    lines.append(f'    method: "{pp["outlier"]["method"]}"')
    lines.append(f'    contamination: "{pp["outlier"]["contamination"]}"')
    wl = pp["outlier"]["winsorize_limits"]
    lines.append(f"    winsorize_limits: [{wl[0]}, {wl[1]}]")
    lines.append("  feature_selection:")
    lines.append(f"    enabled: {str(pp['feature_selection']['enabled']).lower()}")
    lines.append(f'    method: "{pp["feature_selection"]["method"]}"')
    lines.append("    target_column: null")
    lines.append(f"    variance_threshold: {pp['feature_selection']['variance_threshold']}")
    lines.append(f"    correlation_threshold: {pp['feature_selection']['correlation_threshold']}")
    lines.append("    n_features: null")
    lines.append("")

    m = config["model"]
    lines.append("model:")
    lines.append(f"  n_clusters: {m['n_clusters']}")
    lines.append("  selection:")
    sel = m["selection"]
    lines.append(f"    enabled: {str(sel['enabled']).lower()}")
    lines.append(f"    min_clusters: {sel['min_clusters']}")
    lines.append(f"    max_clusters: {sel['max_clusters']}")
    lines.append(f'    criterion: "{sel["criterion"]}"')
    lines.append(f"    min_cluster_size: {sel['min_cluster_size']}")
    lines.append(f"    n_init: {sel['n_init']}")
    lines.append(f"    refit: {str(sel['refit']).lower()}")
    lines.append(f"    n_jobs: {sel['n_jobs']}")
    lines.append("  stepmix:")
    sm = m["stepmix"]
    lines.append(f"    max_iter: {sm['max_iter']}")
    lines.append(f"    abs_tol: {sm['abs_tol']}")
    lines.append(f"    rel_tol: {sm['rel_tol']}")
    lines.append("")

    oc = config["outcome"]
    lines.append("outcome:")
    lines.append(f"  enabled: {str(oc['enabled']).lower()}")
    lines.append("  outcome_columns: []  # TODO: add your binary outcome columns")
    lines.append("")

    st = config["stability"]
    lines.append("stability:")
    lines.append(f"  enabled: {str(st['enabled']).lower()}")
    lines.append(f"  n_runs: {st['n_runs']}")
    lines.append(f"  subsample_fraction: {st['subsample_fraction']}")
    lines.append(f"  n_jobs: {st['n_jobs']}")
    lines.append("")

    sv = config["survival"]
    lines.append("survival:")
    lines.append(f"  enabled: {str(sv['enabled']).lower()}")
    lines.append(f"  use_weighted: {str(sv['use_weighted']).lower()}")
    lines.append("  targets: []")
    lines.append("  # Example target:")
    lines.append('  # - name: "overall_survival"')
    lines.append('  #   time_column: "time_to_event"')
    lines.append('  #   event_column: "event_indicator"')
    lines.append("")

    ms = config["multistate"]
    lines.append("multistate:")
    lines.append(f"  enabled: {str(ms['enabled']).lower()}")
    lines.append("  states: []             # TODO: define states if enabled")
    lines.append("  transitions: []        # TODO: define transitions if enabled")
    lines.append("  baseline_confounders: []")
    lines.append(f"  min_events_per_transition: {ms['min_events_per_transition']}")
    lines.append(f"  default_followup_time: {ms['default_followup_time']}")
    mc = ms["monte_carlo"]
    lines.append("  monte_carlo:")
    lines.append(f"    n_simulations: {mc['n_simulations']}")
    lines.append(f"    time_points: {mc['time_points']}")
    lines.append(f"    max_transitions_per_path: {mc['max_transitions_per_path']}")
    lines.append("")

    inf = config["inference"]
    lines.append("inference:")
    lines.append(f"  enabled: {str(inf['enabled']).lower()}")
    lines.append(f"  confidence_level: {inf['confidence_level']}")
    lines.append(f"  fdr_correction: {str(inf['fdr_correction']).lower()}")
    lines.append(f'  outcome_test: "{inf["outcome_test"]}"')
    lines.append(f"  cox_penalizer: {inf['cox_penalizer']}")
    lines.append("")

    rp = config["reference_phenotype"]
    lines.append("reference_phenotype:")
    lines.append(f'  strategy: "{rp["strategy"]}"')
    lines.append("  # specific_id: 0")
    lines.append('  # health_outcome: "mortality_within_30_days"')
    lines.append("")

    vis = config["visualization"]
    lines.append("visualization:")
    lines.append(f"  save_plots: {str(vis['save_plots']).lower()}")
    lines.append(f"  dpi: {vis['dpi']}")
    lines.append("")

    lg = config["logging"]
    lines.append("logging:")
    lines.append(f'  level: "{lg["level"]}"')
    lines.append(f'  format: "{lg["format"]}"')
    lines.append(f"  log_to_file: {str(lg['log_to_file']).lower()}")
    lines.append(f'  log_file: "{lg["log_file"]}"')
    lines.append(f"  quiet_mode: {str(lg['quiet_mode']).lower()}")
    lines.append("")

    ca = config["cache"]
    lines.append("cache:")
    lines.append(f"  enabled: {str(ca['enabled']).lower()}")
    lines.append(f"  compress_level: {ca['compress_level']}")
    lines.append("")

    dq = config["data_quality"]
    lines.append("data_quality:")
    lines.append(f"  enabled: {str(dq['enabled']).lower()}")
    lines.append(f"  missing_threshold: {dq['missing_threshold']}")
    lines.append(f"  correlation_threshold: {dq['correlation_threshold']}")
    lines.append(f"  variance_threshold: {dq['variance_threshold']}")
    lines.append(f"  generate_report: {str(dq['generate_report']).lower()}")
    lines.append("")

    cf = config["categorical_flow"]
    lines.append("categorical_flow:")
    lines.append(f"  group_by_prefix: {str(cf['group_by_prefix']).lower()}")
    lines.append(f'  prefix_separator: "{cf["prefix_separator"]}"')
    lines.append("  custom_groups: {}")
    lines.append(f"  show_sankey: {str(cf['show_sankey']).lower()}")
    lines.append(f"  show_proportion_heatmap: {str(cf['show_proportion_heatmap']).lower()}")
    lines.append(f"  min_category_pct: {cf['min_category_pct']}")
    lines.append("")

    fc = config["feature_characterization"]
    lines.append("feature_characterization:")
    lines.append(f"  group_by_prefix: {str(fc['group_by_prefix']).lower()}")
    lines.append(f'  prefix_separator: "{fc["prefix_separator"]}"')
    lines.append("  custom_groups: {}")
    lines.append(f"  n_top_per_group: {fc['n_top_per_group']}")
    lines.append(f"  n_top_overall: {fc['n_top_overall']}")
    lines.append("")

    output_path.write_text("\n".join(lines))
