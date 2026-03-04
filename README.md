<p align="center">
  <img src="docs/phenocluster_logo.png" alt="PhenoCluster" width="280"/>
</p>

<p align="center">
  <strong>A flexible data-driven framework for identifying clinical phenotypes using latent class and profile analysis</strong>
</p>

[![PyPI version](https://img.shields.io/pypi/v/phenocluster)](https://pypi.org/project/phenocluster/)
[![Python versions](https://img.shields.io/pypi/pyversions/phenocluster)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/EttoreRocchi/phenocluster/actions/workflows/ci.yml/badge.svg)](https://github.com/EttoreRocchi/phenocluster/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ettorerocchi.github.io/phenocluster)

---

## Overview

PhenoCluster is a Python framework for unsupervised discovery of clinical phenotypes from heterogeneous patient data. It implements an end-to-end pipeline: from data preprocessing and latent class identification to outcome association analysis, survival modelling, and multistate transition modelling.

The framework is **domain-agnostic** and can be applied to any clinical cohort study where the goal is to identify latent patient subgroups and characterise their relationship with clinical outcomes. Users supply a dataset and a YAML configuration file; PhenoCluster handles model selection, phenotype assignment, and downstream inference automatically.

### Key capabilities

- **Latent Class / Profile Analysis** via the [StepMix](https://github.com/Labo-Lacourse/stepmix) framework with native support for mixed continuous/categorical data and missing values
- **Automatic model selection** using information criteria (BIC, AIC, ICL, CAIC, SABIC) with configurable cluster-size constraints
- **Classification quality assessment** with per-phenotype Average Posterior Probability (AvePP) and assignment confidence metrics
- **Outcome association analysis** with logistic regression yielding odds ratios, confidence intervals, and FDR-corrected p-values
- **Survival analysis** with Cox proportional hazards models producing hazard ratios and log-rank tests
- **Multistate modelling** with transition-specific Cox PH analysis, Monte Carlo simulation for state occupation probabilities with confidence interval bands, and clinical pathway enumeration
- **Comprehensive output** including an interactive HTML report, forest plots with confidence intervals, Kaplan-Meier and Nelson-Aalen curves, heatmaps, and JSON/CSV data exports

## Installation

> **Requires Python ≥ 3.11**

### From PyPI

```bash
pip install phenocluster
```

### From source

```bash
git clone https://github.com/EttoreRocchi/phenocluster.git
cd phenocluster
pip install -e ".[dev]"
```

## Quick start

### 1. Generate a configuration file

```bash
phenocluster create-config -p complete -o config.yaml
```

### 2. Edit the configuration

Open `config.yaml` and fill in your dataset-specific parameters:

```yaml
global:
  project_name: "My Study"
  output_dir: "results"
  random_state: 42

data:
  continuous_columns:
    - age
    - bmi
    - lab_value_1
  categorical_columns:
    - sex
    - smoking_status
    - disease_stage
  split:
    test_size: 0.2

outcome:
  enabled: true
  outcome_columns:
    - mortality_30d
    - readmission_30d

survival:
  enabled: true
  targets:
    - name: "overall_survival"
      time_column: "time_to_death"
      event_column: "death_indicator"
```

### 3. Run the pipeline

```bash
phenocluster run -d data.csv -c config.yaml
```

### 4. Inspect results

Results are written to the output directory (default: `results/`):

| File | Description |
|------|-------------|
| `analysis_report.html` | Comprehensive HTML report with all results and visualisations |
| `cluster_statistics.json` | Phenotype sizes, feature distributions, and classification quality |
| `outcome_results.json` | Odds ratios with confidence intervals and p-values |
| `survival_results.json` | Kaplan-Meier estimates and Cox PH hazard ratios |
| `multistate_results.json` | Transition-specific hazard ratios, pathways, and state occupation |
| `data/model_fit_metrics.csv` | Information criteria, entropy, and average posterior probabilities |
| `data/phenotypes_data.csv` | Original data augmented with phenotype assignments |
| `data/posterior_probabilities.csv` | Posterior class membership probabilities |
| `results/model_selection_summary.json` | Model selection comparison table and best model info |
| `results/feature_importance.json` | Feature characterisation per phenotype |
| `results/validation_report.json` | Internal validation metrics (train/test comparison) |
| `results/stability_results.json` | Consensus clustering stability metrics |
| `results/split_info.json` | Train/test split details |
| `results/external_validation_results.json` | External validation results (when enabled) |
| `phenocluster.log` | Pipeline execution log |
| `artifacts/` | Cached intermediate results for incremental re-runs |

## Pipeline overview

PhenoCluster executes the following stages in order:

1. **Data quality assessment.** Missingness patterns, correlations, variance, and MCAR testing.
2. **Train/test split.** Stratified splitting with configurable test size, performed before preprocessing to prevent data leakage.
3. **Preprocessing.** Imputation, outlier handling, categorical encoding, standardization, and feature selection -- fit on training data only, then applied to the test set.
4. **Model selection.** Cross-validated information criterion search over cluster counts (training set only).
5. **Full-cohort refit.** Once K is selected, preprocessing and LCA/LPA model are refitted on the entire cohort; phenotypes reordered by size (largest = Phenotype 0).
6. **Stability analysis.** Consensus clustering over subsampled runs.
7. **Internal validation.** Train/test log-likelihood comparison, cluster proportion stability, and outcome OR consistency.
8. **Outcome association.** Logistic regression for binary outcomes with FDR-corrected p-values (optional).
9. **Survival analysis.** Kaplan-Meier curves, Nelson-Aalen estimators, log-rank tests, and Cox PH hazard ratios (optional).
10. **Multistate modelling.** Transition-specific Cox PH models, transition hazard ratios, and Monte Carlo simulation (optional).
11. **Report generation.** Interactive HTML report with all figures and tables.

## CLI reference

| Command | Description |
|---------|-------------|
| `phenocluster run -d DATA -c CONFIG [--force-rerun]` | Run the full pipeline |
| `phenocluster create-config [-p PROFILE] [-o OUTPUT]` | Generate a config YAML from a profile template |
| `phenocluster validate-config -c CONFIG [-d DATA]` | Validate config structure; cross-check columns against data |
| `phenocluster version` | Show version, repository link, and documentation link |

## Configuration profiles

Profiles set sensible defaults for common use-cases. Generate one with `phenocluster create-config -p <profile>`:

| Profile | Description | Inference | Stability | Multistate |
|---------|-------------|:---------:|:---------:|:----------:|
| `descriptive` | Phenotype discovery only, no statistical inference | off | on | off |
| `complete` | All analyses enabled (outcomes, survival, multistate) | on | on | on |
| `quick` | Fast iteration for development | on | off | off |

## Configuration reference

See the full [Configuration Reference](https://ettorerocchi.github.io/phenocluster/configuration.html) in the documentation.

## Documentation

Full documentation (statistical methods, configuration reference, output descriptions) is available at **[ettorerocchi.github.io/phenocluster](https://ettorerocchi.github.io/phenocluster)**.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

This project is licensed under the [MIT](LICENSE) License.

## Citation

If you use **PhenoCluster** in your research, please cite:

```bibtex

```

## Acknowledgment

This project relies on **StepMix**, a Python package for pseudo-likelihood estimation of generalized mixture models with external variables. We thank the authors for making their work openly available.

If you use this framework, please cite also:

> Morin, S., Legault, R., Laliberté, F., Bakk, Z., Giguère, C.-É., de la Sablonnière, R., & Lacourse, É. (2025). StepMix: A Python Package for Pseudo-Likelihood Estimation of Generalized Mixture Models with External Variables. Journal of Statistical Software, 113(8), 1-39. doi: [10.18637/jss.v113.i08](https://doi.org/10.18637/jss.v113.i08)