Statistical Methods
===================

This page describes the statistical methods used in the PhenoCluster pipeline.

Latent Class / Profile Analysis
-------------------------------

PhenoCluster uses the `StepMix <https://github.com/Labo-Lacourse/stepmix>`_
framework for latent variable mixture modelling --- Latent Class Analysis (LCA)
for categorical indicators, Latent Profile Analysis (LPA) for continuous
indicators, or a mixed-indicator model when both types are present. Missing
values are handled via Full Information Maximum Likelihood (FIML).

The optimal number of latent classes is selected by minimising an information
criterion (BIC by default) over a user-specified range of cluster counts.
Phenotype labels are reordered by cluster size so that the largest cluster is
always Phenotype 0.

Preprocessing and data leakage prevention
------------------------------------------

The train/test split is performed before any preprocessing. Imputation,
outlier handling, categorical encoding, and continuous variable
standardisation are fit exclusively on the training set. The learned
parameters are then applied to the test set for unbiased model-selection
validation. Once the optimal number of phenotypes is determined, the
entire preprocessing pipeline and the LCA model are refitted on the full
cohort for the final descriptive analysis.

Phenotype assignment and downstream inference
----------------------------------------------

Downstream outcome and survival analyses use modal (MAP) class assignment:
each patient is assigned to the phenotype with the highest posterior
membership probability. This approach is straightforward but may
attenuate effect estimates (odds ratios, hazard ratios) toward the null
when classification quality is low. The Classification Quality section
of the report provides per-phenotype Average Posterior Probability (AvePP)
and assignment confidence metrics to help assess the magnitude of this
potential attenuation.

Statistical inference
---------------------

Downstream inference uses classical statistical methods.

**Outcome models** use logistic regression (statsmodels GLM with binomial
family), yielding odds ratios with Wald confidence intervals and p-values.

**Survival models** use Cox proportional hazards regression (lifelines
CoxPHFitter), yielding hazard ratios with confidence intervals and Wald
p-values. Log-rank tests assess overall and pairwise survival differences
between phenotypes.

**Binary outcome associations** are additionally tested with chi-square or
Fisher's exact tests (auto-selected based on expected cell counts).

Multiple comparison correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benjamini-Hochberg FDR correction is applied globally across all comparisons
within each analysis type (outcomes, survival, multistate transitions). This
controls the false discovery rate when testing multiple phenotype comparisons
and endpoints simultaneously.

Multistate modelling
--------------------

Transition-specific Cox PH models are fitted per transition using Cox proportional hazards
(`lifelines <https://lifelines.readthedocs.io/>`_). Proportional hazards
assumptions are checked using Schoenfeld residuals.

Cox PH models produce hazard ratios with confidence intervals and Wald p-values
for each transition, with FDR correction applied across all transitions.

Monte Carlo simulation generates patient trajectories using the fitted Cox PH
models:

1. Bootstrap resampling of model coefficients provides uncertainty quantification
2. For each bootstrap draw, multiple trajectories are simulated: transition
   hazards determine next-state probabilities (multinomial sampling) and
   transition times (inverse CDF from baseline hazard)
3. Trajectories within each draw are averaged (removing MC stochastic noise),
   then the distribution across draws captures parameter uncertainty

Missing data in downstream analyses
------------------------------------

Outcome and survival analyses use complete-case analysis: patients with
missing values in the relevant outcome or time-to-event columns are excluded
from that specific analysis. This assumes Missing Completely at Random (MCAR).
If outcome missingness is related to patient characteristics (MAR/MNAR),
estimates may be biased. The data quality report provides missingness
diagnostics.


References
----------

- Morin, S., Legault, R., Laliberte, F., Bakk, Z., Giguere, C.-E.,
  de la Sablonniere, R., & Lacourse, E. (2025). StepMix: A Python Package for
  Pseudo-Likelihood Estimation of Generalized Mixture Models with External
  Variables. *Journal of Statistical Software*, 113(8), 1--39.
  https://doi.org/10.18637/jss.v113.i08
