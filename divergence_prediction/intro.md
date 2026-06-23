# Supporting Information: KL Divergence Predictability from Catchment Attributes

This book documents the reproducible workflow used to estimate and predict pairwise KL divergence between station flow duration curve PMFs from catchment attributes.

## Purpose

Provide a transparent supporting-information record for:

- Data preparation and quality controls
- Baseline PMF estimation
- KL divergence target construction
- Predictive modeling of entropy and KL divergence
- Interpretation of predictability outcomes

## Precursor analyses

Results in later chapters depend on artifacts generated earlier in the sequence below:

1. Data loading and basic sanity checks
2. Baseline PMF generation and pairwise KL target construction
3. Methods and diagnostics
4. Entropy predictability modeling
5. KL divergence predictability modeling
6. Predictability analysis from modeled KL divergence

## Generated artifacts

Large derived files are not included in version control and are regenerated locally as needed.
This repository avoids Git LFS for generated analysis outputs.

Results / dependencies folders that may need regeneration (`divergence_prediction/notebooks/`):

- `data/baseline_distributions/`
- `data/fdc_estimation_results/`
- `data/kld_batches/`
- `data/parameter_prediction_results/`
- `data/pmf_entropy/`
- `data/results/`

Book build output is generated locally at `divergence_prediction/_build/` and is not tracked in git.

If missing, rerun upstream precursor notebooks before downstream analyses.

```{tableofcontents}
```
