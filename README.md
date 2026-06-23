# divergence_prediction

Predict pairwise KL divergence between station flow duration curve PMFs from catchment attributes.

The repository represents the supporting information for KL-based predictability analysis and KL divergence prediction model training.

## Scope

- Data ingestion and quality checks for HYSETS discharge and catchment descriptors
- Baseline PMF estimation per station (`https://github.com/dankovacek/distribution_estimation`)
- Pairwise KL divergence target generation
- Out of sample entropy prediction (5-fold CV) from catchment attributes
- Graph partitioning for pairwise cross validation fold construction
- Pairwise out-of-sample KL divergence prediction from catchment attributes
- Visualization and interpretation in a Jupyter Book

Published book:
[https://dankovacek.github.io/divergence_prediction](https://dankovacek.github.io/divergence_prediction)

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
python -m pip install uv
uv sync
```

## Precursor analyses and execution order

Run notebooks (`divergence_prediction/notebooks/`) in order because later stages depend on files produced earlier.

1. `1_data.ipynb`
2. `2_Process_Baseline.ipynb`
3. `3_Methods.ipynb`
4. `4_Predictability_of_entropy.ipynb`
5. `5_Predictability_of_KL_Divergence.ipynb`
6. `6_Predicted_Predictability.ipynb`

## Files generated locally and not tracked

Large derived artifacts are intentionally excluded from git to keep repository size manageable.
This project does not rely on Git LFS for these outputs.

Generated directories (under `divergence_prediction/notebooks/`) that may need regeneration:

- `data/baseline_distributions/`
- `data/fdc_estimation_results/`
- `data/kld_batches/`
- `data/parameter_prediction_results/`
- `data/pmf_entropy/`
- `data/results/`
- `images/`
- `results/`

Book build output is generated locally at `divergence_prediction/_build/` and is not tracked in git.

If these artifacts are missing, rerun precursor notebooks in order.

## Build and publish the book

```bash
uv run jupyter-book build divergence_prediction/
uv run ghp-import -n -p -f divergence_prediction/_build/html
```

## Citation

Use the publication citation and supporting references in `divergence_prediction/references.bib`.
