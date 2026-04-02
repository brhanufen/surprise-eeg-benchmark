# Surprise EEG Benchmark

A reproducible benchmark comparing four stochastic surprise formulations as predictors of single-trial EEG prediction-error responses using the ERP CORE dataset.

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate surprise-eeg-benchmark

# 2. Download data
python src/download_erp_core.py

# 3. Run full pipeline (MMN paradigm)
python src/run_pipeline.py --task MMN

# 4. Run P3 internal replication
python src/run_pipeline.py --task P3

# 5. Generate figures
python src/figures/make_figures.py MMN
```

## Repository Structure

```
surprise-eeg-benchmark/
  environment.yml          # Pinned conda environment
  run_all.sh               # Full pipeline script
  DECISIONS.md             # Analysis decision log
  src/
    download_erp_core.py   # Data download
    run_pipeline.py        # Master pipeline
    preprocessing/         # MNE preprocessing
    surprise/              # Four surprise estimators
    encoding/              # Mixed-effects encoding models
    decoding/              # Classification benchmark
    analyses/              # Supplementary analyses
    figures/               # Figure generation
  figures/                 # Publication figures
```

## Surprise Models

1. **Static Shannon**: −log p_global(x_t)
2. **Adaptive Shannon**: −log p_w(x_t) with sliding window
3. **Bayesian**: KL divergence between successive Beta posteriors
4. **Change-point**: Predictive surprise under Adams & MacKay (2007)

## Key Results

- Bayesian surprise shows the strongest association with MMN amplitude (uncorrected p = 0.015, Holm-corrected p = 0.062)
- Similar trend in the P3 paradigm (Bayesian p = 0.044, change-point p = 0.017); no model survives correction
- All partial R² values < 0.02%; cross-validated prediction shows no gain
- High multicollinearity (VIF > 70) limits model comparisons in stationary paradigms
- Contextual surprise does not improve cross-subject decoding

## Dataset

ERP CORE (Kappenman et al., 2021): N = 40 participants, auditory MMN and visual P3 paradigms. N = 38 analyzed for MMN, N = 36 for P3 after artifact-based exclusions.

## Citation

Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021). ERP CORE: An open resource for human event-related potential research. NeuroImage, 225, 117465.

## License

CC BY-SA 4.0
