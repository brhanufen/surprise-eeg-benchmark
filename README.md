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
  DECISIONS.md             # Decision log
  data/
    raw/                   # ERP CORE BIDS data (downloaded)
    processed/             # Preprocessed epochs (.fif)
    regressors/            # Trial-wise surprise tables (.csv)
  src/
    download_erp_core.py   # Data download script
    run_pipeline.py        # Master pipeline
    preprocessing/         # MNE preprocessing
    surprise/              # Four surprise estimators
    encoding/              # Mixed-effects encoding models
    decoding/              # Classification benchmark
    figures/               # Publication figure generation
  results/
    aim1/                  # QC, correlations, surprise summaries
    aim2/                  # Encoding results, time-resolved
    aim3/                  # Decoding results
  figures/                 # Publication-ready figures
  manuscript/              # Manuscript draft
```

## Surprise Models

1. **Static Shannon**: -log p_global(x_t)
2. **Adaptive Shannon**: -log p_w(x_t) with sliding window
3. **Bayesian**: KL divergence between successive Beta posteriors
4. **Change-point**: Predictive surprise under Adams & MacKay (2007)

## Key Results

- Bayesian surprise shows the strongest trend toward predicting MMN amplitude (uncorrected p = 0.015, Holm-corrected p = 0.062)
- A similar trend emerged in the P3 paradigm for P3b amplitude (Bayesian p = 0.044, change-point p = 0.017), but no model survives correction
- All partial R² values are extremely small (<0.02%); cross-validated prediction shows no gain
- High multicollinearity (VIF > 70) limits Shannon vs. change-point comparisons
- Contextual surprise does not improve cross-subject decoding (ΔAUC = −0.093)

## Code

https://github.com/brhanufen/surprise-eeg-benchmark

## Citation

Kappenman, E. S., Farrens, J. L., Luck, S. J., & Proudfit, G. H. (2021). ERP CORE: An open resource for human event-related potential research. NeuroImage, 225, 117465.

## License

CC BY-SA 4.0 (following ERP CORE license)
