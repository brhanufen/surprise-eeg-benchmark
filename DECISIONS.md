# Analysis Decision Log

## Dataset
- **Decision:** ERP CORE (Kappenman et al., 2021) from OSF
- **Rationale:** 40 subjects, published reference results, standardized paradigms
- **Exclusions:** sub-012 missing from original release; sub-007 excluded for 85.4% artifact rejection (N = 38 for MMN); sub-003, sub-005, sub-032 excluded for >80% rejection (N = 36 for P3)

## Event Codes
- **MMN:** 80 = standard, 70 = deviant, 180 = first standards stream (excluded)
- **P3:** 11–55 = stimuli; target = tens digit matches units digit (11, 22, 33, 44, 55); 201/202 = responses (excluded)

## Preprocessing
- **Decision:** 0.1–30 Hz FIR bandpass, 256 Hz resampling, average reference, ICA + amplitude rejection (±150 µV)
- **Note:** Autoreject (Jas et al., 2017) failed due to missing electrode positions in ERP CORE .set files; amplitude threshold used as fallback

## Change-Point Model
- **Decision:** Use predictive surprise (−log P(x|model)) rather than change-point posterior probability
- **Rationale:** Posterior probability is constant (= hazard rate) on stationary sequences; predictive surprise marginalizes over run lengths and has meaningful trial-by-trial variance
- **Note:** Predictive surprise correlates r = 0.99 with static Shannon due to sequence stationarity

## Multicollinearity
- **Decision:** Individual-model-vs-baseline comparisons as primary analysis; no combined models
- **Rationale:** VIF: static Shannon = 70.3, change-point = 105.9, adaptive Shannon = 17.3; only Bayesian surprise (VIF = 1.1) is independent

## Decoding
- **Decision:** Residualize surprise against stimulus type; cross-subject CV as primary evaluation
- **Rationale:** Surprise regressors are deterministic functions of the stimulus sequence; within-subject AUC = 1.0 trivially

## Hazard Rate
- **Decision:** Primary h = 1/200; sensitivity over h ∈ {1/50, 1/100, 1/500}
- **Result:** Minimal impact on results due to sequence stationarity
