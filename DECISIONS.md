# Decision Log

## 2026-03-30: Initial analysis decisions

### Dataset
- **Decision:** Use ERP CORE (Kappenman et al., 2021) from OSF
- **Rationale:** 40 subjects, published reference results, standardized paradigms
- **Note:** sub-012 missing from original release; N=39 usable

### Event codes (verified from task-*_events.json)
- **MMN:** 80=standard, 70=deviant, 180=first standards stream (excluded)
- **P3:** 11-55 are stimuli; target = tens digit matches units digit (11,22,33,44,55); 201/202 are responses (excluded from epoching)

### Preprocessing
- **Decision:** 0.1-30 Hz FIR bandpass, 256 Hz resampling, average reference, ICA + amplitude rejection
- **Note:** Autoreject (Jas et al., 2017) failed due to missing electrode positions in ERP CORE .set files; used ±150 µV amplitude threshold as fallback
- **Impact:** May retain more artifacts than autoreject; conservative threshold mitigates this

### Change-point model viability (Week 0 check)
- **Decision:** Change-point posterior probability is constant (= hazard rate) on stationary sequences; use predictive surprise instead
- **Rationale:** Predictive surprise marginalizes over run lengths and has meaningful trial-by-trial variance (SD > 0.8)
- **Note:** However, predictive surprise correlates r=0.99 with static Shannon due to sequence stationarity

### Multicollinearity handling
- **Decision:** Individual-model-vs-baseline is PRIMARY analysis; no combined models run
- **Rationale:** VIF values: static Shannon=70.3, change-point=105.9, adaptive Shannon=17.3; all exceed threshold of 10 except Bayesian surprise=1.1
- **Impact:** Cannot directly compare Shannon vs change-point models; comparison is only meaningful for Bayesian vs baseline

### Decoding label leakage
- **Decision:** Residualize surprise against stimulus type for decoding; designate cross-subject as PRIMARY
- **Rationale:** Surprise regressors are deterministic functions of stimulus sequence; within-subject AUC=1.0 trivially
- **Impact:** Cross-subject shows surprise does NOT improve decoding (ΔAUC = -0.093 for MMN)

### Hazard rate
- **Decision:** Primary h=1/200; sensitivity over h ∈ {1/50, 1/100, 1/500}
- **Rationale:** h=1/200 implies expected run length of 200 trials (~1 change per block)
- **Result:** Hazard rate has minimal impact on results due to sequence stationarity

### Target journal
- **Decision:** Psychophysiology (Wiley) — primary target
- **Rationale:** $0 APC under UNL BTAA Read & Publish agreement; core ERP audience
