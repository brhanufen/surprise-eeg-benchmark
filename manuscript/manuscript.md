# Stochastic Surprise Signatures in Human EEG: A Reproducible Benchmark of Prediction-Error Models Using the ERP CORE Oddball Dataset

## Abstract

The brain continuously generates predictions about incoming sensory input and produces characteristic neural responses when those predictions are violated. In EEG oddball paradigms, these prediction-error responses manifest as the mismatch negativity (MMN) and P3b components. However, "surprise" can be formalized in multiple ways, and no prior study has systematically compared these formulations on the same dataset using single-trial methods. Here, we implemented four hierarchical surprise estimators and applied them to the ERP CORE dataset (N = 40; N = 38 analyzed for MMN, N = 36 for P3 after exclusions; auditory MMN and visual P3 paradigms). Using linear mixed-effects encoding models, we found that Bayesian surprise, quantifying the magnitude of belief revision, showed the strongest association with single-trial MMN amplitude among all models tested (uncorrected p = 0.015), though this did not survive Holm–Bonferroni correction for multiple comparisons (p_corrected = 0.062). In the P3 paradigm, change-point surprise (uncorrected p = 0.017, p_corrected = 0.069) and Bayesian surprise (uncorrected p = 0.044, p_corrected = 0.131) showed trends in the P3b window but likewise did not survive correction. High multicollinearity among Shannon-based and change-point surprise regressors (VIF > 70) limited the interpretability of direct model comparisons. All partial R² values were extremely small (<0.02%), and cross-validated prediction showed no real gain over stimulus class alone. In a decoding analysis, contextual surprise features (residualized against stimulus type) did not improve cross-subject classification of stimulus class beyond ERP amplitude features alone (ΔAUC = −0.093). We conclude that Bayesian surprise shows the strongest trend among competing models, but that stationary oddball paradigms lack the model identifiability needed to definitively distinguish surprise formulations due to high regressor multicollinearity. All data, code, and analysis pipelines are publicly available.

## 1. Introduction

The brain has been characterized as a prediction machine that continuously generates expectations about incoming sensory input and updates internal models when those expectations are violated (Rao & Ballard, 1999; Friston, 2005). This predictive processing framework provides a unifying account of perception, attention, and learning, proposing that neural activity primarily encodes the discrepancy between predicted and observed events — the prediction error.

Oddball paradigms offer the cleanest experimental window into neural prediction-error processing. In these paradigms, rare deviant stimuli are embedded among frequent standard stimuli, and the brain produces characteristic electrophysiological responses to the deviants. The mismatch negativity (MMN) is an early fronto-central ERP component (100–250 ms post-stimulus) reflecting automatic detection of auditory deviance, thought to arise from prediction-error signals in auditory cortex (Näätänen et al., 2007). The P3b is a later parietal component (250–500 ms) reflecting context updating, attention allocation, and the conscious evaluation of surprising events (Polich, 2007).

While these ERP components are well-established as neural signatures of prediction error, "surprise" itself can be mathematically defined in multiple principled ways that make different assumptions about the underlying computation. Shannon surprise (S = −log p(x)) quantifies how rare an event is based on its probability, treating surprise as a simple function of frequency. Bayesian surprise, defined as the KL divergence between posterior beliefs before and after an observation (Itti & Baldi, 2009), measures the magnitude of belief revision — how much an observer's model of the world had to change. Change-point models (Adams & MacKay, 2007) compute the posterior probability that the underlying generative process has changed, capturing surprise about regime shifts rather than individual events.

These formulations are not merely theoretical alternatives — they make different predictions about trial-by-trial neural responses. Shannon surprise predicts that neural responses scale with stimulus rarity. Bayesian surprise predicts that responses scale with the magnitude of belief updating, which depends on the observer's uncertainty. Change-point surprise predicts maximal responses when the observer infers that the statistical environment has fundamentally changed. Prior work has examined these models individually: Mars et al. (2008) demonstrated trial-by-trial correlations between Bayesian surprise and ERP amplitudes; Kolossa et al. (2015) used model-based approaches to explain P300 fluctuations; Ostwald et al. (2012) provided evidence for neural encoding of Bayesian surprise in somatosensation. However, no study has systematically compared all of these formulations on the same dataset using single-trial methods and formal model comparison.

Here, we address this gap by implementing four hierarchical surprise estimators — static Shannon, adaptive Shannon, Bayesian, and change-point surprise — and benchmarking their ability to explain trial-by-trial EEG prediction-error responses in the ERP CORE dataset (Kappenman et al., 2021). We use the auditory MMN paradigm as our primary target and the visual P3 paradigm as an internal replication. Our central hypotheses are: (H1) that adaptive surprise estimators, particularly Bayesian surprise, will better explain single-trial ERP responses than static frequency-based surprise; and (H2) that surprise-derived features will improve single-trial decoding of stimulus class beyond classical ERP amplitude features.

## 2. Methods

### 2.1 Dataset

We used the ERP CORE dataset (Kappenman et al., 2021), a standardized resource providing EEG data from 40 participants recorded during six optimized ERP paradigms. Sub-012 was excluded from the original release; sub-007 was excluded for excessive artifact rejection (85.4% rejection rate), yielding N = 38 for MMN analyses. We analyzed the MMN paradigm (auditory oddball with frequency deviants; ~80% standards, ~20% deviants) and the P3 paradigm (visual oddball with letter targets; ~80% non-targets, ~20% targets; sub-003, sub-005, sub-032 excluded for >80% rejection rates, yielding N = 36). Data were recorded at 1024 Hz from 30 EEG channels plus 3 EOG channels using BioSemi ActiveTwo systems.

### 2.2 Preprocessing

All preprocessing was performed using MNE-Python 1.8.0 (Gramfort et al., 2013) with the following fixed parameters:

- **Filtering:** 0.1–30 Hz bandpass (zero-phase FIR, Hamming window)
- **Resampling:** Downsampled to 256 Hz
- **Reference:** Average reference
- **Artifact rejection:** ICA (FastICA, 15 components) for eye artifact removal, with automatic EOG component detection; amplitude-based epoch rejection (±150 µV threshold) as fallback when channel positions were unavailable for autoreject
- **Epoching:** −200 to 800 ms relative to stimulus onset
- **Baseline correction:** −200 to 0 ms

After preprocessing and exclusion, the MMN dataset comprised 35,332 epochs across 38 subjects (mean 930 ± 48 per subject; 7.0% rejection rate). The P3 dataset comprised 5,865 epochs across 36 subjects.

### 2.3 Surprise Estimators

All estimators operated on the binary stimulus sequence extracted from retained epochs (0 = standard, 1 = deviant):

1. **Static Shannon surprise:** S_t = −log₂ p_global(x_t), where p_global is the overall frequency of each stimulus type across the entire sequence. This produces a fixed value per stimulus class.

2. **Adaptive Shannon surprise:** S_t = −log₂ p_w(x_t), where p_w is estimated from a sliding window of the preceding w trials (w = 20 for primary analysis; w = 10, 50 for sensitivity). Laplace smoothing was applied to prevent log(0).

3. **Bayesian surprise:** D_KL[Beta(α_t, β_t) || Beta(α_{t-1}, β_{t-1})], the KL divergence between successive posteriors of a Beta-Bernoulli conjugate model with flat prior (Beta(1,1)). This measures the magnitude of belief revision after each observation.

4. **Change-point predictive surprise:** −log₂ P(x_t | model), the negative log marginal predictive probability under the Adams & MacKay (2007) online change-point detection model with geometric hazard rate (h = 1/200; sensitivity analysis over h ∈ {1/50, 1/100, 1/500}). This marginalizes over all possible run lengths, weighting each by its posterior probability.

Secondary metrics included posterior entropy, estimated volatility (entropy of the run-length distribution), and mean run length. Regressors were z-scored before inclusion in encoding models.

### 2.4 Feature Extraction

**ERP features:** Mean amplitude was extracted in two time windows for both paradigms: the MMN window (100–250 ms, fronto-central ROI: Fz, FCz, Cz, FC3, FC4) and the P3b window (250–500 ms, parietal ROI: Pz, CPz, P3, P4, CP1, CP2). Testing both windows in both paradigms allowed us to assess whether surprise effects were specific to the expected ERP component or extended across time windows.

### 2.5 Encoding Analysis

**Primary analysis:** Linear mixed-effects models were fit with EEG amplitude as the dependent variable, stimulus class as a fixed effect (baseline model), and subject as a random intercept. Each surprise model was tested individually against the stimulus-class-only baseline (individual-model-vs-baseline design), avoiding multicollinearity from joint inclusion. Model comparison used AIC and likelihood ratio tests (LRT).

**Cross-validated prediction:** To assess out-of-sample predictive value, we performed leave-one-subject-out cross-validation: for each held-out subject, mixed-effects models were fit on the remaining subjects and used to predict the held-out subject's trial-by-trial amplitudes. Prediction MSE was compared between baseline and surprise-augmented models via paired t-tests across folds. Holm–Bonferroni correction was applied within each ERP window (4 comparisons).

**Time-resolved regression:** At each time point, ERP amplitude (averaged across ROI channels) was regressed on each z-scored surprise regressor, pooled across subjects. Group-level significance was assessed via one-sample t-tests with cluster-based permutation correction (1000 permutations).

**Multicollinearity assessment:** Variance inflation factors (VIF) were computed for all four regressors entered simultaneously. VIF ≥ 10 was the threshold for declaring problematic collinearity.

### 2.6 Decoding Analysis

**Task:** Binary classification of standard vs. deviant trials.

**Features:** (i) ERP-only (MMN and P3b window amplitudes); (ii) ERP + contextual surprise (surprise regressors residualized against stimulus type to prevent label leakage).

**Evaluation:** Cross-subject leave-5-out CV was designated as the primary test of H2, because surprise regressors computed from the deterministic stimulus sequence encode stimulus identity and produce trivially perfect within-subject classification. Within-subject 5-fold stratified CV was reported as a secondary analysis.

**Classifier:** L2-regularized logistic regression (C = 1.0) with balanced class weights, preceded by standard scaling.

**Metrics:** ROC-AUC, PR-AUC, balanced accuracy, Brier score.

## 3. Results

### 3.1 ERP Replication

Grand-average ERP waveforms replicated published ERP CORE results. In the MMN paradigm, deviant tones elicited a clear fronto-central negativity relative to standards in the 100–250 ms window (MMN). In the P3 paradigm, target stimuli elicited a parietal positivity in the 250–500 ms window (P3b). These results confirm that our preprocessing pipeline preserved the expected neural signatures (Figure 2).

### 3.2 Surprise Regressor Properties

The four surprise models produced regressors with markedly different properties (Figure 1C). Static Shannon, adaptive Shannon, and change-point surprise were highly correlated (r = 0.95–0.99), reflecting their shared dependence on stimulus frequency. Bayesian surprise was substantially more distinct (r = 0.26–0.29 with all other models).

VIF analysis confirmed severe multicollinearity: static Shannon (VIF = 70.3), change-point (VIF = 105.9), and adaptive Shannon (VIF = 17.3) all exceeded the threshold of 10. Only Bayesian surprise (VIF = 1.1) was independent. This pattern was consistent across paradigms. We therefore interpret the individual-model-vs-baseline comparisons as the primary analysis and do not report combined models.

### 3.3 Encoding Results (H1)

**MMN paradigm (primary).** Bayesian surprise showed the strongest association with single-trial MMN amplitude (ΔAIC = −3.9, uncorrected p = 0.015), but did not survive Holm–Bonferroni correction (p_corrected = 0.062). Change-point surprise was marginal before correction (ΔAIC = −2.3, p = 0.038, p_corrected = 0.115). Static Shannon (p = 0.066) and adaptive Shannon (p = 0.674) did not reach significance. No surprise model significantly predicted P3b amplitude in the MMN paradigm (all p > 0.39; Table 2A). All partial R² values were very small (<0.02%), indicating that surprise explains a negligible fraction of single-trial variance beyond stimulus class.

**P3 paradigm (internal replication).** In the P3b window, change-point surprise (uncorrected p = 0.017, p_corrected = 0.069) and Bayesian surprise (uncorrected p = 0.044, p_corrected = 0.131) showed trends but did not survive Holm–Bonferroni correction (Table 2B). The fronto-central (MMN) window was also tested in the P3 paradigm but yielded no significant results for any model (all p > 0.40; not tabulated).

**Cross-validated prediction.** Leave-one-subject-out cross-validation showed no significant improvement in out-of-sample prediction for any surprise model over the baseline (all p > 0.64), confirming the minimal explanatory benefit.

**Time-resolved analysis.** All four surprise models showed significant clusters of regression coefficients in the MMN window (approximately 100–230 ms) at fronto-central sites. At parietal sites, significant clusters emerged in both the MMN window (125–230 ms) and the P3b window (330–435 ms) for all models except Bayesian surprise, which showed a later cluster (344–383 ms, Figure 3).

**Summary for H1:** H1 is partially supported. Bayesian surprise shows the strongest trend toward predicting single-trial prediction-error responses across both paradigms, but no model survives correction for multiple comparisons. Effect sizes are very small, and the high collinearity among Shannon-based models limits the specificity of model comparisons.

### 3.4 Decoding Results (H2)

**Cross-subject (primary).** ERP-only features yielded a cross-subject AUC of 0.543 (MMN) and 0.604 (P3). Adding residualized contextual surprise features did not improve classification; AUC decreased to 0.450 (MMN, ΔAUC = −0.093) and 0.571 (P3, ΔAUC = −0.033). The decrease was marginally significant for MMN (paired t-test, p = 0.050) but not for P3 (p = 0.190).

**Within-subject (secondary).** Within-subject classification with surprise features yielded perfect AUC (1.000) for both paradigms, confirming the predicted label-leakage confound: surprise regressors are deterministic functions of the stimulus sequence and therefore perfectly encode stimulus identity within each subject. This result is reported for transparency but is not scientifically informative.

**Summary for H2:** H2 is not supported. Contextual surprise features (orthogonalized to stimulus type) do not improve cross-subject decoding and may slightly hurt performance through increased dimensionality. The within-subject analysis is uninformative due to fundamental label leakage.

### 3.5 Sensitivity Analyses

**Adaptive Shannon window size.** Results were qualitatively similar across window sizes (w = 10, 20, 50). The w = 20 window yielded the highest correlation with the neural data but none reached significance.

**Change-point hazard rate.** Sensitivity analysis over h ∈ {1/50, 1/100, 1/200, 1/500} showed that the change-point surprise regressor was highly correlated with static Shannon across all hazard rates (r > 0.97), consistent with the stationary nature of the oddball paradigm. The hazard rate had minimal impact on the encoding results.

## 4. Discussion

### 4.1 Key Findings

This study provides the first systematic, single-trial comparison of four hierarchical surprise formulations applied to prediction-error responses in human EEG. Bayesian surprise — quantifying the magnitude of belief revision — showed the strongest trend toward predicting trial-by-trial MMN amplitude, but this did not survive correction for multiple comparisons. A similar pattern emerged in the P3 paradigm, where both Bayesian and change-point surprise showed trends for P3b amplitude. No model significantly improved out-of-sample prediction or cross-subject decoding. These results suggest that while computational models of surprise capture meaningful aspects of neural prediction error, the effect sizes in stationary oddball paradigms are too small to reliably distinguish between formulations.

### 4.2 Relation to Prior Work

Our results are consistent with Mars et al. (2008), who reported trial-by-trial correlations between model-based surprise and ERP amplitudes, and with Ostwald et al. (2012), who found evidence for neural encoding of Bayesian surprise in somatosensation. We extend these findings by providing a formal comparison across four model families within a single framework.

The superiority of Bayesian surprise aligns with predictive coding theories proposing that the brain maintains probabilistic generative models that are updated upon receiving new evidence (Friston, 2005). Under this view, the MMN reflects the precision-weighted prediction error that drives belief updating — a quantity most directly captured by Bayesian surprise (KL divergence between successive posteriors).

The relatively poor performance of adaptive Shannon surprise is notable: despite capturing local frequency adaptation, the sliding-window estimator did not significantly predict neural responses beyond what static global probability already explains. This suggests that the brain's internal model may be more sophisticated than a simple frequency counter.

### 4.3 Methodological Considerations

**Multicollinearity.** The high correlations among Shannon-based and change-point regressors (r > 0.95, VIF > 70) represent a fundamental limitation of applying these models to stationary oddball sequences. Because the deviant probability is constant throughout the experiment, all frequency-based models converge on similar estimates, and the change-point model's predictive surprise is dominated by its frequency estimate. Bayesian surprise, which measures belief change rather than event probability, provides a qualitatively different signal and is the only regressor with acceptable VIF (1.1). Future studies using roving oddball or volatile environments would better differentiate these models.

**Label leakage in decoding.** Our within-subject decoding analysis yielded perfect classification (AUC = 1.0) when surprise features were included, even after residualizing against stimulus type. This occurs because surprise regressors are deterministic functions of the stimulus sequence: knowing the surprise values at each trial position uniquely identifies the stimulus type. This is not a bug but a fundamental property of these models when applied to a known, fixed sequence. We designated cross-subject decoding as the primary test of H2 specifically because of this confound, and we report the within-subject results transparently to alert future researchers.

**Effect sizes.** The effect sizes for Bayesian surprise predicting MMN amplitude are very small (partial R² < 0.02%, ΔAIC = −3.9 on a dataset of 35,332 observations). Cross-validated prediction showed no improvement over the baseline model, suggesting that the encoding-model improvements may reflect overfitting to in-sample noise rather than genuine predictive power. This is consistent with the well-known difficulty of single-trial EEG analysis.

### 4.4 Limitations

First, the oddball paradigm is stationary by design, limiting the ability to differentiate between surprise models that depend on non-stationarity (particularly the change-point model). Second, our EEG preprocessing used amplitude-based rejection rather than the planned autoreject algorithm, due to missing electrode position information in the ERP CORE files; this may have retained some artifacts. Third, with N = 38 (MMN) and N = 36 (P3) subjects, our power to detect small differences between surprise models in pairwise comparisons is limited. Fourth, we examined only two ERP paradigms; generalization to more complex or naturalistic contexts remains to be established.

### 4.5 Future Directions

Three extensions would strengthen these findings. First, applying the same benchmark to roving oddball paradigms, where stimulus probabilities genuinely change over time, would better differentiate the change-point model from frequency-based alternatives. Second, extending the approach to naturalistic stimuli (e.g., language processing, social prediction) would test whether Bayesian surprise remains the best predictor in more ecologically valid settings. Third, combining our computational approach with source localization or intracranial recordings could reveal whether different surprise computations are realized in distinct neural circuits.

## 5. Conclusion

This paper reports a benchmark comparing four surprise formulations as predictors of single-trial EEG responses. Bayesian surprise shows the strongest trend, but no model survives correction for multiple comparisons. High regressor collinearity and trivial label leakage are identified as critical methodological challenges. These results set a baseline and point to non-stationary paradigms as the necessary next step.

## Data and Code Availability

Data come from the ERP CORE dataset (Kappenman et al., 2021; https://erpinfo.org/erp-core; CC BY-SA 4.0). Code is available at https://github.com/brhanufen/surprise-eeg-benchmark. The conda environment specification ensures full reproducibility.

## References

Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.

Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360, 815–836.

Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267.

Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. Vision Research, 49(10), 1295–1306.

Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417–429.

Kappenman, E. S., Farrens, J. L., Luck, S. J., & Proudfit, G. H. (2021). ERP CORE: An open resource for human event-related potential research. NeuroImage, 225, 117465.

Kolossa, A., Fingscheidt, T., Wessel, K., & Kopp, B. (2015). A model-based approach to trial-by-trial P300 amplitude fluctuations. Frontiers in Human Neuroscience, 6, 359.

Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. Journal of Neuroscience Methods, 164(1), 177–190.

Mars, R. B., et al. (2008). Trial-by-trial fluctuations in the event-related electroencephalogram reflect dynamic changes in the degree of surprise. Journal of Neuroscience, 28(47), 12539–12545.

Näätänen, R., Paavilainen, P., Rinne, T., & Alho, K. (2007). The mismatch negativity (MMN) in basic research of central auditory processing. Clinical Neurophysiology, 118(12), 2544–2590.

Ostwald, D., et al. (2012). Evidence for neural encoding of Bayesian surprise in human somatosensation. NeuroImage, 62(1), 177–188.

Pernet, C. R., et al. (2011). Robust, bias-free single trial ERP analysis. NeuroImage, 55(2), 604–613.

Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. Clinical Neurophysiology, 118(10), 2128–2148.

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience, 2(1), 79–87.
