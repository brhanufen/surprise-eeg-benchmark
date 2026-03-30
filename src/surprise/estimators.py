#!/usr/bin/env python3
"""Four hierarchical surprise estimators for binary stimulus sequences.

Estimators:
  1. Static Shannon surprise: -log p_global(x_t)
  2. Adaptive Shannon surprise: -log p_w(x_t) with sliding window
  3. Bayesian surprise: KL divergence between successive Beta posteriors
  4. Change-point surprise: Adams & MacKay (2007) posterior change probability

All operate on a binary stimulus sequence (0=standard, 1=deviant).
"""

import numpy as np
from scipy.special import betaln


def static_shannon_surprise(sequence):
    """Static Shannon surprise: S_t = -log2 p_global(x_t).

    Uses the global frequency of each stimulus type across the entire sequence.
    """
    seq = np.asarray(sequence, dtype=float)
    n = len(seq)
    p_deviant = np.sum(seq) / n
    p_standard = 1.0 - p_deviant

    # Avoid log(0)
    p_deviant = np.clip(p_deviant, 1e-10, 1 - 1e-10)
    p_standard = np.clip(p_standard, 1e-10, 1 - 1e-10)

    surprise = np.where(seq == 1, -np.log2(p_deviant), -np.log2(p_standard))

    return surprise


def adaptive_shannon_surprise(sequence, window_size=20):
    """Adaptive Shannon surprise: S_t = -log2 p_w(x_t).

    Uses a sliding window of the last `window_size` trials to estimate p(x_t).
    For the first `window_size` trials, uses all available history.
    """
    seq = np.asarray(sequence, dtype=float)
    n = len(seq)
    surprise = np.zeros(n)

    for t in range(n):
        if t == 0:
            # No history: use uniform prior (p=0.5)
            p_deviant = 0.5
        else:
            start = max(0, t - window_size)
            window = seq[start:t]
            # Add Laplace smoothing to avoid log(0)
            p_deviant = (np.sum(window) + 1) / (len(window) + 2)

        p_standard = 1.0 - p_deviant

        if seq[t] == 1:
            surprise[t] = -np.log2(p_deviant)
        else:
            surprise[t] = -np.log2(p_standard)

    return surprise


def bayesian_surprise(sequence):
    """Bayesian surprise: KL divergence between successive Beta posteriors.

    Uses a Beta-Bernoulli conjugate model with flat prior Beta(1,1).
    At trial t, the posterior is Beta(alpha_t, beta_t) where:
      alpha_t = 1 + sum of deviants up to t
      beta_t = 1 + sum of standards up to t

    Bayesian surprise = KL[Beta(alpha_t, beta_t) || Beta(alpha_{t-1}, beta_{t-1})]
    """
    seq = np.asarray(sequence, dtype=float)
    n = len(seq)
    surprise = np.zeros(n)
    posterior_entropy = np.zeros(n)

    # Prior: Beta(1, 1) = uniform
    alpha_prev = 1.0
    beta_prev = 1.0

    for t in range(n):
        # Update posterior
        if seq[t] == 1:
            alpha_curr = alpha_prev + 1
            beta_curr = beta_prev
        else:
            alpha_curr = alpha_prev
            beta_curr = beta_prev + 1

        # KL divergence: KL[Beta(a2,b2) || Beta(a1,b1)]
        # = ln B(a1,b1) - ln B(a2,b2) + (a2-a1)*psi(a2) + (b2-b1)*psi(b2)
        #   - (a2-a1+b2-b1)*psi(a2+b2)
        # where B is the beta function and psi is the digamma function
        from scipy.special import digamma

        kl = (betaln(alpha_prev, beta_prev) - betaln(alpha_curr, beta_curr)
              + (alpha_curr - alpha_prev) * digamma(alpha_curr)
              + (beta_curr - beta_prev) * digamma(beta_curr)
              - (alpha_curr - alpha_prev + beta_curr - beta_prev)
                * digamma(alpha_curr + beta_curr))

        surprise[t] = max(0, kl)  # KL should be >= 0; numerical floor

        # Posterior entropy of Beta distribution
        ent = (betaln(alpha_curr, beta_curr)
               - (alpha_curr - 1) * digamma(alpha_curr)
               - (beta_curr - 1) * digamma(beta_curr)
               + (alpha_curr + beta_curr - 2) * digamma(alpha_curr + beta_curr))
        posterior_entropy[t] = ent

        alpha_prev = alpha_curr
        beta_prev = beta_curr

    return surprise, posterior_entropy


def changepoint_surprise(sequence, hazard_rate=1/200):
    """Change-point surprise using Adams & MacKay (2007) framework.

    Computes the posterior probability of a regime change at each trial.
    Uses a geometric hazard function: P(changepoint) = hazard_rate.

    The observation model is Bernoulli with Beta conjugate prior.

    Parameters
    ----------
    sequence : array-like
        Binary stimulus sequence (0=standard, 1=deviant).
    hazard_rate : float
        Prior probability of a change-point on any trial.

    Returns
    -------
    predictive_surprise : array
        -log2 P(x_t | model), the predictive surprise under the full
        change-point model (marginalizing over run lengths). This is the
        primary regressor — it varies trial-by-trial even in stationary
        sequences because the model's predictions depend on local history.
    change_prob : array
        Posterior probability of a change-point at each trial.
    run_length_mean : array
        Posterior mean run length at each trial.
    estimated_volatility : array
        Entropy of the run length distribution (proxy for volatility).
    """
    seq = np.asarray(sequence, dtype=float)
    n = len(seq)

    # Truncated run length to keep computation tractable
    # After ~300 trials the run length posterior is negligible
    max_rl = min(n + 1, 500)

    # Run length probabilities: P(r_t = r | x_{1:t})
    rl_dist = np.zeros(max_rl)
    rl_dist[0] = 1.0

    # Sufficient statistics for each run length hypothesis
    alphas = np.ones(max_rl)
    betas = np.ones(max_rl)

    predictive_surprise = np.zeros(n)
    change_prob = np.zeros(n)
    run_length_mean = np.zeros(n)
    estimated_volatility = np.zeros(n)

    run_lengths_arr = np.arange(max_rl, dtype=float)

    for t in range(n):
        x = seq[t]
        active = min(t + 1, max_rl)

        # 1. Vectorized predictive probability
        total = alphas[:active] + betas[:active]
        if x == 1:
            pred_probs_active = alphas[:active] / total
        else:
            pred_probs_active = betas[:active] / total

        # Weight by run length distribution
        weighted = rl_dist[:active] * pred_probs_active

        # Marginal predictive probability
        marginal_pred = np.sum(weighted)
        marginal_pred = max(marginal_pred, 1e-300)
        predictive_surprise[t] = -np.log2(marginal_pred)

        # 2. Growth probabilities (vectorized)
        growth = weighted * (1 - hazard_rate)

        # 3. Change-point mass
        cp_mass = np.sum(weighted * hazard_rate)

        # 4. Update run length distribution
        new_active = min(active + 1, max_rl)
        new_rl_dist = np.zeros(max_rl)
        new_rl_dist[0] = cp_mass
        shift_len = min(active, max_rl - 1)
        new_rl_dist[1:shift_len+1] = growth[:shift_len]

        # 5. Normalize
        evidence = np.sum(new_rl_dist[:new_active])
        if evidence > 0:
            new_rl_dist[:new_active] /= evidence

        # 6. Record outputs
        change_prob[t] = new_rl_dist[0]
        run_length_mean[t] = np.dot(run_lengths_arr[:new_active],
                                     new_rl_dist[:new_active])
        nonzero_mask = new_rl_dist[:new_active] > 1e-300
        if np.any(nonzero_mask):
            nz = new_rl_dist[:new_active][nonzero_mask]
            estimated_volatility[t] = -np.sum(nz * np.log2(nz))

        # 7. Update sufficient statistics (vectorized)
        new_alphas = np.ones(max_rl)
        new_betas = np.ones(max_rl)
        if x == 1:
            new_alphas[1:shift_len+1] = alphas[:shift_len] + 1
            new_betas[1:shift_len+1] = betas[:shift_len]
        else:
            new_alphas[1:shift_len+1] = alphas[:shift_len]
            new_betas[1:shift_len+1] = betas[:shift_len] + 1

        rl_dist = new_rl_dist
        alphas = new_alphas
        betas = new_betas

    return predictive_surprise, change_prob, run_length_mean, estimated_volatility


def compute_all_surprise(sequence, window_sizes=[10, 20, 50],
                         hazard_rates=[1/200]):
    """Compute all four surprise estimators for a stimulus sequence.

    Parameters
    ----------
    sequence : array-like
        Binary stimulus sequence (0=standard, 1=deviant).
    window_sizes : list of int
        Window sizes for adaptive Shannon surprise.
    hazard_rates : list of float
        Hazard rates for change-point model.

    Returns
    -------
    results : dict
        Dictionary with all surprise values and auxiliary metrics.
    """
    seq = np.asarray(sequence, dtype=float)

    results = {
        'trial': np.arange(len(seq)),
        'stimulus': seq.astype(int),
        'static_shannon': static_shannon_surprise(seq),
    }

    # Adaptive Shannon for each window size
    for w in window_sizes:
        results[f'adaptive_shannon_w{w}'] = adaptive_shannon_surprise(seq, w)

    # Bayesian surprise
    bayes_surp, post_entropy = bayesian_surprise(seq)
    results['bayesian_surprise'] = bayes_surp
    results['posterior_entropy'] = post_entropy

    # Change-point surprise (primary hazard rate)
    h = hazard_rates[0]
    cp_pred_surp, cp_prob, rl_mean, volatility = changepoint_surprise(seq, h)
    results['changepoint_surprise'] = cp_pred_surp  # Predictive surprise (primary)
    results['changepoint_prob'] = cp_prob  # Change-point posterior probability
    results['run_length_mean'] = rl_mean
    results['estimated_volatility'] = volatility

    # Additional hazard rates for sensitivity analysis
    for h in hazard_rates[1:]:
        h_label = f"{h:.4f}".replace(".", "p")
        cp_s, _, _, _ = changepoint_surprise(seq, h)
        results[f'changepoint_h{h_label}'] = cp_s

    return results


if __name__ == "__main__":
    # Quick test with a simple sequence
    np.random.seed(42)
    p_dev = 0.15
    seq = (np.random.rand(200) < p_dev).astype(int)

    print(f"Sequence: {sum(seq)} deviants out of {len(seq)} trials ({100*sum(seq)/len(seq):.1f}%)")

    results = compute_all_surprise(seq, window_sizes=[10, 20, 50],
                                    hazard_rates=[1/200, 1/50, 1/100, 1/500])

    print(f"\nStatic Shannon - mean: {results['static_shannon'].mean():.3f}, "
          f"std: {results['static_shannon'].std():.3f}")
    print(f"Adaptive Shannon w20 - mean: {results['adaptive_shannon_w20'].mean():.3f}, "
          f"std: {results['adaptive_shannon_w20'].std():.3f}")
    print(f"Bayesian surprise - mean: {results['bayesian_surprise'].mean():.4f}, "
          f"std: {results['bayesian_surprise'].std():.4f}")
    print(f"Change-point pred. surprise - mean: {results['changepoint_surprise'].mean():.4f}, "
          f"std: {results['changepoint_surprise'].std():.4f}")
    print(f"Change-point probability - mean: {results['changepoint_prob'].mean():.4f}, "
          f"std: {results['changepoint_prob'].std():.6f}")

    # Check variance of change-point regressor (Week 0 viability check)
    cp_std = results['changepoint_surprise'].std()
    print(f"\nChange-point predictive surprise SD: {cp_std:.6f}")
    print(f"Viability check (SD > 0.01): {'PASS' if cp_std > 0.01 else 'FAIL'}")

    # Correlation matrix
    import pandas as pd
    cols = ['static_shannon', 'adaptive_shannon_w20', 'bayesian_surprise', 'changepoint_surprise']
    df = pd.DataFrame({c: results[c] for c in cols})
    print("\nCorrelation matrix:")
    print(df.corr().round(3))
