"""
clt_analysis.py — Statistical inference on the mean fairness ratio.

Phase 4 of the CS109 project. The Monte Carlo simulator produces n independent
realizations of the fairness ratio ρ. We use the CLT to attach uncertainty
quantification to E[ρ] and to formally test whether the routed amount beats
the commercial benchmark.

CS109 CONTENT
-------------

(1) Central Limit Theorem

    For i.i.d. ρ_1, ..., ρ_n with finite variance,
        √n · (ρ̄ − E[ρ])  →  N(0, Var[ρ])  in distribution.
    Equivalently, for large n,
        ρ̄ ≈ N( E[ρ], Var[ρ] / n ).

(2) Two-sided confidence interval (CLT-based)

    A 100·(1 − α)% confidence interval for E[ρ] is
        ρ̄ ± z_{1 − α/2} · σ̂ / √n.
    For α = 0.05, z = Φ⁻¹(0.975) ≈ 1.959963985.

(3) One-sample z-test (Wald form)

    H₀: E[ρ] = 1   vs   H₁: E[ρ] > 1   (does the routed path beat commercial?)

    Under H₀, by the CLT,
        Z = (ρ̄ − 1) / (σ̂ / √n)  →  N(0, 1).
    The (one-sided) p-value is P(Z > z_obs) = 1 − Φ(z_obs).
    Reject H₀ at level α iff p < α.

Marco Paes — CS109 Probability Challenge, June 2026
"""

from math import erf, sqrt
import numpy as np
from typing import Tuple, List, Optional


# Φ⁻¹(0.975) — 0.975 quantile of the standard normal, hard-coded so the
# module has no scipy dependency.
Z_975 = 1.959963984540054


# ----------------------------------------------------------------------------
# Standard normal CDF (closed-form via erf, no scipy)
# ----------------------------------------------------------------------------

def norm_cdf(z: float) -> float:
    """Standard normal CDF, Φ(z) = (1/2)·(1 + erf(z/√2))."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# ----------------------------------------------------------------------------
# Confidence interval for the mean
# ----------------------------------------------------------------------------

def mean_ci_normal(values: np.ndarray, alpha: float = 0.05
                  ) -> Tuple[float, float, float, float]:
    """CLT-based confidence interval for the population mean.

    Returns (sample_mean, lower, upper, standard_error).

    Uses the unbiased sample variance (ddof = 1) and the normal quantile;
    valid for large n by the CLT. For small n with unknown variance the
    t-distribution would be more conservative, but our typical n (300+)
    makes the normal approximation indistinguishable.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        nan = float("nan")
        return nan, nan, nan, nan
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    if abs(alpha - 0.05) > 1e-10:
        raise NotImplementedError("Only alpha=0.05 supported (Z_975 hard-coded)")
    return mean, mean - Z_975 * se, mean + Z_975 * se, se


# ----------------------------------------------------------------------------
# One-sample z-test
# ----------------------------------------------------------------------------

def one_sample_z_test(values: np.ndarray, null_value: float = 1.0,
                      alternative: str = "greater"
                      ) -> Tuple[float, float, float, float, float]:
    """One-sample z-test for the population mean.

    H₀: μ = null_value
    H₁ depends on `alternative` in {'greater', 'less', 'two-sided'}.

    Returns (sample_mean, sample_std, standard_error, z_statistic, p_value).
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    if se == 0.0:
        return mean, std, se, float("inf"), 0.0

    z = (mean - null_value) / se
    if alternative == "greater":
        p = 1.0 - norm_cdf(z)
    elif alternative == "less":
        p = norm_cdf(z)
    elif alternative == "two-sided":
        p = 2.0 * (1.0 - norm_cdf(abs(z)))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return mean, std, se, z, float(p)


# ----------------------------------------------------------------------------
# Printout helper to integrate into the simulator
# ----------------------------------------------------------------------------

def print_inference_block(
    results,
    dest_currency: str,
    source_bank: str = "",
    source_currency: str = "",
    dest_bank: str = "",
) -> None:
    """Pretty-print a CLT-based inference block from a list of GreedyResults.

    Pass the same `results` list produced by run_greedy_experiments().
    """
    successful = [r for r in results if r.success]
    n = len(successful)
    if n < 30:
        print(f"  Only {n} successful runs — CLT may not apply.")
        return

    ratios = np.array([r.ratio_vs_commercial for r in successful])
    amounts = np.array([r.final_amount for r in successful])

    mean_r, lo_r, hi_r, se_r = mean_ci_normal(ratios)
    mean_a, lo_a, hi_a, se_a = mean_ci_normal(amounts)

    # Does the route beat commercial?
    _, _, _, z_stat, p_val = one_sample_z_test(ratios, null_value=1.0,
                                               alternative="greater")

    print()
    print("=" * 60)
    print("PHASE 4 — CLT-BASED STATISTICAL INFERENCE")
    print("=" * 60)
    print(f"Sample size (successful runs):       n = {n}")
    print()
    print(f"Mean fairness ratio  E[ρ]   = {mean_r:.4f}")
    print(f"  95% CI (CLT)             = ({lo_r:.4f},  {hi_r:.4f})")
    print(f"  standard error           = {se_r:.5f}")
    print()
    print(f"Mean final amount  E[A_T]   = {mean_a:,.2f} {dest_currency}")
    print(f"  95% CI (CLT)             = ({lo_a:,.2f},  {hi_a:,.2f}) {dest_currency}")
    print()
    print("Hypothesis test: does the routed path beat commercial?")
    print("  H₀: E[ρ] = 1     H₁: E[ρ] > 1")
    print(f"  Z-statistic              = {z_stat:.3f}")
    if p_val < 1e-10:
        p_str = f"< 1e-10"
    elif p_val < 1e-4:
        p_str = f"{p_val:.2e}"
    else:
        p_str = f"{p_val:.4f}"
    print(f"  p-value (one-sided)      = {p_str}")
    if p_val < 0.001:
        verdict = "REJECT H₀ at α = 0.001 — overwhelming evidence the route beats commercial"
    elif p_val < 0.01:
        verdict = "REJECT H₀ at α = 0.01 — strong evidence"
    elif p_val < 0.05:
        verdict = "REJECT H₀ at α = 0.05 — moderate evidence"
    else:
        verdict = "FAIL TO REJECT H₀ — insufficient evidence at α = 0.05"
    print(f"  Verdict:                   {verdict}")
    print("=" * 60)
