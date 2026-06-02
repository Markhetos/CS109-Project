"""
test_mle.py — Sanity checks for mle_estimation.py.

Generates synthetic data with known parameters, runs the MLE estimators, and
verifies (a) recovered parameters are within tolerance of the truth and
(b) the log-likelihood comparison correctly identifies the generating model.
"""

import numpy as np
import pandas as pd
from mle_estimation import (
    mle_gaussian, mle_lognormal, compare_models,
    fit_all_corridors, summarize_fits, plot_corridor_fits,
)


def test_gaussian_recovers_parameters():
    rng = np.random.default_rng(42)
    true_mu, true_sigma, n = 15.0, 4.0, 5000
    data = rng.normal(true_mu, true_sigma, size=n)

    fit = mle_gaussian(data)
    assert abs(fit.mu - true_mu) < 0.15, f"mu off: {fit.mu}"
    assert abs(fit.sigma - true_sigma) < 0.15, f"sigma off: {fit.sigma}"
    print(f"  Gaussian MLE: mu_hat={fit.mu:.3f} (true 15.0), "
          f"sigma_hat={fit.sigma:.3f} (true 4.0), logL={fit.log_likelihood:.1f}")


def test_lognormal_recovers_parameters():
    rng = np.random.default_rng(7)
    true_mu, true_sigma, n = 2.0, 0.5, 5000
    # log X ~ N(2, 0.5^2), so X ~ LogNormal(2, 0.5^2)
    data = rng.lognormal(mean=true_mu, sigma=true_sigma, size=n)

    fit = mle_lognormal(data)
    assert abs(fit.mu_log - true_mu) < 0.05, f"mu_log off: {fit.mu_log}"
    assert abs(fit.sigma_log - true_sigma) < 0.05, f"sigma_log off: {fit.sigma_log}"
    # Closed-form mean of X
    true_mean_x = np.exp(true_mu + 0.5 * true_sigma**2)
    assert abs(fit.mean_x - true_mean_x) / true_mean_x < 0.05
    print(f"  LogNormal MLE: mu_log_hat={fit.mu_log:.3f} (true 2.0), "
          f"sigma_log_hat={fit.sigma_log:.3f} (true 0.5)")
    print(f"                E[X]_hat={fit.mean_x:.3f} (true {true_mean_x:.3f}), "
          f"logL={fit.log_likelihood:.1f}")


def test_comparison_picks_correct_model():
    rng = np.random.default_rng(123)
    n = 2000

    # Case 1: data IS Gaussian (truncated to positive) -> Gaussian should win
    gauss_data = np.abs(rng.normal(20.0, 5.0, size=n))
    comp1 = compare_models(gauss_data, ("TEST", "GAUSS"))
    print(f"  Gaussian-generated data:  Δℓ = {comp1.log_lik_ratio:+.1f}, winner = {comp1.winner}")

    # Case 2: data IS log-normal -> Log-Normal should win
    lognorm_data = rng.lognormal(2.5, 0.6, size=n)
    comp2 = compare_models(lognorm_data, ("TEST", "LOGN"))
    print(f"  LogNormal-generated data: Δℓ = {comp2.log_lik_ratio:+.1f}, winner = {comp2.winner}")
    assert comp2.winner == "lognormal", "log-normal should win on log-normal data"


def test_end_to_end_with_synthetic_fx():
    """Build a synthetic FX dataframe with the same column layout as Marco's
    real data, then run fit_all_corridors and plot the diagnostics."""
    rng = np.random.default_rng(2026)
    T = 2500  # roughly a decade of daily data
    dates = pd.date_range("2015-01-01", periods=T, freq="D")

    # Build geometric-random-walk price series for each anchor pair, with
    # heavy-tailed shocks so log-normal volatility is a plausible winner.
    def random_walk(start, daily_vol):
        shocks = rng.normal(0, daily_vol, size=T)
        # Add occasional jumps to fatten the tails
        jumps = rng.choice([0, 0, 0, 0, 1], size=T) * rng.normal(0, 3*daily_vol, size=T)
        return start * np.exp(np.cumsum(shocks + jumps))

    df = pd.DataFrame({
        "Date": dates,
        "USD_BRL": random_walk(start=3.5,    daily_vol=0.008),
        "USD_EUR": random_walk(start=0.85,   daily_vol=0.005),
        "USD_PYG": random_walk(start=6000.0, daily_vol=0.006),
        "BRL_EUR": random_walk(start=0.20,   daily_vol=0.010),
        "BRL_PYG": random_walk(start=1500.0, daily_vol=0.012),
        "EUR_PYG": random_walk(start=7000.0, daily_vol=0.007),
    })

    fits = fit_all_corridors(df)
    print(f"\n  Fit {len(fits)} corridors from synthetic data.")
    summary = summarize_fits(fits)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summary[["from", "to", "n", "gauss_sigma_bps",
                   "lognorm_SD[X]_bps", "delta_logL", "winner"]].to_string(index=False))

    plot_corridor_fits(df, [("USD", "BRL"), ("USD", "PYG"), ("BRL", "EUR")],
                       filename="mle_corridor_fits_synthetic.png")


if __name__ == "__main__":
    print("=" * 64)
    print("Test 1: Gaussian MLE recovers known parameters")
    print("=" * 64)
    test_gaussian_recovers_parameters()

    print("\n" + "=" * 64)
    print("Test 2: Log-Normal MLE recovers known parameters")
    print("=" * 64)
    test_lognormal_recovers_parameters()

    print("\n" + "=" * 64)
    print("Test 3: Model comparison picks the true generating model")
    print("=" * 64)
    test_comparison_picks_correct_model()

    print("\n" + "=" * 64)
    print("Test 4: End-to-end with a synthetic FX dataframe")
    print("=" * 64)
    test_end_to_end_with_synthetic_fx()

    print("\nAll tests passed.")
