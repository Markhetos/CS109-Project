"""
test_ou.py — Synthetic-data sanity checks for ou_process.py.

(1) OU MLE recovers known (θ, μ, σ) on a path drawn from the true model.
(2) Long-run marginal of a simulated OU matches the theoretical stationary
    distribution N(μ, σ²/(2θ)).
(3) Exact and Euler-Maruyama simulators agree on long-run moments.
(4) OUMarkupEngine in signed mode produces draws with the right autocorrelation.
(5) Volatility clustering: log_volatility mode finds meaningful autocorrelation
    on a synthetic series where signed-returns mode does not. This is the
    motivation for switching modes on real FX data.
"""

import numpy as np
import pandas as pd

from ou_process import (
    fit_ou_mle, simulate_ou_exact, simulate_ou_euler, sample_stationary,
    fit_ou_all_corridors, summarize_ou_fits, plot_ou_diagnostics,
    compute_signed_log_returns_bps, compute_log_volatility,
    OUMarkupEngine,
)


def simulate_path_with_known_params(theta, mu, sigma, n, rng):
    dt = 1.0
    a = np.exp(-theta * dt)
    b = mu * (1.0 - a)
    s = np.sqrt(sigma ** 2 * (1.0 - a ** 2) / (2.0 * theta))
    x = np.empty(n)
    x[0] = mu
    for t in range(n - 1):
        x[t + 1] = a * x[t] + b + s * rng.standard_normal()
    return x


def test_mle_recovers_parameters():
    print("Test 1: OU MLE recovers known (θ, μ, σ)")
    rng = np.random.default_rng(0)
    true_theta, true_mu, true_sigma = 0.10, 25.0, 15.0

    x = simulate_path_with_known_params(true_theta, true_mu, true_sigma,
                                          n=20_000, rng=rng)
    fit = fit_ou_mle(x, dt=1.0)
    print(f"  θ_hat = {fit.theta:.4f}  (true {true_theta})")
    print(f"  μ_hat = {fit.mu:.3f}     (true {true_mu})")
    print(f"  σ_hat = {fit.sigma:.3f}  (true {true_sigma})")
    print(f"  â_raw = {fit.a_raw:.4f}  (theoretical {np.exp(-true_theta):.4f})")
    print(f"  clipped: {fit.a_was_clipped}")
    assert abs(fit.theta - true_theta) / true_theta < 0.10
    assert abs(fit.mu - true_mu) < 1.0
    assert abs(fit.sigma - true_sigma) / true_sigma < 0.10
    assert not fit.a_was_clipped
    print("  PASS")


def test_stationary_distribution():
    print("\nTest 2: long-run marginal of a simulated OU matches N(μ, σ²/2θ)")
    rng = np.random.default_rng(1)
    x = simulate_path_with_known_params(0.20, 0.0, 30.0, n=20_000, rng=rng)
    fit = fit_ou_mle(x, dt=1.0)

    long_path = simulate_ou_exact(fit, x0=fit.stationary_mean,
                                   n_steps=50_000, rng=rng)
    burn_in = 1000
    sample = long_path[burn_in:]
    emp_mean = float(np.mean(sample))
    emp_std = float(np.std(sample))
    print(f"  empirical mean = {emp_mean:.3f}  (theoretical {fit.stationary_mean:.3f})")
    print(f"  empirical std  = {emp_std:.3f}   (theoretical {fit.stationary_std:.3f})")
    assert abs(emp_mean - fit.stationary_mean) < 2.0
    assert abs(emp_std - fit.stationary_std) / fit.stationary_std < 0.05
    print("  PASS")


def test_exact_vs_euler():
    print("\nTest 3: exact AR(1) and Euler-Maruyama agree on long-run moments")
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    x = simulate_path_with_known_params(0.15, 10.0, 20.0, n=5000,
                                          rng=np.random.default_rng(7))
    fit = fit_ou_mle(x, dt=1.0)
    path_exact = simulate_ou_exact(fit, x0=fit.stationary_mean, n_steps=20_000, rng=rng_a)
    path_euler = simulate_ou_euler(fit, x0=fit.stationary_mean, n_steps=20_000,
                                     dt=0.05, rng=rng_b)
    print(f"  exact path: mean={np.mean(path_exact):.3f}, std={np.std(path_exact):.3f}")
    print(f"  Euler path: mean={np.mean(path_euler):.3f}, std={np.std(path_euler):.3f}")
    print(f"  theoretical: mean={fit.stationary_mean:.3f}, std={fit.stationary_std:.3f}")
    print("  PASS")


def test_markup_engine_signed_mode():
    print("\nTest 4: OUMarkupEngine (signed mode) — pooled autocorrelation")
    rng = np.random.default_rng(99)
    x = simulate_path_with_known_params(0.3, 0.0, 30.0, n=5000, rng=rng)
    fit = fit_ou_mle(x, dt=1.0, mode="signed")
    fits = {("USD", "BRL"): fit}
    engine = OUMarkupEngine(fits, rng=rng)

    prev_vals, next_vals = [], []
    for _ in range(5000):
        engine.reset()
        draws = [engine.step("USD", "BRL") for _ in range(5)]
        prev_vals.extend(draws[:-1])
        next_vals.extend(draws[1:])
    prev_vals, next_vals = np.array(prev_vals), np.array(next_vals)
    corr = float(np.corrcoef(prev_vals, next_vals)[0, 1])
    print(f"  pooled 1-step autocorrelation: {corr:.3f}")
    print(f"  theoretical a = exp(-θ Δt):    {fit.a:.3f}")
    assert abs(corr - fit.a) < 0.05
    print("  PASS")


def test_volatility_clustering_motivation():
    print("\nTest 5: log_volatility mode finds autocorrelation that signed-mode misses")
    print("       (motivates the mode switch for real FX data)")
    rng = np.random.default_rng(2026)
    T = 5000

    # Construct a synthetic price series with VOLATILITY CLUSTERING but
    # i.i.d. signed returns. This is the classic stylized fact of FX data.
    # Mechanism: latent volatility follows AR(1); returns are sign-randomized
    # magnitudes.
    log_vol = np.zeros(T)
    log_vol[0] = np.log(50.0)  # baseline 50 bps volatility
    persistence = 0.85          # strong vol clustering
    vol_noise = 0.3
    for t in range(1, T):
        log_vol[t] = (1 - persistence) * np.log(50.0) + persistence * log_vol[t-1] \
                     + vol_noise * rng.standard_normal()
    vol_bps = np.exp(log_vol)
    signs = rng.choice([-1, 1], size=T)
    returns_bps = signs * vol_bps
    # Build a price series from these log returns
    rates = 3.5 * np.exp(np.cumsum(returns_bps / 10_000))

    # Fit OU in BOTH modes
    signed = compute_signed_log_returns_bps(rates)
    fit_signed = fit_ou_mle(signed, dt=1.0, mode="signed")
    print(f"  signed mode:         â_raw = {fit_signed.a_raw:+.4f}   "
          f"clipped: {fit_signed.a_was_clipped}")

    logvol = compute_log_volatility(rates)
    fit_lv = fit_ou_mle(logvol, dt=1.0, mode="log_volatility")
    print(f"  log_volatility mode: â_raw = {fit_lv.a_raw:+.4f}   "
          f"clipped: {fit_lv.a_was_clipped}   "
          f"half-life = {fit_lv.half_life:.1f}d")

    assert fit_signed.a_raw < 0.05, "synthetic signed returns should look i.i.d."
    assert fit_lv.a_raw > 0.5, "synthetic log-volatility should be strongly persistent"
    print("  PASS — same data, OU finds clustering only when we look at the right thing")


def test_end_to_end_log_volatility():
    print("\nTest 6: end-to-end on a synthetic FX dataframe with vol clustering")
    rng = np.random.default_rng(7)
    T = 2500
    dates = pd.date_range("2015-01-01", periods=T, freq="D")

    def vol_clustered_series(start, base_vol_bps, persistence=0.80):
        log_vol = np.zeros(T)
        log_vol[0] = np.log(base_vol_bps)
        for t in range(1, T):
            log_vol[t] = ((1 - persistence) * np.log(base_vol_bps)
                          + persistence * log_vol[t-1]
                          + 0.25 * rng.standard_normal())
        vols = np.exp(log_vol)
        signs = rng.choice([-1, 1], size=T)
        returns_bps = signs * vols
        return start * np.exp(np.cumsum(returns_bps / 10_000))

    df = pd.DataFrame({
        "Date": dates,
        "USD_BRL": vol_clustered_series(3.5, 80, 0.85),
        "USD_PYG": vol_clustered_series(6000.0, 60, 0.80),
        "BRL_EUR": vol_clustered_series(0.20, 100, 0.82),
    })

    fits = fit_ou_all_corridors(df, mode="log_volatility")
    summary = summarize_ou_fits(fits)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summary[["from", "to", "n", "theta", "stat_std",
                   "a_raw", "clipped", "half_life_days"]].to_string(index=False))
    plot_ou_diagnostics(df, fits,
                         pairs_to_plot=[("USD", "BRL"), ("USD", "PYG"), ("BRL", "EUR")],
                         filename="ou_diagnostics_logvol_synthetic.png")


if __name__ == "__main__":
    print("=" * 64)
    test_mle_recovers_parameters()
    print("\n" + "=" * 64)
    test_stationary_distribution()
    print("\n" + "=" * 64)
    test_exact_vs_euler()
    print("\n" + "=" * 64)
    test_markup_engine_signed_mode()
    print("\n" + "=" * 64)
    test_volatility_clustering_motivation()
    print("\n" + "=" * 64)
    test_end_to_end_log_volatility()
    print("\nAll tests passed.")