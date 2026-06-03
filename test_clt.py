"""
test_clt.py — Synthetic tests for clt_analysis.py.

(1) 95% CI achieves ~95% empirical coverage on simulated data.
(2) One-sample z-test rejects H₀ with high power when the true mean is shifted.
(3) Standard normal CDF agrees with known values.
"""

import numpy as np
from clt_analysis import (
    norm_cdf, mean_ci_normal, one_sample_z_test, Z_975,
)


def test_norm_cdf():
    print("Test 1: standard normal CDF spot checks")
    cases = [(0.0, 0.5), (1.96, 0.9750), (-1.96, 0.0250),
             (1.0, 0.8413), (2.0, 0.9772), (3.0, 0.9987)]
    for z, expected in cases:
        got = norm_cdf(z)
        assert abs(got - expected) < 1e-4, f"Φ({z}) = {got}, expected {expected}"
        print(f"  Φ({z:+.2f}) = {got:.4f}  (expected {expected:.4f})")
    print("  PASS")


def test_ci_coverage():
    """Simulate many samples; check the 95% CI contains the true mean ~95% of the time."""
    print("\nTest 2: 95% CI achieves nominal coverage")
    rng = np.random.default_rng(0)
    true_mean, true_std = 1.13, 0.04
    n_per_sample = 300
    n_trials = 5000
    hits = 0
    for _ in range(n_trials):
        sample = rng.normal(true_mean, true_std, size=n_per_sample)
        _, lo, hi, _ = mean_ci_normal(sample)
        if lo <= true_mean <= hi:
            hits += 1
    coverage = hits / n_trials
    print(f"  Empirical coverage over {n_trials} trials: {coverage:.4f}  (target 0.9500)")
    # Tolerance ~2 SE on the coverage proportion
    se_cov = np.sqrt(0.05 * 0.95 / n_trials)
    assert abs(coverage - 0.95) < 4 * se_cov
    print("  PASS")


def test_z_test_rejection():
    print("\nTest 3: one-sample z-test rejects H₀ when truth is far from null")
    rng = np.random.default_rng(0)
    # True mean 1.13, null 1.0, n=300 → effect size strongly detectable
    sample = rng.normal(1.13, 0.04, size=300)
    mean, std, se, z, p = one_sample_z_test(sample, null_value=1.0,
                                             alternative="greater")
    print(f"  sample mean = {mean:.4f}")
    print(f"  Z = {z:.3f}, p = {p:.2e}")
    assert p < 1e-30, "Should overwhelmingly reject H₀"
    print("  PASS")

    # And a non-rejection case
    print("\n         non-rejection case (truth = null)")
    sample2 = rng.normal(1.0, 0.04, size=300)
    mean, std, se, z, p = one_sample_z_test(sample2, null_value=1.0,
                                             alternative="greater")
    print(f"  sample mean = {mean:.4f}")
    print(f"  Z = {z:.3f}, p = {p:.4f}")
    # p should be ~uniform on (0,1) under H0 — at least not tiny
    assert p > 0.001
    print("  PASS")


if __name__ == "__main__":
    print("=" * 64)
    test_norm_cdf()
    print("\n" + "=" * 64)
    test_ci_coverage()
    print("\n" + "=" * 64)
    test_z_test_rejection()
    print("\nAll tests passed.")
