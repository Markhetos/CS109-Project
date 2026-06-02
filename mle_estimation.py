"""
mle_estimation.py — Maximum Likelihood Estimation of FX corridor volatility.

This module fits Gaussian and Log-Normal distributions to historical corridor
volatility (daily absolute log-returns in basis points). All MLEs are derived
in closed form (no scipy.stats.fit) so the work is genuine CS109 content.

MLE DERIVATIONS
---------------

(1) Gaussian   X ~ N(mu, sigma^2),  x in R

  Log-likelihood for n iid samples:
      ell(mu, sigma^2) = -n/2 * log(2*pi*sigma^2)
                        - (1/(2*sigma^2)) * sum_i (x_i - mu)^2

  Setting d ell / d mu = 0:          mu_hat     = (1/n) sum_i x_i
  Setting d ell / d sigma^2 = 0:     sigma2_hat = (1/n) sum_i (x_i - mu_hat)^2

  Plug-in maximum log-likelihood:
      ell_max = -n/2 * log(2*pi*sigma2_hat) - n/2


(2) Log-Normal   X ~ LogN(mu, sigma^2),  x > 0
    (equivalently, log X ~ N(mu, sigma^2))

  PDF:  f(x) = (1 / (x * sigma * sqrt(2*pi))) * exp( -(log x - mu)^2 / (2*sigma^2) )

  Log-likelihood:
      ell(mu, sigma^2) = -sum_i log(x_i)
                         - n/2 * log(2*pi*sigma^2)
                         - (1/(2*sigma^2)) * sum_i (log x_i - mu)^2

  The -sum log(x_i) term is the Jacobian from the change of variables
  Y = log X. Crucially, it keeps the log-likelihood as a log-density w.r.t.
  Lebesgue measure on (0, infty), so it is directly comparable to the
  Gaussian log-likelihood on the same positive data.

  Differentiate:                     mu_hat     = (1/n) sum_i log(x_i)
                                     sigma2_hat = (1/n) sum_i (log x_i - mu_hat)^2

  Plug-in maximum log-likelihood:
      ell_max = -sum_i log(x_i) - n/2 * log(2*pi*sigma2_hat) - n/2

MODEL COMPARISON
----------------
Both models have k = 2 parameters, so AIC = 2k - 2*ell_max gives the same
ordering as ell_max alone. Higher ell_max  <=>  lower AIC  <=>  better fit.
We pick the model with higher plug-in log-likelihood.

Marco Paes — CS109 Probability Challenge, June 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# Currency pairs we track (both directions handled separately so each gets
# its own fit; the inverse pair is a different time series after we invert).
CURRENCY_PAIRS: List[Tuple[str, str]] = [
    ("USD", "BRL"), ("BRL", "USD"),
    ("USD", "EUR"), ("EUR", "USD"),
    ("USD", "PYG"), ("PYG", "USD"),
    ("BRL", "EUR"), ("EUR", "BRL"),
    ("BRL", "PYG"), ("PYG", "BRL"),
    ("EUR", "PYG"), ("PYG", "EUR"),
]


# ----------------------------------------------------------------------------
# Data extraction
# ----------------------------------------------------------------------------

def get_rate_series(df: pd.DataFrame, c_from: str, c_to: str) -> Optional[np.ndarray]:
    """Time series of commercial rates c_from -> c_to.

    Tries the direct column first; falls back to the inverse if missing.
    Returns None if neither column exists.
    """
    direct = f"{c_from}_{c_to}"
    inverse = f"{c_to}_{c_from}"

    if direct in df.columns:
        rates = df[direct].to_numpy(dtype=float)
    elif inverse in df.columns:
        inv = df[inverse].to_numpy(dtype=float)
        rates = np.where(inv > 0, 1.0 / inv, np.nan)
    else:
        return None

    rates = rates[~np.isnan(rates)]
    rates = rates[rates > 0]
    return rates


def compute_corridor_volatility_bps(rates: np.ndarray) -> np.ndarray:
    """Daily absolute log-returns in basis points.

    Given a price series R_1, ..., R_T, returns
        v_t = |log(R_t / R_{t-1})| * 10_000   in basis points

    This is a positive RV that captures how much the corridor moved
    day-over-day — our empirical proxy for spread variability.
    """
    if len(rates) < 2:
        return np.array([])
    log_returns = np.diff(np.log(rates))
    return np.abs(log_returns) * 10_000.0


# ----------------------------------------------------------------------------
# Gaussian MLE
# ----------------------------------------------------------------------------

@dataclass
class GaussianFit:
    mu: float
    sigma: float
    n: int
    log_likelihood: float
    aic: float


def mle_gaussian(data: np.ndarray) -> GaussianFit:
    """Closed-form Gaussian MLE. See module docstring for derivation."""
    data = np.asarray(data, dtype=float)
    n = int(data.size)
    if n < 2:
        raise ValueError("Need at least 2 data points for Gaussian MLE.")

    mu_hat = float(np.mean(data))
    sigma2_hat = float(np.mean((data - mu_hat) ** 2))
    if sigma2_hat <= 0:
        raise ValueError("Degenerate data: zero variance.")
    sigma_hat = float(np.sqrt(sigma2_hat))

    log_lik = -0.5 * n * np.log(2.0 * np.pi * sigma2_hat) - 0.5 * n
    aic = 2 * 2 - 2 * log_lik

    return GaussianFit(mu=mu_hat, sigma=sigma_hat, n=n,
                       log_likelihood=float(log_lik), aic=float(aic))


# ----------------------------------------------------------------------------
# Log-Normal MLE
# ----------------------------------------------------------------------------

@dataclass
class LogNormalFit:
    mu_log: float       # MLE of mu_log = E[log X]
    sigma_log: float    # MLE of sigma_log = sd[log X]
    n: int
    log_likelihood: float
    aic: float
    # Implied moments of X itself (closed form for log-normal)
    mean_x: float       # E[X] = exp(mu + sigma^2 / 2)
    var_x: float        # Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)


def mle_lognormal(data: np.ndarray) -> LogNormalFit:
    """Closed-form Log-Normal MLE. See module docstring for derivation.

    The Jacobian -sum log(x_i) is included in the returned log-likelihood
    so the value is directly comparable to mle_gaussian(data).log_likelihood
    on the same positive data.
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]  # log-normal requires strictly positive support
    n = int(data.size)
    if n < 2:
        raise ValueError("Need at least 2 positive data points for Log-Normal MLE.")

    log_data = np.log(data)
    mu_log = float(np.mean(log_data))
    sigma2_log = float(np.mean((log_data - mu_log) ** 2))
    if sigma2_log <= 0:
        raise ValueError("Degenerate log-data: zero variance.")
    sigma_log = float(np.sqrt(sigma2_log))

    log_lik = (
        -float(np.sum(log_data))
        - 0.5 * n * np.log(2.0 * np.pi * sigma2_log)
        - 0.5 * n
    )
    aic = 2 * 2 - 2 * log_lik

    mean_x = float(np.exp(mu_log + 0.5 * sigma2_log))
    var_x = float((np.exp(sigma2_log) - 1.0) * np.exp(2.0 * mu_log + sigma2_log))

    return LogNormalFit(
        mu_log=mu_log, sigma_log=sigma_log, n=n,
        log_likelihood=float(log_lik), aic=float(aic),
        mean_x=mean_x, var_x=var_x,
    )


# ----------------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------------

@dataclass
class ModelComparison:
    pair: Tuple[str, str]
    gaussian: GaussianFit
    lognormal: LogNormalFit
    log_lik_ratio: float    # ell_lognormal - ell_gaussian
    aic_diff: float         # AIC_gaussian - AIC_lognormal  (positive => log-normal wins)
    winner: str             # "lognormal" or "gaussian"


def compare_models(data: np.ndarray, pair: Tuple[str, str]) -> ModelComparison:
    """Fit both models to the same positive data and pick the winner."""
    g = mle_gaussian(data)
    ln = mle_lognormal(data)
    delta = ln.log_likelihood - g.log_likelihood
    return ModelComparison(
        pair=pair,
        gaussian=g, lognormal=ln,
        log_lik_ratio=delta,
        aic_diff=(g.aic - ln.aic),
        winner=("lognormal" if delta > 0 else "gaussian"),
    )


def fit_all_corridors(df: pd.DataFrame,
                      min_obs: int = 30) -> Dict[Tuple[str, str], ModelComparison]:
    """Fit and compare both models on every corridor in CURRENCY_PAIRS.

    Returns a dict keyed by (c_from, c_to). Corridors with insufficient data
    are silently skipped.
    """
    out: Dict[Tuple[str, str], ModelComparison] = {}
    for (c_from, c_to) in CURRENCY_PAIRS:
        rates = get_rate_series(df, c_from, c_to)
        if rates is None or len(rates) < min_obs:
            continue
        vol = compute_corridor_volatility_bps(rates)
        vol = vol[vol > 0]
        if len(vol) < min_obs:
            continue
        try:
            out[(c_from, c_to)] = compare_models(vol, (c_from, c_to))
        except ValueError:
            continue
    return out


def summarize_fits(fits: Dict[Tuple[str, str], ModelComparison]) -> pd.DataFrame:
    """Tidy summary table of all corridor fits."""
    rows = []
    for pair, c in fits.items():
        rows.append({
            "from": pair[0],
            "to": pair[1],
            "n": c.gaussian.n,
            "gauss_mu_bps": c.gaussian.mu,
            "gauss_sigma_bps": c.gaussian.sigma,
            "gauss_logL": c.gaussian.log_likelihood,
            "lognorm_E[X]_bps": c.lognormal.mean_x,
            "lognorm_SD[X]_bps": np.sqrt(c.lognormal.var_x),
            "lognorm_logL": c.lognormal.log_likelihood,
            "delta_logL": c.log_lik_ratio,
            "winner": c.winner,
        })
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
# Diagnostic plot
# ----------------------------------------------------------------------------

def plot_corridor_fits(df: pd.DataFrame,
                       pairs_to_plot: List[Tuple[str, str]],
                       filename: str = "mle_corridor_fits.png") -> None:
    """Histogram + fitted Gaussian + fitted Log-Normal for a list of corridors."""
    pairs_to_plot = [p for p in pairs_to_plot
                     if get_rate_series(df, *p) is not None]
    if not pairs_to_plot:
        print("plot_corridor_fits: no plottable corridors")
        return

    n = len(pairs_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (c_from, c_to) in zip(axes, pairs_to_plot):
        rates = get_rate_series(df, c_from, c_to)
        vol = compute_corridor_volatility_bps(rates)
        vol = vol[vol > 0]
        if len(vol) < 30:
            ax.set_title(f"{c_from}->{c_to}: too little data")
            continue

        comp = compare_models(vol, (c_from, c_to))

        # Clip extreme tail for readability without altering the fit
        x_hi = float(np.quantile(vol, 0.995))
        bins = np.linspace(0, x_hi, 60)
        ax.hist(vol, bins=bins, density=True, alpha=0.45,
                color="#888780", edgecolor="white", label="empirical")

        x_grid = np.linspace(1e-3, x_hi, 500)

        g = comp.gaussian
        gauss_pdf = (1.0 / (g.sigma * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((x_grid - g.mu) / g.sigma) ** 2)
        ax.plot(x_grid, gauss_pdf, color="#534AB7", lw=2.0,
                label=f"Gaussian  $\\ell$={g.log_likelihood:.0f}")

        ln = comp.lognormal
        ln_pdf = (1.0 / (x_grid * ln.sigma_log * np.sqrt(2 * np.pi))) * \
                 np.exp(-0.5 * ((np.log(x_grid) - ln.mu_log) / ln.sigma_log) ** 2)
        ax.plot(x_grid, ln_pdf, color="#1D9E75", lw=2.0,
                label=f"Log-Normal  $\\ell$={ln.log_likelihood:.0f}")

        ax.set_xlim(0, x_hi)
        ax.set_xlabel("daily |log return|  (bps)")
        ax.set_ylabel("density")
        marker = "  ★" if comp.winner == "lognormal" else ""
        ax.set_title(f"{c_from} → {c_to}   "
                     f"$\\Delta\\ell$ = {comp.log_lik_ratio:+.1f}{marker}",
                     fontsize=11)
        ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ----------------------------------------------------------------------------
# Integration with the routing simulator
# ----------------------------------------------------------------------------

# Per-institution tier multipliers — premium fintechs run tighter spreads,
# traditional banks run looser ones. These are structural priors (not fit
# from data); the writeup should be explicit about this.
INSTITUTION_TIER: Dict[str, float] = {
    "Wise_US":            0.7,
    "Revolut_EU":         0.7,
    "Chase_US":           0.9,
    "N26_EU":             1.0,
    "Nubank_BR":          1.1,
    "BancoDoBrasil_BR":   1.2,
    "Itau_BR":            1.2,
    "Continental_PY":     1.3,
    "BancoNacional_PY":   1.3,
    "Santander_ES":       1.3,
    "VisionBanco_PY":     1.4,
    "SFCU_US":            1.0,   # no FX, irrelevant
}


def corridor_sigma_bps(fits: Dict[Tuple[str, str], ModelComparison],
                       pair: Tuple[str, str],
                       default: float = 10.0) -> float:
    """sigma (in bps) of corridor volatility, taken from the winning fit.

    For Gaussian: the fitted sigma.
    For Log-Normal: sqrt(Var[X]) — std of X itself, not of log X.
    """
    if pair not in fits:
        return default
    c = fits[pair]
    if c.winner == "lognormal":
        return float(np.sqrt(c.lognormal.var_x))
    else:
        return float(c.gaussian.sigma)


def apply_mle_estimates(institutions: dict,
                        fits: Dict[Tuple[str, str], ModelComparison],
                        verbose: bool = True) -> dict:
    """Overwrite every institution's fx_markup_std_bps using MLE estimates.

    sigma_inst,pair = tier_multiplier(inst) * sigma_corridor(pair)

    Means (fx_markup_mean_bps) are LEFT ALONE — those represent structural
    institutional bias and we don't have observable per-institution markup
    data to fit them from.
    """
    n_updated = 0
    for name, inst in institutions.items():
        tier = INSTITUTION_TIER.get(name, 1.0)
        for pair in inst.fx_pairs:
            sigma = corridor_sigma_bps(fits, pair, default=10.0)
            inst.fx_markup_std_bps[pair] = sigma * tier
            n_updated += 1
    if verbose:
        print(f"apply_mle_estimates: updated sigma for {n_updated} "
              f"(institution, pair) combinations.")
    return institutions
