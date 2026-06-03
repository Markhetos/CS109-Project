"""
ou_process.py — Ornstein-Uhlenbeck dynamics for FX corridor noise.

Phase 1 modeled corridor volatility as an i.i.d. draw from a static distribution.
Phase 2 lifts this to a temporally-correlated stochastic process.

WHAT WE MODEL
-------------
A first pass might fit OU to signed daily log returns r_t. In practice that
fails empirically: daily FX returns have near-zero autocorrelation (a known
fact, consistent with weak-form market efficiency). What IS persistent is
volatility — high-volatility days cluster with high-volatility days. So we
model log-volatility:

        Y_t = log |r_t|              (log of absolute daily log return, in bps)

and fit Y_t as an Ornstein-Uhlenbeck process:

        dY_t = θ(μ − Y_t) dt + σ dW_t

The stationary distribution of Y is N(μ, σ²/(2θ)), which means |r_t| in the
stationary regime is Log-Normal — EXACTLY the distribution Phase 1 fit by MLE.
The two phases are mathematically consistent: Phase 1 finds the marginal,
Phase 2 adds the time-domain dynamics on top.

DERIVATIONS  (unchanged from the signed-return formulation)
-----------

(1) Closed-form solution of the SDE
        Y_t = μ + (Y_0 − μ) e^{−θ t} + σ ∫_0^t e^{−θ(t−s)} dW_s

(2) Stationary distribution
        Y_∞ ~ N(μ, σ² / (2θ))

(3) Transition distribution (Markov property)
        Y_{t+Δt} | Y_t ~ N( μ + (Y_t − μ) e^{−θ Δt},
                            σ² (1 − e^{−2θ Δt}) / (2θ) )

(4) Discrete-time form: OU is AR(1)
        a = e^{−θ Δt},  b = μ(1−a),  s² = σ²(1−a²) / (2θ)
        Y_{t+Δt} = a · Y_t + b + ε,    ε ~ N(0, s²)

(5) MLE via AR(1) OLS (closed form)
        â  = Σ_t (Y_t − Ȳ_p)(Y_{t+1} − Ȳ_n) / Σ_t (Y_t − Ȳ_p)²
        b̂  = Ȳ_n − â Ȳ_p
        ŝ² = (1/n) Σ_t (Y_{t+1} − â Y_t − b̂)²

(6) Back to continuous-time parameters
        θ̂ = − log(â) / Δt,    μ̂ = b̂ / (1 − â),    σ̂² = ŝ² · 2θ̂ / (1 − â²)

Marco Paes — CS109 Probability Challenge, June 2026
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from mle_estimation import get_rate_series, CURRENCY_PAIRS, INSTITUTION_TIER


# Small floor for log(|r|) to avoid log(0) on zero-return days
LOG_VOL_EPS_BPS = 0.5


# ----------------------------------------------------------------------------
# Data preprocessing — TWO modes
# ----------------------------------------------------------------------------

def compute_signed_log_returns_bps(rates: np.ndarray) -> np.ndarray:
    """Signed daily log returns in basis points. Used for the 'signed' mode."""
    if len(rates) < 2:
        return np.array([])
    return np.diff(np.log(rates)) * 10_000.0


def compute_log_volatility(rates: np.ndarray,
                           eps_bps: float = LOG_VOL_EPS_BPS) -> np.ndarray:
    """log |daily log return in bps|. Used for the 'log_volatility' mode.

    Rates that don't move (|r| below eps_bps) are dropped, since log(0) is -∞.
    Empirically real FX data has near-zero-return days, especially for
    illiquid corridors; dropping them is more honest than clamping.
    """
    if len(rates) < 2:
        return np.array([])
    r_bps = np.diff(np.log(rates)) * 10_000.0
    abs_r = np.abs(r_bps)
    abs_r = abs_r[abs_r > eps_bps]
    return np.log(abs_r)


# ----------------------------------------------------------------------------
# OU MLE via AR(1) OLS — now surfaces both raw and clipped â
# ----------------------------------------------------------------------------

@dataclass
class OUFit:
    theta: float
    mu: float
    sigma: float
    dt: float
    n: int
    log_likelihood: float
    a: float                     # clipped to (0, 1) for stable OU
    a_raw: float                 # RAW OLS estimate (may be ≤ 0 or > 1)
    a_was_clipped: bool          # True if a_raw fell outside (0, 1)
    b: float
    s: float
    stationary_mean: float
    stationary_std: float
    half_life: float
    mode: str = "signed"         # "signed" or "log_volatility"


def fit_ou_mle(x: np.ndarray, dt: float = 1.0, mode: str = "signed") -> OUFit:
    """Closed-form OU MLE via AR(1) OLS. See module docstring for derivation.

    The raw OLS â is returned as `a_raw`; we additionally clip into (0, 1)
    so the back-conversion to (θ, μ, σ) is stable. `a_was_clipped` flags when
    this happened — useful for diagnostics on empirical data where weak/no
    mean reversion is common.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        raise ValueError("Need at least 10 observations for OU MLE.")

    x_prev = x[:-1]
    x_next = x[1:]
    n = len(x_prev)

    x_bar = float(np.mean(x_prev))
    y_bar = float(np.mean(x_next))
    cov_xy = float(np.mean((x_prev - x_bar) * (x_next - y_bar)))
    var_x = float(np.mean((x_prev - x_bar) ** 2))
    if var_x <= 0:
        raise ValueError("Degenerate: zero variance in the lagged series.")

    a_raw = cov_xy / var_x
    a_clipped = max(1e-4, min(0.9999, a_raw))
    was_clipped = (a_raw != a_clipped)

    b_hat = y_bar - a_clipped * x_bar
    residuals = x_next - a_clipped * x_prev - b_hat
    s2_hat = float(np.mean(residuals ** 2))
    if s2_hat <= 0:
        raise ValueError("Degenerate: zero residual variance.")
    s_hat = float(np.sqrt(s2_hat))

    theta_hat = -np.log(a_clipped) / dt
    mu_hat = b_hat / (1.0 - a_clipped) if abs(1.0 - a_clipped) > 1e-10 else y_bar
    sigma2_hat = s2_hat * 2.0 * theta_hat / (1.0 - a_clipped ** 2)
    sigma_hat = float(np.sqrt(max(sigma2_hat, 1e-12)))

    stat_std = float(np.sqrt(sigma_hat ** 2 / (2.0 * theta_hat)))
    half_life = float(np.log(2.0) / theta_hat) if theta_hat > 0 else float("inf")
    log_lik = -0.5 * n * np.log(2.0 * np.pi * s2_hat) - 0.5 * n

    return OUFit(
        theta=float(theta_hat), mu=float(mu_hat), sigma=sigma_hat,
        dt=float(dt), n=int(n), log_likelihood=float(log_lik),
        a=float(a_clipped), a_raw=float(a_raw), a_was_clipped=was_clipped,
        b=float(b_hat), s=s_hat,
        stationary_mean=float(mu_hat),
        stationary_std=stat_std,
        half_life=half_life,
        mode=mode,
    )


# ----------------------------------------------------------------------------
# Simulators
# ----------------------------------------------------------------------------

def simulate_ou_exact(fit: OUFit, x0: float, n_steps: int,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simulate an OU path using the EXACT discrete-time transition (AR(1))."""
    if rng is None:
        rng = np.random.default_rng()
    path = np.empty(n_steps + 1)
    path[0] = x0
    for t in range(n_steps):
        path[t + 1] = fit.a * path[t] + fit.b + fit.s * rng.standard_normal()
    return path


def simulate_ou_euler(fit: OUFit, x0: float, n_steps: int,
                      dt: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simulate an OU path via Euler-Maruyama (the standard SDE recipe).

        X_{t+dt} = X_t + θ(μ − X_t) dt + σ √dt · N(0, 1)
    """
    if rng is None:
        rng = np.random.default_rng()
    if dt is None:
        dt = fit.dt
    sqrt_dt = np.sqrt(dt)
    path = np.empty(n_steps + 1)
    path[0] = x0
    for t in range(n_steps):
        drift = fit.theta * (fit.mu - path[t]) * dt
        diffusion = fit.sigma * sqrt_dt * rng.standard_normal()
        path[t + 1] = path[t] + drift + diffusion
    return path


def sample_stationary(fit: OUFit, rng: Optional[np.random.Generator] = None) -> float:
    """Draw one sample from N(μ, σ²/(2θ)) — the OU stationary distribution."""
    if rng is None:
        rng = np.random.default_rng()
    return fit.stationary_mean + fit.stationary_std * rng.standard_normal()


# ----------------------------------------------------------------------------
# Corridor-level fitting — picks the data prep based on `mode`
# ----------------------------------------------------------------------------

def fit_ou_all_corridors(df: pd.DataFrame,
                         dt: float = 1.0,
                         mode: str = "log_volatility",
                         min_obs: int = 50) -> Dict[Tuple[str, str], OUFit]:
    """Fit an OU process per corridor.

    mode='log_volatility' (default and recommended): models log |daily log return|.
        Volatility clusters in time, so OU has real autocorrelation to find.
        Stationary distribution on the raw |r| scale is Log-Normal — matches
        the Phase 1 fit by construction.

    mode='signed': models signed daily log returns. Honest but typically gives
        â ≈ 0 (FX returns are nearly i.i.d.), so OU collapses to i.i.d. baseline.
        Included for completeness and comparison.
    """
    if mode not in ("log_volatility", "signed"):
        raise ValueError("mode must be 'log_volatility' or 'signed'")

    prep = (compute_log_volatility if mode == "log_volatility"
            else compute_signed_log_returns_bps)

    out: Dict[Tuple[str, str], OUFit] = {}
    for (c_from, c_to) in CURRENCY_PAIRS:
        rates = get_rate_series(df, c_from, c_to)
        if rates is None or len(rates) < min_obs:
            continue
        series = prep(rates)
        if len(series) < min_obs:
            continue
        try:
            out[(c_from, c_to)] = fit_ou_mle(series, dt=dt, mode=mode)
        except ValueError:
            continue
    return out


def summarize_ou_fits(fits: Dict[Tuple[str, str], OUFit]) -> pd.DataFrame:
    rows = []
    for pair, f in fits.items():
        rows.append({
            "from": pair[0],
            "to": pair[1],
            "n": f.n,
            "theta": f.theta,
            "mu": f.mu,
            "sigma": f.sigma,
            "stat_std": f.stationary_std,
            "a_raw": f.a_raw,
            "a_used": f.a,
            "clipped": f.a_was_clipped,
            "half_life_days": f.half_life,
            "logL": f.log_likelihood,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Diagnostic plot — mode-aware
# ----------------------------------------------------------------------------

def plot_ou_diagnostics(df: pd.DataFrame,
                        fits: Dict[Tuple[str, str], OUFit],
                        pairs_to_plot: List[Tuple[str, str]],
                        filename: str = "ou_diagnostics.png") -> None:
    """For each corridor: empirical histogram of the modeled series + OU
    stationary PDF overlay + a simulated path next to a snippet of the real one.
    Picks the data prep automatically from each fit's `mode`.
    """
    pairs_to_plot = [p for p in pairs_to_plot if p in fits]
    if not pairs_to_plot:
        return

    n = len(pairs_to_plot)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), squeeze=False)
    rng = np.random.default_rng(0)

    for row_idx, (c_from, c_to) in enumerate(pairs_to_plot):
        fit = fits[(c_from, c_to)]
        rates = get_rate_series(df, c_from, c_to)
        prep = (compute_log_volatility if fit.mode == "log_volatility"
                else compute_signed_log_returns_bps)
        emp = prep(rates)
        if len(emp) < 30:
            continue

        # --- Left: empirical histogram + OU stationary ---
        ax_l = axes[row_idx, 0]
        lo, hi = np.quantile(emp, [0.005, 0.995])
        bins = np.linspace(lo, hi, 60)
        ax_l.hist(emp, bins=bins, density=True, alpha=0.45,
                  color="#888780", edgecolor="white", label="empirical")
        x_grid = np.linspace(lo, hi, 400)
        pdf = (1.0 / (fit.stationary_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_grid - fit.stationary_mean) / fit.stationary_std) ** 2
        )
        ax_l.plot(x_grid, pdf, color="#534AB7", lw=2.0,
                  label=f"OU stationary  N({fit.stationary_mean:.2f}, "
                        f"{fit.stationary_std:.2f}²)")
        ax_l.set_xlim(lo, hi)
        xlabel = ("log |daily log return|  (log-bps)"
                  if fit.mode == "log_volatility"
                  else "daily log return (bps)")
        ax_l.set_xlabel(xlabel)
        ax_l.set_ylabel("density")
        ax_l.set_title(f"{c_from} → {c_to}  stationary fit  [{fit.mode}]")
        ax_l.legend(fontsize=9, loc="upper right")

        # --- Right: empirical vs simulated trajectory ---
        ax_r = axes[row_idx, 1]
        window = min(250, len(emp))
        emp_window = emp[:window]
        sim = simulate_ou_exact(fit, x0=fit.stationary_mean,
                                 n_steps=window - 1, rng=rng)
        ax_r.plot(emp_window, color="#888780", lw=0.9, alpha=0.85, label="empirical")
        ax_r.plot(sim, color="#1D9E75", lw=0.9, alpha=0.85, label="OU simulated")
        ax_r.set_xlabel("day")
        ax_r.set_ylabel(xlabel)
        clipped_note = " (clipped)" if fit.a_was_clipped else ""
        ax_r.set_title(
            f"{c_from} → {c_to}   θ={fit.theta:.3f}/day   "
            f"half-life={fit.half_life:.1f}d   â_raw={fit.a_raw:.3f}{clipped_note}"
        )
        ax_r.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ----------------------------------------------------------------------------
# Simulator integration: OUMarkupEngine — mode-aware
# ----------------------------------------------------------------------------

def _canonical_pair(fits: Dict[Tuple[str, str], OUFit],
                    c_from: str, c_to: str) -> Optional[Tuple[str, str]]:
    if (c_from, c_to) in fits:
        return (c_from, c_to)
    if (c_to, c_from) in fits:
        return (c_to, c_from)
    return None


class OUMarkupEngine:
    """Stateful per-episode source of corridor-correlated markup noise.

    Mode-aware conversion of the OU state Y_t to a markup noise (bps):

      mode='log_volatility'  (recommended):
          Y_t is log-volatility; v_t = exp(Y_t) is volatility magnitude in bps.
          Returned noise has a random sign for symmetry:
              noise_t = sign_t · (v_t − E_stat[v])
          where sign_t ~ Uniform{+1, −1}, and E_stat[v] = exp(μ + (σ²/(2θ))/2)
          is the Log-Normal stationary mean of v. Subtracting it centers the
          noise around zero so the institutional bias μ_inst still represents
          the mean markup. The MAGNITUDE of noise is autocorrelated through
          OU dynamics (volatility clustering); the sign is independent.

      mode='signed':
          noise_t = Y_t directly (Y is already a signed return).

    Within an episode, magnitudes are autocorrelated through the OU process.
    Across episodes, reset() re-draws initial states from the stationary
    distribution.
    """

    def __init__(self, fits: Dict[Tuple[str, str], OUFit],
                 rng: Optional[np.random.Generator] = None):
        self.fits = fits
        self.rng = rng if rng is not None else np.random.default_rng()
        self.states: Dict[Tuple[str, str], float] = {}

        modes = {f.mode for f in fits.values()}
        if len(modes) > 1:
            warnings.warn(f"Mixed OU modes in fits: {modes}; behavior undefined.")
        self.mode = next(iter(modes)) if modes else "signed"

        # Precompute the stationary mean of v = exp(Y) for centering
        self._E_stat_v: Dict[Tuple[str, str], float] = {}
        if self.mode == "log_volatility":
            for pair, f in fits.items():
                self._E_stat_v[pair] = float(
                    np.exp(f.stationary_mean + 0.5 * f.stationary_std ** 2)
                )

    def reset(self) -> None:
        self.states = {}

    def step(self, c_from: str, c_to: str) -> float:
        pair = _canonical_pair(self.fits, c_from, c_to)
        if pair is None:
            return 0.0
        fit = self.fits[pair]

        if pair not in self.states:
            y = sample_stationary(fit, self.rng)
        else:
            y_prev = self.states[pair]
            y = fit.a * y_prev + fit.b + fit.s * self.rng.standard_normal()
        self.states[pair] = y

        if self.mode == "log_volatility":
            v = float(np.exp(y))
            sign = 1.0 if self.rng.standard_normal() > 0 else -1.0
            return sign * (v - self._E_stat_v[pair])
        else:
            return float(y)


def sample_markup_bps(institution_name: str,
                      mean_bps: float,
                      c_from: str,
                      c_to: str,
                      ou_engine: Optional[OUMarkupEngine],
                      tier_table: Dict[str, float],
                      rng: Optional[np.random.Generator] = None,
                      fallback_std_bps: float = 10.0) -> float:
    """Unified markup sampler used by the simulator. See OUMarkupEngine docstring."""
    if rng is None:
        rng = np.random.default_rng()
    tier = tier_table.get(institution_name, 1.0)
    if ou_engine is not None:
        corridor_noise = ou_engine.step(c_from, c_to)
        return mean_bps + tier * corridor_noise
    else:
        return mean_bps + tier * rng.standard_normal() * fallback_std_bps