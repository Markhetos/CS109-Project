"""
efficient_frontier.py — Mean-variance efficient frontier of feasible payment paths.

Phase 3 of the CS109 project. For a given transfer directive, enumerate every
feasible multi-hop path through the institution-currency network, simulate each
path under the Phase 1/2 stochastic model, and plot them in (risk, return) space
to reveal the Pareto frontier of cross-border payment routing.

CS109 CONTENT
-------------

(1) Mean-variance utility (CS109 expected utility + risk aversion)

    For a risk-averse agent with mean-variance preferences,
        U(X) = E[X] − (λ/2) · Var[X],
    where λ ≥ 0 is the risk-aversion parameter. This is the canonical
    first-order approximation to E[U(X)] for any concave U (Pratt-Arrow):
        E[U(X)]  ≈  U(E[X])  +  (1/2) · U''(E[X]) · Var[X]
    Setting U(x) = x − (λ/2)·x² (quadratic utility) gives U''(x) = −λ, so
    E[U(X)] reduces (up to a constant offset) to E[X] − (λ/2)·Var[X].

(2) Pareto optimality (multi-objective decision making)

    Path A dominates path B iff  E_A ≥ E_B  AND  σ_A ≤ σ_B,
    with at least one strict inequality. The Pareto frontier is the set of
    non-dominated paths. For every λ ≥ 0, the path maximizing the mean-variance
    utility above lies on the Pareto frontier — varying λ traces along it.

(3) Monte Carlo estimators

    For each path, we estimate E[ratio] and Var[ratio] from N independent
    routing episodes (one random FX date + OU-driven markup noise per run).
    By the CLT, the standard error of the mean estimator scales as σ̂ / √N.

Marco Paes — CS109 Probability Challenge, June 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ----------------------------------------------------------------------------
# Path enumeration and parsing
# ----------------------------------------------------------------------------

def enumerate_paths(
    graph: nx.DiGraph,
    source: Tuple[str, str],
    dest: Tuple[str, str],
    max_length: int = 6,
) -> List[List[Tuple[str, str]]]:
    """All simple paths from source to dest of length ≤ max_length (in edges)."""
    if source not in graph or dest not in graph:
        return []
    return list(nx.all_simple_paths(graph, source, dest, cutoff=max_length))


def parse_path(nodes: List[Tuple[str, str]]) -> List[tuple]:
    """Convert a sequence of (bank, currency) nodes into a list of actions.

    Each consecutive pair (n_i, n_{i+1}) is either a same-currency transfer
    (different banks) or a same-bank FX (different currencies).
    """
    actions = []
    for i in range(len(nodes) - 1):
        b_from, c_from = nodes[i]
        b_to, c_to = nodes[i + 1]
        if b_from == b_to and c_from != c_to:
            actions.append(("fx", b_from, c_from, c_to))
        elif b_from != b_to and c_from == c_to:
            actions.append(("transfer", b_from, b_to, c_from))
        else:
            raise ValueError(f"Invalid transition: {nodes[i]} -> {nodes[i+1]}")
    return actions


# ----------------------------------------------------------------------------
# Path statistics
# ----------------------------------------------------------------------------

@dataclass
class PathStats:
    path_id: int
    nodes: List[Tuple[str, str]]
    actions: List[tuple]
    n_steps: int
    n_fx: int
    n_transfers: int
    final_amounts: np.ndarray
    ratios: np.ndarray
    mean_amount: float
    std_amount: float
    mean_ratio: float
    std_ratio: float
    success_rate: float

    def label(self, max_len: int = 28) -> str:
        """Compact route label, e.g. 'EUR→BRL@Itau → trf BR→PY → BRL→PYG'."""
        parts = []
        for a in self.actions:
            if a[0] == "fx":
                _, bank, c_from, c_to = a
                parts.append(f"{c_from}→{c_to}@{bank.split('_')[0]}")
            else:
                _, bf, bt, _ = a
                parts.append(f"trf {bf.split('_')[1]}→{bt.split('_')[1]}")
        s = " → ".join(parts)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"


# ----------------------------------------------------------------------------
# Path simulation under the Phase 1/2 stochastic model
# ----------------------------------------------------------------------------

def simulate_path_once(
    actions,
    institutions,
    df: pd.DataFrame,
    initial_amount: float,
    ou_engine,
    tier_table,
    rng: np.random.Generator,
    apply_transfer_fn,
    apply_fx_fn,
    get_commercial_rate_fn,
) -> Tuple[float, pd.Series]:
    """One Monte Carlo realization of a FIXED path of actions.

    Returns (final_amount, sampled_row). Returns (0, row) if any step yields
    a non-positive amount.

    The function takes the simulator helpers as parameters to avoid circular
    imports between this module and routing_sim_v2.py.
    """
    # Sample one historical FX day for this run
    idx = int(rng.integers(0, len(df)))
    row = df.iloc[idx]

    if ou_engine is not None:
        ou_engine.reset()

    amount = float(initial_amount)
    for action in actions:
        if action[0] == "transfer":
            _, bank_from, _, currency = action
            inst = institutions[bank_from]
            amount, _ = apply_transfer_fn(inst, currency, amount)
        else:  # "fx"
            _, bank, c_from, c_to = action
            inst = institutions[bank]
            amount, _, _, _ = apply_fx_fn(
                inst, c_from, c_to, amount, row,
                ou_engine=ou_engine, tier_table=tier_table,
            )
        if amount <= 0:
            return 0.0, row
    return amount, row


def evaluate_all_paths(
    paths: List[List[Tuple[str, str]]],
    institutions,
    df: pd.DataFrame,
    initial_amount: float,
    source_currency: str,
    dest_currency: str,
    apply_transfer_fn,
    apply_fx_fn,
    get_commercial_rate_fn,
    ou_engine=None,
    tier_table=None,
    num_runs: int = 300,
    seed: int = 42,
) -> List[PathStats]:
    """Run num_runs Monte Carlo episodes on every path; return PathStats list."""
    rng = np.random.default_rng(seed)
    out: List[PathStats] = []

    for path_id, nodes in enumerate(paths):
        try:
            actions = parse_path(nodes)
        except ValueError:
            continue
        if not actions:
            continue

        n_fx = sum(1 for a in actions if a[0] == "fx")
        n_transfers = sum(1 for a in actions if a[0] == "transfer")

        finals = np.empty(num_runs)
        ratios = np.empty(num_runs)

        for k in range(num_runs):
            amount, row = simulate_path_once(
                actions, institutions, df, initial_amount,
                ou_engine, tier_table, rng,
                apply_transfer_fn, apply_fx_fn, get_commercial_rate_fn,
            )
            try:
                R = get_commercial_rate_fn(source_currency, dest_currency, row)
                bench = initial_amount * R
            except (KeyError, ValueError):
                bench = 0.0
            finals[k] = amount
            ratios[k] = (amount / bench) if bench > 0 and amount > 0 else 0.0

        ok = ratios > 0
        if ok.sum() < 10:
            continue
        success = float(ok.mean())
        finals_ok = finals[ok]
        ratios_ok = ratios[ok]

        out.append(PathStats(
            path_id=path_id,
            nodes=nodes,
            actions=actions,
            n_steps=len(actions),
            n_fx=n_fx,
            n_transfers=n_transfers,
            final_amounts=finals,
            ratios=ratios,
            mean_amount=float(np.mean(finals_ok)),
            std_amount=float(np.std(finals_ok)),
            mean_ratio=float(np.mean(ratios_ok)),
            std_ratio=float(np.std(ratios_ok)),
            success_rate=success,
        ))
    return out


# ----------------------------------------------------------------------------
# Pareto frontier
# ----------------------------------------------------------------------------

def find_pareto_frontier(stats: List[PathStats]) -> List[int]:
    """Indices of paths NOT dominated by any other.

    Path j dominates path i iff
        mean_j >= mean_i  AND  std_j <= std_i,
    with at least one strict.
    """
    front = []
    for i, s in enumerate(stats):
        dominated = False
        for j, t in enumerate(stats):
            if i == j:
                continue
            if (t.mean_ratio >= s.mean_ratio
                and t.std_ratio <= s.std_ratio
                and (t.mean_ratio > s.mean_ratio or t.std_ratio < s.std_ratio)):
                dominated = True
                break
        if not dominated:
            front.append(i)
    # Sort along the frontier from low-risk to high-risk
    front.sort(key=lambda i: stats[i].std_ratio)
    return front


# ----------------------------------------------------------------------------
# Mean-variance optima over a λ grid
# ----------------------------------------------------------------------------

def mean_variance_optima(
    stats: List[PathStats],
    lambdas: np.ndarray,
    verbose: bool = False,
) -> Dict[float, int]:
    """For each λ, return the path index maximizing  E[ρ] − λ·Var[ρ].

    Defensive against pathological values: any path whose score is non-finite
    (NaN / ±inf) is skipped. If every score is non-finite for a given λ, we
    fall back to the path with the highest finite mean_ratio (the λ=0 answer).

    Set verbose=True to print the chosen path's (E, σ) for each λ — useful
    when the plotted stars look like they're in the wrong order.
    """
    if not stats:
        return {float(lam): -1 for lam in lambdas}

    # Pre-extract arrays so we can vectorize and guard against bad values.
    means = np.array([s.mean_ratio for s in stats], dtype=float)
    stds  = np.array([s.std_ratio  for s in stats], dtype=float)
    finite_mask = np.isfinite(means) & np.isfinite(stds)
    if not finite_mask.any():
        return {float(lam): -1 for lam in lambdas}

    out: Dict[float, int] = {}
    for lam_np in lambdas:
        lam = float(lam_np)
        scores = means - lam * (stds ** 2)
        # Invalidate non-finite entries explicitly
        scores = np.where(finite_mask & np.isfinite(scores), scores, -np.inf)
        best_i = int(np.argmax(scores))
        if not np.isfinite(scores[best_i]):
            # Fall back to highest finite mean
            best_i = int(np.argmax(np.where(finite_mask, means, -np.inf)))
        out[lam] = best_i
        if verbose:
            s = stats[best_i]
            print(f"  λ={lam:>8.3g}: path {best_i:>3d}   "
                  f"E[ρ]={s.mean_ratio:.4f}   σ[ρ]={s.std_ratio:.4f}   "
                  f"score={scores[best_i]:.4f}")
    return out


# ----------------------------------------------------------------------------
# The headline figure
# ----------------------------------------------------------------------------

def match_greedy_path(
    stats: List[PathStats],
    greedy_nodes: List[Tuple[str, str]],
) -> Optional[int]:
    """Find the index in `stats` whose node sequence matches `greedy_nodes`.

    Returns None if no enumerated path matches (which can happen if the
    greedy policy used more hops than max_length, or if the path was
    pruned for low success rate).
    """
    for i, s in enumerate(stats):
        if s.nodes == greedy_nodes:
            return i
    return None


def plot_efficient_frontier(
    stats: List[PathStats],
    frontier_idx: List[int],
    optima: Dict[float, int],
    filename: str = "efficient_frontier.png",
    directive_str: Optional[str] = None,
    greedy_idx: Optional[int] = None,
) -> None:
    """Two-panel headline figure:
        Left  — risk/return scatter with Pareto frontier and λ-optima
        Right — text panel listing Pareto-frontier paths in detail
    """
    fig = plt.figure(figsize=(14.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    front_set = set(frontier_idx)

    # ---- Non-frontier points ----
    for i, s in enumerate(stats):
        if i in front_set:
            continue
        ax.scatter(
            s.std_ratio, s.mean_ratio,
            s=40 + 12 * s.n_steps,
            c="#bdbdbd", alpha=0.55,
            edgecolors="white", linewidths=0.6, zorder=2,
        )

    # ---- Pareto frontier line + points ----
    if len(frontier_idx) >= 2:
        xs = [stats[i].std_ratio for i in frontier_idx]
        ys = [stats[i].mean_ratio for i in frontier_idx]
        ax.plot(xs, ys, "-", color="#534AB7", lw=1.6, alpha=0.55, zorder=3)
    for i in frontier_idx:
        s = stats[i]
        ax.scatter(
            s.std_ratio, s.mean_ratio,
            s=70 + 12 * s.n_steps,
            c="#534AB7", alpha=0.95,
            edgecolors="white", linewidths=1.3, zorder=4,
        )

    # ---- λ-optimal stars ----
    lambdas_sorted = sorted(optima.keys())
    cmap = cm.get_cmap("plasma")
    unique_label_seen = set()
    for j, lam in enumerate(lambdas_sorted):
        idx = optima[lam]
        s = stats[idx]
        c = cmap(j / max(len(lambdas_sorted) - 1, 1))
        label = f"λ={lam:g}" if idx not in unique_label_seen else None
        unique_label_seen.add(idx)
        ax.scatter(
            s.std_ratio, s.mean_ratio,
            s=260, marker="*", c=[c],
            edgecolors="black", linewidths=0.8, zorder=5, label=label,
        )

    # ---- Greedy policy choice (hollow orange ring) ----
    if greedy_idx is not None and 0 <= greedy_idx < len(stats):
        gs = stats[greedy_idx]
        ax.scatter(
            gs.std_ratio, gs.mean_ratio,
            s=420, marker="o", facecolors="none",
            edgecolors="#D85A30", linewidths=2.4,
            zorder=4.5, label="greedy policy choice",
        )

    # ---- Decorations ----
    ax.axhline(1.0, color="#888", lw=0.7, ls="--", alpha=0.6)
    ax.text(
        0.01, 1.0, "commercial benchmark (ρ = 1)",
        transform=ax.get_yaxis_transform(),
        fontsize=9, color="#666", va="bottom",
    )
    ax.set_xlabel("Risk:  σ[fairness ratio]", fontsize=11)
    ax.set_ylabel("Return:  E[fairness ratio]", fontsize=11)

    title = "Pareto frontier of feasible payment paths"
    if directive_str:
        title += f"\n{directive_str}"
    ax.set_title(title, fontsize=12)

    # Two legends: one for sizing/colors, one for λ
    handles_size = [
        plt.scatter([], [], s=40 + 12 * k, c="#bdbdbd",
                    edgecolors="white", linewidths=0.6, label=f"{k} steps")
        for k in [2, 3, 4, 5]
    ]
    legend1 = ax.legend(
        handles=handles_size, loc="lower right",
        fontsize=8, framealpha=0.9, title="Path length",
        title_fontsize=8,
    )
    ax.add_artist(legend1)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2,
              title="Mean-variance optimum at λ", title_fontsize=9)

    # ---- Right panel: frontier path details ----
    ax_text.axis("off")
    header = f"Pareto frontier ({len(frontier_idx)} paths)\n"
    header += "─" * 44 + "\n"
    text_lines = [header]
    for rank, i in enumerate(frontier_idx, start=1):
        s = stats[i]
        text_lines.append(
            f"{rank:>2}.  E[ρ] = {s.mean_ratio:6.4f}   "
            f"σ[ρ] = {s.std_ratio:6.4f}   ({s.n_steps} steps, "
            f"{s.n_fx} FX)"
        )
        # show node path, abbreviated
        node_str = " → ".join(f"{n[0].split('_')[0]}({n[1]})" for n in s.nodes)
        if len(node_str) > 50:
            node_str = node_str[:47] + "..."
        text_lines.append(f"     {node_str}\n")

    ax_text.text(
        0.0, 0.98, "\n".join(text_lines),
        transform=ax_text.transAxes, fontsize=8.5,
        family="monospace", verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ----------------------------------------------------------------------------
# CLT confidence interval on the population mean fairness ratio
# ----------------------------------------------------------------------------

def clt_ci_for_path(stats: PathStats, alpha: float = 0.05) -> Tuple[float, float]:
    """95% CI on E[ρ] for a single path, by the CLT.

        ρ̄ ~ N(E[ρ], Var[ρ] / N)   (asymptotically)
        CI = ρ̄ ± z_{α/2} · σ̂_ρ / √N
    """
    z = 1.959963984540054  # 0.975 quantile of N(0, 1) — hard-coded to avoid scipy
    n = stats.ratios[stats.ratios > 0].size
    if n < 2:
        return (float("nan"), float("nan"))
    se = stats.std_ratio / np.sqrt(n)
    return (stats.mean_ratio - z * se, stats.mean_ratio + z * se)
