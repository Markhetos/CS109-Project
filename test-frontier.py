"""
test_frontier.py — Synthetic tests for efficient_frontier.py.

(1) parse_path correctly classifies transfer vs FX edges.
(2) Pareto frontier correctly identifies non-dominated paths on hand-checked
    fake data.
(3) Mean-variance λ-sweep recovers the expected path at λ=0 (max E) and λ→∞
    (min σ).
(4) End-to-end: small dummy network → enumerate paths → simulate → frontier.
"""

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple

from efficient_frontier import (
    parse_path, find_pareto_frontier, mean_variance_optima,
    enumerate_paths, evaluate_all_paths, plot_efficient_frontier,
    PathStats,
)


# ---------- Test 1: parse_path ----------

def test_parse_path():
    print("Test 1: parse_path classifies transitions correctly")
    nodes = [
        ("Itau_BR", "EUR"),
        ("Itau_BR", "BRL"),         # FX
        ("BancoNacional_PY", "BRL"),# transfer
        ("BancoNacional_PY", "PYG"),# FX
        ("Continental_PY", "PYG"),  # transfer
    ]
    actions = parse_path(nodes)
    assert len(actions) == 4
    assert actions[0] == ("fx", "Itau_BR", "EUR", "BRL")
    assert actions[1] == ("transfer", "Itau_BR", "BancoNacional_PY", "BRL")
    assert actions[2] == ("fx", "BancoNacional_PY", "BRL", "PYG")
    assert actions[3] == ("transfer", "BancoNacional_PY", "Continental_PY", "PYG")
    print("  PASS")


# ---------- Test 2: Pareto frontier on hand-built fake data ----------

def make_fake_path_stat(idx, mean, std):
    """Build a fake PathStats with only mean_ratio and std_ratio populated."""
    return PathStats(
        path_id=idx, nodes=[], actions=[],
        n_steps=0, n_fx=0, n_transfers=0,
        final_amounts=np.array([1.0]), ratios=np.array([1.0]),
        mean_amount=0.0, std_amount=0.0,
        mean_ratio=mean, std_ratio=std,
        success_rate=1.0,
    )


def test_pareto_frontier():
    print("\nTest 2: find_pareto_frontier on hand-checked fake data")
    # Fake paths:
    #   A: (σ=0.01, μ=1.05) — low risk, low return
    #   B: (σ=0.05, μ=1.20) — high risk, high return
    #   C: (σ=0.03, μ=1.10) — mid both
    #   D: (σ=0.04, μ=1.05) — dominated by C (lower σ, same μ)
    #   E: (σ=0.02, μ=1.15) — strictly better than C and D — frontier
    stats = [
        make_fake_path_stat(0, 1.05, 0.01),  # A
        make_fake_path_stat(1, 1.20, 0.05),  # B
        make_fake_path_stat(2, 1.10, 0.03),  # C
        make_fake_path_stat(3, 1.05, 0.04),  # D - dominated
        make_fake_path_stat(4, 1.15, 0.02),  # E
    ]
    frontier = find_pareto_frontier(stats)
    print(f"  Frontier indices (sorted by σ): {frontier}")
    # Expected: A (lowest σ), E (better than C), B (highest μ).
    # C is dominated by E. D is dominated by A and E.
    assert set(frontier) == {0, 4, 1}, f"Got {set(frontier)}"
    assert frontier == [0, 4, 1], "Frontier should be sorted by σ ascending"
    print("  PASS")


# ---------- Test 3: λ-sweep extremes ----------

def test_mean_variance_optima():
    print("\nTest 3: λ-sweep extremes")
    stats = [
        make_fake_path_stat(0, 1.05, 0.01),
        make_fake_path_stat(1, 1.20, 0.05),
        make_fake_path_stat(2, 1.10, 0.03),
        make_fake_path_stat(3, 1.05, 0.04),
        make_fake_path_stat(4, 1.15, 0.02),
    ]
    # λ = 0: pure expected value maximization → path 1 (μ=1.20)
    # λ = ∞ (practically: 1e6): variance dominates → path 0 (σ=0.01)
    lambdas = np.array([0.0, 1.0, 10.0, 100.0, 1e6])
    opt = mean_variance_optima(stats, lambdas)
    print(f"  Optima: {opt}")
    assert opt[0.0] == 1, "λ=0 should pick max E"
    assert opt[1e6] == 0, "λ→∞ should pick min σ"
    print("  PASS")


# ---------- Test 4: end-to-end on a tiny synthetic setup ----------

def test_end_to_end_synthetic():
    """Two-bank toy network with hand-crafted parameters; verify pipeline runs."""
    print("\nTest 4: end-to-end on toy network (2 banks, 2 currencies)")

    # Mini Institution dataclass with just enough fields
    @dataclass
    class Inst:
        name: str
        fx_markup_mean_bps: dict = None
        fx_markup_std_bps: dict = None
        fx_perc_fee: dict = None
        transfer_fee: dict = None
        account_currencies: List[str] = None
        fx_pairs: List = None

    inst_A = Inst(
        name="BankA",
        fx_markup_mean_bps={("USD", "BRL"): 20.0},
        fx_markup_std_bps={("USD", "BRL"): 10.0},
        fx_perc_fee={("USD", "BRL"): 0.003},
        transfer_fee={"USD": 5.0, "BRL": 5.0},
        account_currencies=["USD", "BRL"],
        fx_pairs=[("USD", "BRL")],
    )
    inst_B = Inst(
        name="BankB",
        fx_markup_mean_bps={("USD", "BRL"): 30.0},
        fx_markup_std_bps={("USD", "BRL"): 15.0},
        fx_perc_fee={("USD", "BRL"): 0.005},
        transfer_fee={"USD": 3.0, "BRL": 3.0},
        account_currencies=["USD", "BRL"],
        fx_pairs=[("USD", "BRL")],
    )
    institutions = {"BankA": inst_A, "BankB": inst_B}

    # Tiny FX dataframe
    rng_data = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "USD_BRL": 5.0 * np.exp(rng_data.normal(0, 0.01, size=n)),
    })

    # Build a reachability graph by hand
    G = nx.DiGraph()
    for name in ["BankA", "BankB"]:
        for c in ["USD", "BRL"]:
            G.add_node((name, c))
    # Transfers same-currency between banks
    G.add_edge(("BankA", "USD"), ("BankB", "USD"), edge_type="transfer")
    G.add_edge(("BankB", "USD"), ("BankA", "USD"), edge_type="transfer")
    G.add_edge(("BankA", "BRL"), ("BankB", "BRL"), edge_type="transfer")
    G.add_edge(("BankB", "BRL"), ("BankA", "BRL"), edge_type="transfer")
    # FX within each bank
    G.add_edge(("BankA", "USD"), ("BankA", "BRL"), edge_type="fx")
    G.add_edge(("BankB", "USD"), ("BankB", "BRL"), edge_type="fx")

    # Local fakes for the simulator helpers (mimic routing_sim_v2.py signatures)
    def get_commercial_rate(c_from, c_to, row):
        if c_from == c_to: return 1.0
        if c_from == "USD" and c_to == "BRL": return float(row["USD_BRL"])
        if c_from == "BRL" and c_to == "USD": return 1.0 / float(row["USD_BRL"])
        raise KeyError(f"No rate for {c_from}->{c_to}")

    def apply_transfer_fn(inst, currency, amount):
        fee = inst.transfer_fee.get(currency, 0.0)
        return max(amount - fee, 0.0), fee

    def apply_fx_fn(inst, c_from, c_to, amount, row, ou_engine=None, tier_table=None):
        R = get_commercial_rate(c_from, c_to, row)
        mean_bps = inst.fx_markup_mean_bps.get((c_from, c_to), 0.0)
        std_bps = inst.fx_markup_std_bps.get((c_from, c_to), 0.0)
        markup_bps = np.random.normal(mean_bps, std_bps)
        R_eff = R * (1 + markup_bps / 10000.0)
        gross = amount * R_eff
        fee = inst.fx_perc_fee.get((c_from, c_to), 0.0) * gross
        net = max(gross - fee, 0.0)
        return net, R_eff, fee, R_eff - R

    paths = enumerate_paths(G, ("BankA", "USD"), ("BankB", "BRL"), max_length=4)
    print(f"  Found {len(paths)} paths.")
    assert len(paths) >= 2, "Should find multiple paths through this network"

    stats = evaluate_all_paths(
        paths, institutions, df, initial_amount=1000.0,
        source_currency="USD", dest_currency="BRL",
        apply_transfer_fn=apply_transfer_fn,
        apply_fx_fn=apply_fx_fn,
        get_commercial_rate_fn=get_commercial_rate,
        num_runs=100, seed=7,
    )
    print(f"  Evaluated {len(stats)} paths.")
    for s in stats:
        print(f"    path {s.path_id} ({s.n_steps} steps, {s.n_fx} FX): "
              f"E[ρ]={s.mean_ratio:.4f}, σ[ρ]={s.std_ratio:.4f}")

    frontier = find_pareto_frontier(stats)
    print(f"  Frontier: {frontier}")
    assert len(frontier) >= 1

    lambdas = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
    opt = mean_variance_optima(stats, lambdas)

    plot_efficient_frontier(
        stats, frontier, opt,
        filename="efficient_frontier_synthetic.png",
        directive_str="Toy: BankA(USD) → BankB(BRL), 1000 USD",
    )
    print("  PASS")


if __name__ == "__main__":
    print("=" * 64)
    test_parse_path()
    print("\n" + "=" * 64)
    test_pareto_frontier()
    print("\n" + "=" * 64)
    test_mean_variance_optima()
    print("\n" + "=" * 64)
    test_end_to_end_synthetic()
    print("\nAll tests passed.")