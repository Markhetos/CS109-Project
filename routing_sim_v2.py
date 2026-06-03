#!/usr/bin/env python3
"""
Stochastic Routing for Cross-Border Payments

This script simulates international payment routing across a network of banks.
The model captures:
- Institutions (banks) with different supported currencies and FX pairs
- Transfer relationships between banks (who can send to whom in which currencies)
- Institution-specific spreads vs commercial rate and fees
- A greedy, stepwise stochastic routing algorithm
- Monte-Carlo simulation over many FX scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from mle_estimation import (
    fit_all_corridors, summarize_fits,
    plot_corridor_fits, apply_mle_estimates,
    INSTITUTION_TIER,
)

from ou_process import (
    fit_ou_all_corridors, summarize_ou_fits, plot_ou_diagnostics,
    OUMarkupEngine,
)


# ============================================================================
# FX DATA LOADING
# ============================================================================

def load_fx_data(path: str) -> pd.DataFrame:
    """Load and clean FX rate data from CSV."""
    df = pd.read_csv(path)
    for col in df.columns:
        if col == "Date":
            continue
        df = df[df[col] > 0]
        # Simple outlier filter: keep within [1%, 99%] quantiles
        low, high = df[col].quantile([0.01, 0.99])
        df = df[(df[col] >= low) & (df[col] <= high)]
    return df.reset_index(drop=True)


# ============================================================================
# CURRENCY HELPERS
# ============================================================================

CURRENCIES = {"BRL", "PYG", "USD", "EUR"}

CURRENCY_DECIMALS = {
    "USD": 2,
    "EUR": 2,
    "BRL": 2,
    "PYG": 0,  # no cents
}


def round_currency(currency: str, amount: float) -> float:
    """Round amount to appropriate decimal places for currency."""
    decimals = CURRENCY_DECIMALS.get(currency, 2)
    return round(amount, decimals)


def _column_name(c_from: str, c_to: str) -> str:
    return f"{c_from}_{c_to}"


def get_commercial_rate(c_from: str, c_to: str, row: pd.Series) -> float:
    """
    Retrieve commercial rate c_from -> c_to from the row.
    If direct column is missing, try the inverse and invert.
    """
    if c_from == c_to:
        return 1.0

    col_direct = _column_name(c_from, c_to)
    if col_direct in row.index:
        return float(row[col_direct])

    col_inverse = _column_name(c_to, c_from)
    if col_inverse in row.index:
        val = float(row[col_inverse])
        if val <= 0:
            raise ValueError(f"Non-positive FX rate in inverse column {col_inverse}")
        return 1.0 / val

    raise KeyError(f"No commercial rate found for {c_from}->{c_to} in this row.")


# ============================================================================
# INSTITUTION CONFIGURATION
# ============================================================================

@dataclass
class Institution:
    """
    Represents a bank or financial institution in the payment network.
    
    Attributes:
        name: Unique identifier for the institution
        account_currencies: Currencies in which this bank can hold accounts
        fx_pairs: Allowed internal FX directions (cur_from, cur_to)
        transfer_fee: Flat fee per outbound transfer in a given currency
        fx_markup_mean_bps: Mean spread vs commercial in basis points per FX pair
        fx_markup_std_bps: Std dev of spread in bps per FX pair
        fx_perc_fee: Percentage FX fee on gross converted amount per FX pair
    """
    name: str
    account_currencies: List[str]
    fx_pairs: List[Tuple[str, str]] = field(default_factory=list)
    transfer_fee: Dict[str, float] = field(default_factory=dict)
    fx_markup_mean_bps: Dict[Tuple[str, str], float] = field(default_factory=dict)
    fx_markup_std_bps: Dict[Tuple[str, str], float] = field(default_factory=dict)
    fx_perc_fee: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def can_hold(self, currency: str) -> bool:
        return currency in self.account_currencies

    def can_fx(self, c_from: str, c_to: str) -> bool:
        return (c_from, c_to) in self.fx_pairs


def get_institutions() -> Dict[str, Institution]:
    """Create all institutions in the payment network."""
    institutions = {}

    # --- Brazil (BRL-focused) ---
    institutions["Itau_BR"] = Institution(
        name="Itau_BR",
        account_currencies=["BRL", "USD", "EUR", "PYG"],
        fx_pairs=[
            ("BRL", "USD"), ("USD", "BRL"),
            ("BRL", "EUR"), ("EUR", "BRL"),
            ("BRL", "PYG"), ("PYG", "BRL"),
        ],
        transfer_fee={"BRL": 5.0, "USD": 8.0, "EUR": 7.0, "PYG": 5000.0},
        fx_markup_mean_bps={
            ("BRL", "USD"): 25.0, ("USD", "BRL"): 25.0,
            ("BRL", "EUR"): 28.0, ("EUR", "BRL"): 28.0,
            ("BRL", "PYG"): 35.0, ("PYG", "BRL"): 35.0,
        },
        fx_markup_std_bps={
            ("BRL", "USD"): 10.0, ("USD", "BRL"): 10.0,
            ("BRL", "EUR"): 12.0, ("EUR", "BRL"): 12.0,
            ("BRL", "PYG"): 15.0, ("PYG", "BRL"): 15.0,
        },
        fx_perc_fee={
            ("BRL", "USD"): 0.003, ("USD", "BRL"): 0.003,
            ("BRL", "EUR"): 0.0035, ("EUR", "BRL"): 0.0035,
            ("BRL", "PYG"): 0.004, ("PYG", "BRL"): 0.004,
        },
    )

    institutions["BancoDoBrasil_BR"] = Institution(
        name="BancoDoBrasil_BR",
        account_currencies=["BRL", "USD", "EUR"],
        fx_pairs=[
            ("BRL", "USD"), ("USD", "BRL"),
            ("BRL", "EUR"), ("EUR", "BRL"),
        ],
        transfer_fee={"BRL": 4.0, "USD": 6.0, "EUR": 6.0},
        fx_markup_mean_bps={
            ("BRL", "USD"): 20.0, ("USD", "BRL"): 20.0,
            ("BRL", "EUR"): 22.0, ("EUR", "BRL"): 22.0,
        },
        fx_markup_std_bps={
            ("BRL", "USD"): 8.0, ("USD", "BRL"): 8.0,
            ("BRL", "EUR"): 9.0, ("EUR", "BRL"): 9.0,
        },
        fx_perc_fee={
            ("BRL", "USD"): 0.004, ("USD", "BRL"): 0.004,
            ("BRL", "EUR"): 0.0045, ("EUR", "BRL"): 0.0045,
        },
    )

    institutions["Nubank_BR"] = Institution(
        name="Nubank_BR",
        account_currencies=["BRL", "USD"],
        fx_pairs=[("BRL", "USD"), ("USD", "BRL")],
        transfer_fee={"BRL": 0.0, "USD": 3.0},
        fx_markup_mean_bps={
            ("BRL", "USD"): 35.0, ("USD", "BRL"): 35.0,
        },
        fx_markup_std_bps={
            ("BRL", "USD"): 12.0, ("USD", "BRL"): 12.0,
        },
        fx_perc_fee={
            ("BRL", "USD"): 0.006, ("USD", "BRL"): 0.006,
        },
    )

    # --- Paraguay (PYG-focused) ---
    institutions["BancoNacional_PY"] = Institution(
        name="BancoNacional_PY",
        account_currencies=["PYG", "USD", "BRL"],
        fx_pairs=[
            ("PYG", "USD"), ("USD", "PYG"),
            ("PYG", "BRL"), ("BRL", "PYG"),
        ],
        transfer_fee={"PYG": 15000.0, "USD": 5.0, "BRL": 8.0},
        fx_markup_mean_bps={
            ("PYG", "USD"): 30.0, ("USD", "PYG"): 30.0,
            ("PYG", "BRL"): 32.0, ("BRL", "PYG"): 32.0,
        },
        fx_markup_std_bps={
            ("PYG", "USD"): 12.0, ("USD", "PYG"): 12.0,
            ("PYG", "BRL"): 14.0, ("BRL", "PYG"): 14.0,
        },
        fx_perc_fee={
            ("PYG", "USD"): 0.002, ("USD", "PYG"): 0.002,
            ("PYG", "BRL"): 0.0025, ("BRL", "PYG"): 0.0025,
        },
    )

    institutions["VisionBanco_PY"] = Institution(
        name="VisionBanco_PY",
        account_currencies=["PYG", "USD"],
        fx_pairs=[("PYG", "USD"), ("USD", "PYG")],
        transfer_fee={"PYG": 8000.0, "USD": 4.0},
        fx_markup_mean_bps={
            ("PYG", "USD"): 40.0, ("USD", "PYG"): 40.0,
        },
        fx_markup_std_bps={
            ("PYG", "USD"): 15.0, ("USD", "PYG"): 15.0,
        },
        fx_perc_fee={
            ("PYG", "USD"): 0.0035, ("USD", "PYG"): 0.0035,
        },
    )

    institutions["Continental_PY"] = Institution(
        name="Continental_PY",
        account_currencies=["PYG", "USD", "EUR"],
        fx_pairs=[
            ("PYG", "USD"), ("USD", "PYG"),
            ("PYG", "EUR"), ("EUR", "PYG"),
        ],
        transfer_fee={"PYG": 10000.0, "USD": 6.0, "EUR": 7.0},
        fx_markup_mean_bps={
            ("PYG", "USD"): 35.0, ("USD", "PYG"): 35.0,
            ("PYG", "EUR"): 38.0, ("EUR", "PYG"): 38.0,
        },
        fx_markup_std_bps={
            ("PYG", "USD"): 10.0, ("USD", "PYG"): 10.0,
            ("PYG", "EUR"): 12.0, ("EUR", "PYG"): 12.0,
        },
        fx_perc_fee={
            ("PYG", "USD"): 0.0025, ("USD", "PYG"): 0.0025,
            ("PYG", "EUR"): 0.003, ("EUR", "PYG"): 0.003,
        },
    )

    # --- US (USD-focused) ---
    institutions["Chase_US"] = Institution(
        name="Chase_US",
        account_currencies=["USD"],  # No PYG accounts!
        fx_pairs=[
            ("USD", "BRL"), ("BRL", "USD"),
            ("USD", "EUR"), ("EUR", "USD"),
        ],
        transfer_fee={"USD": 10.0},
        fx_markup_mean_bps={
            ("USD", "BRL"): 18.0, ("BRL", "USD"): 18.0,
            ("USD", "EUR"): 15.0, ("EUR", "USD"): 15.0,
        },
        fx_markup_std_bps={
            ("USD", "BRL"): 6.0, ("BRL", "USD"): 6.0,
            ("USD", "EUR"): 5.0, ("EUR", "USD"): 5.0,
        },
        fx_perc_fee={
            ("USD", "BRL"): 0.002, ("BRL", "USD"): 0.002,
            ("USD", "EUR"): 0.0015, ("EUR", "USD"): 0.0015,
        },
    )

    institutions["SFCU_US"] = Institution(
        name="SFCU_US",  # Stanford Credit Union - sender's bank, no FX
        account_currencies=["USD"],
        fx_pairs=[],  # Cannot do FX
        transfer_fee={"USD": 5.0},
        fx_markup_mean_bps={},
        fx_markup_std_bps={},
        fx_perc_fee={},
    )

    institutions["Wise_US"] = Institution(
        name="Wise_US",
        account_currencies=["USD", "EUR", "BRL"],  # No PYG
        fx_pairs=[
            ("USD", "BRL"), ("BRL", "USD"),
            ("USD", "EUR"), ("EUR", "USD"),
            ("BRL", "EUR"), ("EUR", "BRL"),
        ],
        transfer_fee={"USD": 2.0, "EUR": 2.0, "BRL": 3.0},
        fx_markup_mean_bps={
            ("USD", "BRL"): 10.0, ("BRL", "USD"): 10.0,
            ("USD", "EUR"): 8.0, ("EUR", "USD"): 8.0,
            ("BRL", "EUR"): 12.0, ("EUR", "BRL"): 12.0,
        },
        fx_markup_std_bps={
            ("USD", "BRL"): 5.0, ("BRL", "USD"): 5.0,
            ("USD", "EUR"): 4.0, ("EUR", "USD"): 4.0,
            ("BRL", "EUR"): 6.0, ("EUR", "BRL"): 6.0,
        },
        fx_perc_fee={
            ("USD", "BRL"): 0.004, ("BRL", "USD"): 0.004,
            ("USD", "EUR"): 0.003, ("EUR", "USD"): 0.003,
            ("BRL", "EUR"): 0.0045, ("EUR", "BRL"): 0.0045,
        },
    )

    # --- Europe (EUR-focused) ---
    institutions["N26_EU"] = Institution(
        name="N26_EU",
        account_currencies=["EUR", "USD", "BRL"],  # No PYG
        fx_pairs=[
            ("EUR", "USD"), ("USD", "EUR"),
            ("EUR", "BRL"), ("BRL", "EUR"),
        ],
        transfer_fee={"EUR": 3.0, "USD": 4.0, "BRL": 5.0},
        fx_markup_mean_bps={
            ("EUR", "USD"): 22.0, ("USD", "EUR"): 22.0,
            ("EUR", "BRL"): 25.0, ("BRL", "EUR"): 25.0,
        },
        fx_markup_std_bps={
            ("EUR", "USD"): 8.0, ("USD", "EUR"): 8.0,
            ("EUR", "BRL"): 10.0, ("BRL", "EUR"): 10.0,
        },
        fx_perc_fee={
            ("EUR", "USD"): 0.003, ("USD", "EUR"): 0.003,
            ("EUR", "BRL"): 0.0035, ("BRL", "EUR"): 0.0035,
        },
    )

    institutions["Santander_ES"] = Institution(
        name="Santander_ES",
        account_currencies=["EUR", "BRL", "PYG"],
        fx_pairs=[
            ("EUR", "BRL"), ("BRL", "EUR"),
            ("EUR", "PYG"), ("PYG", "EUR"),
            ("BRL", "PYG"), ("PYG", "BRL"),
        ],
        transfer_fee={"EUR": 4.0, "BRL": 6.0, "PYG": 12000.0},
        fx_markup_mean_bps={
            ("EUR", "BRL"): 32.0, ("BRL", "EUR"): 32.0,
            ("EUR", "PYG"): 38.0, ("PYG", "EUR"): 38.0,
            ("BRL", "PYG"): 35.0, ("PYG", "BRL"): 35.0,
        },
        fx_markup_std_bps={
            ("EUR", "BRL"): 12.0, ("BRL", "EUR"): 12.0,
            ("EUR", "PYG"): 15.0, ("PYG", "EUR"): 15.0,
            ("BRL", "PYG"): 14.0, ("PYG", "BRL"): 14.0,
        },
        fx_perc_fee={
            ("EUR", "BRL"): 0.0045, ("BRL", "EUR"): 0.0045,
            ("EUR", "PYG"): 0.005, ("PYG", "EUR"): 0.005,
            ("BRL", "PYG"): 0.0048, ("PYG", "BRL"): 0.0048,
        },
    )

    institutions["Revolut_EU"] = Institution(
        name="Revolut_EU",
        account_currencies=["EUR", "USD", "BRL"],  # No PYG
        fx_pairs=[
            ("EUR", "USD"), ("USD", "EUR"),
            ("EUR", "BRL"), ("BRL", "EUR"),
        ],
        transfer_fee={"EUR": 0.0, "USD": 1.0, "BRL": 2.0},
        fx_markup_mean_bps={
            ("EUR", "USD"): 12.0, ("USD", "EUR"): 12.0,
            ("EUR", "BRL"): 15.0, ("BRL", "EUR"): 15.0,
        },
        fx_markup_std_bps={
            ("EUR", "USD"): 6.0, ("USD", "EUR"): 6.0,
            ("EUR", "BRL"): 8.0, ("BRL", "EUR"): 8.0,
        },
        fx_perc_fee={
            ("EUR", "USD"): 0.005, ("USD", "EUR"): 0.005,
            ("EUR", "BRL"): 0.0055, ("BRL", "EUR"): 0.0055,
        },
    )

    return institutions


# ============================================================================
# TRANSFER GRAPH
# ============================================================================

def get_transfer_graph() -> Dict[str, List[Tuple[str, str]]]:
    """
    Define which banks can transfer to which in each currency.
    TRANSFER_GRAPH[currency] = list of (from_bank, to_bank) tuples
    """
    return {
        "USD": [
            # US internal
            ("SFCU_US", "Chase_US"),
            ("SFCU_US", "Wise_US"),
            ("Chase_US", "Wise_US"),
            ("Wise_US", "Chase_US"),
            # US to Paraguay
            ("Chase_US", "BancoNacional_PY"),
            ("Chase_US", "VisionBanco_PY"),
            ("Chase_US", "Continental_PY"),
            ("Wise_US", "BancoNacional_PY"),
            ("Wise_US", "VisionBanco_PY"),
            # US to Brazil
            ("Chase_US", "Itau_BR"),
            ("Chase_US", "BancoDoBrasil_BR"),
            ("Chase_US", "Nubank_BR"),
            ("Wise_US", "Itau_BR"),
            ("Wise_US", "BancoDoBrasil_BR"),
            ("Wise_US", "Nubank_BR"),
            # US to Europe
            ("Wise_US", "N26_EU"),
            ("Wise_US", "Revolut_EU"),
            # Brazil to Paraguay
            ("Itau_BR", "BancoNacional_PY"),
            ("BancoDoBrasil_BR", "BancoNacional_PY"),
            # Europe to others
            ("N26_EU", "Wise_US"),
            ("Revolut_EU", "Wise_US"),
        ],
        "PYG": [
            # Paraguay internal
            ("BancoNacional_PY", "VisionBanco_PY"),
            ("BancoNacional_PY", "Continental_PY"),
            ("VisionBanco_PY", "BancoNacional_PY"),
            ("VisionBanco_PY", "Continental_PY"),
            ("Continental_PY", "BancoNacional_PY"),
            ("Continental_PY", "VisionBanco_PY"),
            # Paraguay to Brazil (PYG accounts)
            ("BancoNacional_PY", "Itau_BR"),
            ("VisionBanco_PY", "Itau_BR"),
            # Paraguay to Europe
            ("Continental_PY", "Santander_ES"),
            ("BancoNacional_PY", "Santander_ES"),
        ],
        "BRL": [
            # Brazil internal
            ("Itau_BR", "BancoDoBrasil_BR"),
            ("Itau_BR", "Nubank_BR"),
            ("BancoDoBrasil_BR", "Itau_BR"),
            ("BancoDoBrasil_BR", "Nubank_BR"),
            ("Nubank_BR", "Itau_BR"),
            ("Nubank_BR", "BancoDoBrasil_BR"),
            # Brazil to Paraguay
            ("Itau_BR", "BancoNacional_PY"),
            ("BancoDoBrasil_BR", "BancoNacional_PY"),
            # Brazil to US
            ("Itau_BR", "Wise_US"),
            ("BancoDoBrasil_BR", "Wise_US"),
            ("Nubank_BR", "Wise_US"),
            # Brazil to Europe
            ("Itau_BR", "N26_EU"),
            ("Itau_BR", "Santander_ES"),
            ("BancoDoBrasil_BR", "Santander_ES"),
            ("Itau_BR", "Revolut_EU"),
        ],
        "EUR": [
            # Europe internal
            ("N26_EU", "Santander_ES"),
            ("N26_EU", "Revolut_EU"),
            ("Santander_ES", "N26_EU"),
            ("Santander_ES", "Revolut_EU"),
            ("Revolut_EU", "N26_EU"),
            ("Revolut_EU", "Santander_ES"),
            # Europe to Brazil
            ("N26_EU", "Itau_BR"),
            ("Santander_ES", "Itau_BR"),
            ("Revolut_EU", "Itau_BR"),
            # Europe to US
            ("N26_EU", "Wise_US"),
            ("Revolut_EU", "Wise_US"),
            # Europe to Paraguay
            ("Santander_ES", "Continental_PY"),
        ],
    }


# ============================================================================
# FX CONVERSION HELPERS
# ============================================================================

def sample_effective_rate(
    inst: Institution,
    c_from: str,
    c_to: str,
    row: pd.Series,
    ou_engine=None,
    tier_table=None,
) -> Tuple[float, float]:
    """Sample an institution-specific effective rate for c_from -> c_to.
    Returns (R_eff, delta_vs_commercial).
    """
    R_comm = get_commercial_rate(c_from, c_to, row)
    mean_bps = inst.fx_markup_mean_bps.get((c_from, c_to), 0.0)
    std_bps  = inst.fx_markup_std_bps.get((c_from, c_to), 0.0)

    if ou_engine is not None and tier_table is not None:
        # OU-driven temporally correlated draw (Phase 2)
        tier = tier_table.get(inst.name, 1.0)
        corridor_noise_bps = ou_engine.step(c_from, c_to)
        markup_bps = mean_bps + tier * corridor_noise_bps
    else:
        # i.i.d. Gaussian fallback (Phase 1 behavior)
        markup_bps = np.random.normal(mean_bps, std_bps)

    multiplier = 1.0 + markup_bps / 10_000.0
    R_eff = R_comm * multiplier
    delta = R_eff - R_comm
    return R_eff, delta


def apply_fx_conversion(
    inst: Institution,
    c_from: str,
    c_to: str,
    amount: float,
    row: pd.Series,
    ou_engine=None,
    tier_table=None,
) -> Tuple[float, float, float, float]:
    """Apply a stochastic FX conversion within an institution.
    Returns (net_amount, R_eff, fee, delta_vs_commercial).
    """
    if amount <= 0:
        return 0.0, 0.0, 0.0, 0.0

    R_eff, delta = sample_effective_rate(
        inst, c_from, c_to, row, ou_engine=ou_engine, tier_table=tier_table
    )
    gross = amount * R_eff

    perc_fee_rate = inst.fx_perc_fee.get((c_from, c_to), 0.0)
    fee = perc_fee_rate * gross
    net = max(gross - fee, 0.0)

    net = round_currency(c_to, net)
    return net, R_eff, fee, delta


def apply_transfer(
    inst: Institution,
    currency: str,
    amount: float
) -> Tuple[float, float]:
    """
    Apply a transfer fee when sending from an institution.
    Returns (net_amount, fee).
    """
    if amount <= 0:
        return 0.0, 0.0

    fee = inst.transfer_fee.get(currency, 0.0)
    net = max(amount - fee, 0.0)
    net = round_currency(currency, net)
    return net, fee


# ============================================================================
# STATE AND STEP RECORDS
# ============================================================================

@dataclass
class State:
    """Current state in the routing: (institution, currency, amount)."""
    institution: str
    currency: str
    amount: float


@dataclass
class StepRecord:
    """Record of a single routing step."""
    step_type: str  # "transfer" or "fx"
    inst_from: str
    inst_to: str
    currency_from: str
    currency_to: str
    amount_before: float
    amount_after: float
    rate_eff: Optional[float]  # Only for FX steps
    fee: float
    delta_vs_comm: Optional[float]  # Only for FX steps


@dataclass
class GreedyResult:
    """Result of a greedy routing simulation."""
    success: bool
    final_amount: float
    ratio_vs_commercial: float
    path: List[StepRecord]
    commercial_benchmark: float
    source_bank: str
    source_currency: str
    dest_bank: str
    dest_currency: str
    initial_amount: float


# ============================================================================
# REACHABILITY GRAPH
# ============================================================================

def build_reachability_graph(
    institutions: Dict[str, Institution],
    transfer_graph: Dict[str, List[Tuple[str, str]]]
) -> nx.DiGraph:
    """
    Build a directed graph where nodes are (bank, currency) pairs.
    Edges represent either transfers or FX conversions.
    Used for reachability checks.
    """
    G = nx.DiGraph()

    # Add all valid (bank, currency) nodes
    for inst_name, inst in institutions.items():
        for cur in inst.account_currencies:
            G.add_node((inst_name, cur))

    # Add transfer edges (same currency, different banks)
    for currency, transfers in transfer_graph.items():
        for (from_bank, to_bank) in transfers:
            if (from_bank in institutions and to_bank in institutions and
                institutions[from_bank].can_hold(currency) and
                institutions[to_bank].can_hold(currency)):
                G.add_edge((from_bank, currency), (to_bank, currency), edge_type="transfer")

    # Add FX edges (same bank, different currencies)
    for inst_name, inst in institutions.items():
        for (c_from, c_to) in inst.fx_pairs:
            if inst.can_hold(c_from) and inst.can_hold(c_to):
                G.add_edge((inst_name, c_from), (inst_name, c_to), edge_type="fx")

    return G


def can_reach(
    G: nx.DiGraph,
    from_node: Tuple[str, str],
    to_node: Tuple[str, str]
) -> bool:
    """Check if to_node is reachable from from_node in the graph."""
    if from_node not in G or to_node not in G:
        return False
    return nx.has_path(G, from_node, to_node)


# ============================================================================
# CANDIDATE EVALUATION
# ============================================================================

def get_shortest_path_length(
    G: nx.DiGraph,
    from_node: Tuple[str, str],
    to_node: Tuple[str, str]
) -> int:
    """Get the shortest path length between two nodes, or infinity if unreachable."""
    if from_node not in G or to_node not in G:
        return float('inf')
    try:
        return nx.shortest_path_length(G, from_node, to_node)
    except nx.NetworkXNoPath:
        return float('inf')


def evaluate_transfer_candidate(
    inst: Institution,
    currency: str,
    amount: float,
    num_samples: int = 20
) -> float:
    """
    Estimate expected net amount after a transfer (deterministic for transfers).
    """
    net, _ = apply_transfer(inst, currency, amount)
    return net


def evaluate_fx_candidate(
    inst: Institution,
    c_from: str,
    c_to: str,
    amount: float,
    row: pd.Series,
    num_samples: int = 20
) -> float:
    """
    Estimate expected net amount after an FX conversion using Monte Carlo.
    """
    if amount <= 0:
        return 0.0

    outcomes = []
    for _ in range(num_samples):
        net, _, _, _ = apply_fx_conversion(inst, c_from, c_to, amount, row)
        outcomes.append(net)

    return float(np.mean(outcomes)) if outcomes else 0.0


def compute_candidate_score(
    expected_amount: float,
    current_distance: int,
    next_distance: int,
    dest_currency: str,
    next_currency: str,
    row: pd.Series,
) -> float:
    """
    Compute a score for a candidate action that balances:
    1. Expected amount (higher is better)
    2. Progress toward destination (decreasing distance is better)
    
    We normalize the amount by converting to destination currency equivalent
    using commercial rates, then add a bonus for reducing distance.
    """
    if expected_amount <= 0:
        return -float('inf')
    
    # Convert amount to destination currency equivalent for fair comparison
    try:
        rate_to_dest = get_commercial_rate(next_currency, dest_currency, row)
        dest_equivalent = expected_amount * rate_to_dest
    except (KeyError, ValueError):
        dest_equivalent = expected_amount  # fallback
    
    # Distance bonus: reward moves that get us closer to destination
    # Each step closer is worth a 5% bonus
    distance_change = current_distance - next_distance
    distance_bonus = 1.0 + (distance_change * 0.10)  # 10% bonus per step closer
    
    return dest_equivalent * distance_bonus


# ============================================================================
# GREEDY ROUTING POLICY
# ============================================================================

def greedy_policy(
    institutions: Dict[str, Institution],
    transfer_graph: Dict[str, List[Tuple[str, str]]],
    reachability_graph: nx.DiGraph,
    row: pd.Series,
    source_bank: str,
    source_currency: str,
    dest_bank: str,
    dest_currency: str,
    initial_amount: float,
    max_hops: int = 8,
    num_samples_per_candidate: int = 20,
    ou_engine=None,
    tier_table=None,
) -> GreedyResult:
    """
    Greedy stepwise stochastic routing with (institution, currency, amount) state.
    
    At each step:
    1. Build candidate actions (transfers and FX)
    2. Estimate expected amount for each using Monte Carlo
    3. Filter out actions that make destination unreachable
    4. Score candidates by amount normalized to dest currency + distance bonus
    5. Pick action with highest score
    6. Execute one actual realization of that action
    """
    current_state = State(
        institution=source_bank,
        currency=source_currency,
        amount=float(initial_amount)
    )

    dest_node = (dest_bank, dest_currency)
    visited_nodes: Set[Tuple[str, str]] = {(source_bank, source_currency)}
    path: List[StepRecord] = []

    # Commercial benchmark
    R_comm_s_t = get_commercial_rate(source_currency, dest_currency, row)
    benchmark = initial_amount * R_comm_s_t

    for hop in range(max_hops):
        current_node = (current_state.institution, current_state.currency)

        # Check if we've reached the destination
        if current_node == dest_node:
            break

        inst = institutions[current_state.institution]
        candidates = []
        
        # Get current distance to destination for scoring
        current_distance = get_shortest_path_length(reachability_graph, current_node, dest_node)

        # === Transfer candidates (same currency, different bank) ===
        for (from_bank, to_bank) in transfer_graph.get(current_state.currency, []):
            if from_bank != current_state.institution:
                continue
            if to_bank not in institutions:
                continue
            if not institutions[to_bank].can_hold(current_state.currency):
                continue

            next_node = (to_bank, current_state.currency)
            if next_node in visited_nodes:
                continue  # Avoid cycles

            # Check reachability to destination
            if not can_reach(reachability_graph, next_node, dest_node):
                continue

            expected_net = evaluate_transfer_candidate(
                inst, current_state.currency, current_state.amount, num_samples_per_candidate
            )

            if expected_net > 0:
                next_distance = get_shortest_path_length(reachability_graph, next_node, dest_node)
                score = compute_candidate_score(
                    expected_net, current_distance, next_distance,
                    dest_currency, current_state.currency, row
                )
                candidates.append({
                    "type": "transfer",
                    "to_bank": to_bank,
                    "to_currency": current_state.currency,
                    "expected_amount": expected_net,
                    "score": score,
                })

        # === FX candidates (same bank, different currency) ===
        for (c_from, c_to) in inst.fx_pairs:
            if c_from != current_state.currency:
                continue
            if not inst.can_hold(c_to):
                continue

            next_node = (current_state.institution, c_to)
            if next_node in visited_nodes:
                continue  # Avoid cycles

            # Check reachability to destination
            if not can_reach(reachability_graph, next_node, dest_node):
                continue

            expected_net = evaluate_fx_candidate(
                inst, c_from, c_to, current_state.amount, row, num_samples_per_candidate
            )

            if expected_net > 0:
                next_distance = get_shortest_path_length(reachability_graph, next_node, dest_node)
                score = compute_candidate_score(
                    expected_net, current_distance, next_distance,
                    dest_currency, c_to, row
                )
                candidates.append({
                    "type": "fx",
                    "to_bank": current_state.institution,
                    "to_currency": c_to,
                    "expected_amount": expected_net,
                    "score": score,
                })

        if not candidates:
            # Dead end
            break

        # Choose best candidate by score (considers amount + distance progress)
        best = max(candidates, key=lambda x: x["score"])

        # Execute the action
        amount_before = current_state.amount

        if best["type"] == "transfer":
            amount_after, fee = apply_transfer(inst, current_state.currency, current_state.amount)
            step = StepRecord(
                step_type="transfer",
                inst_from=current_state.institution,
                inst_to=best["to_bank"],
                currency_from=current_state.currency,
                currency_to=best["to_currency"],
                amount_before=amount_before,
                amount_after=amount_after,
                rate_eff=None,
                fee=fee,
                delta_vs_comm=None,
            )
            current_state = State(
                institution=best["to_bank"],
                currency=best["to_currency"],
                amount=amount_after
            )

        else:  # FX
            amount_after, rate_eff, fee, delta = apply_fx_conversion(
                inst, current_state.currency, best["to_currency"], current_state.amount, row,
                ou_engine=ou_engine, tier_table=tier_table,
            )
            step = StepRecord(
                step_type="fx",
                inst_from=current_state.institution,
                inst_to=current_state.institution,
                currency_from=current_state.currency,
                currency_to=best["to_currency"],
                amount_before=amount_before,
                amount_after=amount_after,
                rate_eff=rate_eff,
                fee=fee,
                delta_vs_comm=delta,
            )
            current_state = State(
                institution=current_state.institution,
                currency=best["to_currency"],
                amount=amount_after
            )

        path.append(step)
        visited_nodes.add((current_state.institution, current_state.currency))

        if current_state.amount <= 0:
            break

    # Check success
    final_node = (current_state.institution, current_state.currency)
    success = (final_node == dest_node) and (current_state.amount > 0)
    final_amount = current_state.amount if success else 0.0
    ratio = (final_amount / benchmark) if benchmark > 0 else 0.0

    return GreedyResult(
        success=success,
        final_amount=final_amount,
        ratio_vs_commercial=ratio,
        path=path,
        commercial_benchmark=benchmark,
        source_bank=source_bank,
        source_currency=source_currency,
        dest_bank=dest_bank,
        dest_currency=dest_currency,
        initial_amount=initial_amount,
    )


# ============================================================================
# EXPERIMENTS
# ============================================================================

def sample_commercial_row(df: pd.DataFrame) -> pd.Series:
    """Sample a random row from the FX dataframe."""
    return df.sample(1).iloc[0]


def run_greedy_experiments(
    df, institutions, transfer_graph,
    source_bank, source_currency, dest_bank, dest_currency, initial_amount,
    num_runs: int = 500, max_hops: int = 8,
    num_samples_per_candidate: int = 20,
    ou_fits=None,               # NEW
    tier_table=None,            # NEW
) -> List[GreedyResult]:
    reachability_graph = build_reachability_graph(institutions, transfer_graph)
    ou_engine = OUMarkupEngine(ou_fits) if ou_fits else None  # NEW
    results: List[GreedyResult] = []

    for i in range(num_runs):
        row = sample_commercial_row(df)
        if ou_engine is not None:
            ou_engine.reset()   # fresh corridor states for this episode

        res = greedy_policy(
            institutions=institutions,
            transfer_graph=transfer_graph,
            reachability_graph=reachability_graph,
            row=row,
            source_bank=source_bank,
            source_currency=source_currency,
            dest_bank=dest_bank,
            dest_currency=dest_currency,
            initial_amount=initial_amount,
            max_hops=max_hops,
            num_samples_per_candidate=num_samples_per_candidate,
            ou_engine=ou_engine,        # NEW
            tier_table=tier_table,      # NEW
        )
        results.append(res)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_runs} simulations...")
    return results


def summarize_results(results: List[GreedyResult]) -> Dict[str, float]:
    """Compute summary statistics from simulation results."""
    final_amounts = np.array([r.final_amount for r in results], dtype=float)
    ratios = np.array([r.ratio_vs_commercial for r in results], dtype=float)
    successes = np.array([r.success for r in results], dtype=bool)
    path_lengths = np.array([len(r.path) for r in results], dtype=int)

    # Only consider successful runs for ratio stats
    success_ratios = ratios[successes]
    success_amounts = final_amounts[successes]

    summary = {
        "num_runs": len(results),
        "success_rate": float(np.mean(successes)),
        "mean_final_amount": float(np.mean(success_amounts)) if len(success_amounts) > 0 else 0.0,
        "std_final_amount": float(np.std(success_amounts)) if len(success_amounts) > 0 else 0.0,
        "mean_ratio_vs_commercial": float(np.mean(success_ratios)) if len(success_ratios) > 0 else 0.0,
        "std_ratio_vs_commercial": float(np.std(success_ratios)) if len(success_ratios) > 0 else 0.0,
        "fraction_ratio_ge_1": float(np.mean(success_ratios >= 1.0)) if len(success_ratios) > 0 else 0.0,
        "mean_path_length": float(np.mean(path_lengths)),
        "max_path_length": int(np.max(path_lengths)) if len(path_lengths) > 0 else 0,
    }
    return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_network_graph(
    institutions: Dict[str, Institution],
    transfer_graph: Dict[str, List[Tuple[str, str]]],
    filename: str = "network_all_nodes.png",
    highlight_path: Optional[List[StepRecord]] = None,
    source_node: Optional[Tuple[str, str]] = None,
    dest_node: Optional[Tuple[str, str]] = None,
):
    """
    Draw the payment network graph with (bank, currency) nodes.
    
    Args:
        institutions: Dict of institution objects
        transfer_graph: Transfer relationships by currency
        filename: Output filename
        highlight_path: Optional path to highlight
        source_node: Optional source (bank, currency) to highlight
        dest_node: Optional destination (bank, currency) to highlight
    """
    G = nx.DiGraph()

    # Color mapping for currencies
    currency_colors = {
        "USD": "#2ecc71",  # Green
        "EUR": "#3498db",  # Blue
        "BRL": "#f1c40f",  # Yellow
        "PYG": "#e74c3c",  # Red
    }

    # Add nodes
    node_colors = []
    node_labels = {}
    for inst_name, inst in institutions.items():
        for cur in inst.account_currencies:
            node = (inst_name, cur)
            G.add_node(node)
            node_labels[node] = f"{inst_name}\n({cur})"

    # Add transfer edges
    for currency, transfers in transfer_graph.items():
        for (from_bank, to_bank) in transfers:
            from_node = (from_bank, currency)
            to_node = (to_bank, currency)
            if from_node in G.nodes and to_node in G.nodes:
                G.add_edge(from_node, to_node, edge_type="transfer", currency=currency)

    # Add FX edges
    for inst_name, inst in institutions.items():
        for (c_from, c_to) in inst.fx_pairs:
            from_node = (inst_name, c_from)
            to_node = (inst_name, c_to)
            if from_node in G.nodes and to_node in G.nodes:
                G.add_edge(from_node, to_node, edge_type="fx", currency=f"{c_from}→{c_to}")

    # Build path edges set for highlighting
    path_edges = set()
    if highlight_path:
        for step in highlight_path:
            if step.step_type == "transfer":
                path_edges.add(((step.inst_from, step.currency_from), (step.inst_to, step.currency_to)))
            else:  # fx
                path_edges.add(((step.inst_from, step.currency_from), (step.inst_to, step.currency_to)))

    # Layout
    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Assign node colors
    for node in G.nodes:
        cur = node[1]
        if node == source_node:
            node_colors.append("#9b59b6")  # Purple for source
        elif node == dest_node:
            node_colors.append("#e67e22")  # Orange for dest
        else:
            node_colors.append(currency_colors.get(cur, "#95a5a6"))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.9)

    # Draw edges by type
    transfer_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "transfer"]
    fx_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "fx"]

    # Regular edges
    regular_transfer = [e for e in transfer_edges if e not in path_edges]
    regular_fx = [e for e in fx_edges if e not in path_edges]

    nx.draw_networkx_edges(G, pos, edgelist=regular_transfer, edge_color="#7f8c8d",
                           style="solid", alpha=0.5, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=regular_fx, edge_color="#9b59b6",
                           style="dashed", alpha=0.5, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")

    # Highlighted path edges
    if highlight_path:
        path_transfer = [e for e in transfer_edges if e in path_edges]
        path_fx = [e for e in fx_edges if e in path_edges]

        nx.draw_networkx_edges(G, pos, edgelist=path_transfer, edge_color="#e74c3c",
                               style="solid", width=3, alpha=0.9, arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edges(G, pos, edgelist=path_fx, edge_color="#e74c3c",
                               style="dashed", width=3, alpha=0.9, arrows=True, arrowsize=20,
                               connectionstyle="arc3,rad=0.1")

    # Node labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=7, font_weight="bold")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#7f8c8d", linewidth=2, label="Transfer (solid)"),
        plt.Line2D([0], [0], color="#9b59b6", linewidth=2, linestyle="--", label="FX (dashed)"),
        plt.scatter([], [], c="#2ecc71", s=100, label="USD"),
        plt.scatter([], [], c="#3498db", s=100, label="EUR"),
        plt.scatter([], [], c="#f1c40f", s=100, label="BRL"),
        plt.scatter([], [], c="#e74c3c", s=100, label="PYG"),
    ]
    if source_node:
        legend_elements.append(plt.scatter([], [], c="#9b59b6", s=100, marker="s", label="Source"))
    if dest_node:
        legend_elements.append(plt.scatter([], [], c="#e67e22", s=100, marker="s", label="Destination"))
    if highlight_path:
        legend_elements.append(plt.Line2D([0], [0], color="#e74c3c", linewidth=3, label="Selected Path"))

    plt.legend(handles=legend_elements, loc="upper left", fontsize=10)
    plt.title("Cross-Border Payment Network\n(Bank, Currency) Nodes", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved network graph to {filename}")


def plot_statistics(
    results: List[GreedyResult],
    dest_currency: str,
    base_filename: str = "stats"
):
    """Generate statistical plots from simulation results."""
    successful = [r for r in results if r.success]

    if not successful:
        print("No successful runs to plot statistics for.")
        return

    final_amounts = [r.final_amount for r in successful]
    ratios = [r.ratio_vs_commercial for r in successful]
    path_lengths = [len(r.path) for r in successful]

    # Count bank and edge usage
    bank_usage = Counter()
    fx_edge_usage = Counter()
    transfer_edge_usage = Counter()

    for r in successful:
        for step in r.path:
            if step.step_type == "transfer":
                bank_usage[step.inst_from] += 1
                bank_usage[step.inst_to] += 1
                transfer_edge_usage[(step.inst_from, step.inst_to, step.currency_from)] += 1
            else:  # fx
                bank_usage[step.inst_from] += 1
                fx_edge_usage[(step.inst_from, step.currency_from, step.currency_to)] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Final amount histogram
    ax1 = axes[0, 0]
    ax1.hist(final_amounts, bins=40, edgecolor="black", alpha=0.7, color="#3498db")
    ax1.axvline(np.mean(final_amounts), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(final_amounts):,.0f}")
    ax1.set_xlabel(f"Final Amount ({dest_currency})", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of Final Amounts", fontsize=12, fontweight="bold")
    ax1.legend()

    # 2. Fairness ratio histogram
    ax2 = axes[0, 1]
    ax2.hist(ratios, bins=40, edgecolor="black", alpha=0.7, color="#2ecc71")
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Ratio = 1.0 (fair)")
    ax2.axvline(np.mean(ratios), color="blue", linestyle=":", linewidth=2,
                label=f"Mean: {np.mean(ratios):.4f}")
    ax2.set_xlabel("Fairness Ratio (vs Commercial)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Distribution of Fairness Ratio", fontsize=12, fontweight="bold")
    ax2.legend()

    # 3. Path length distribution
    ax3 = axes[1, 0]
    path_counts = Counter(path_lengths)
    lengths = sorted(path_counts.keys())
    counts = [path_counts[l] for l in lengths]
    ax3.bar(lengths, counts, edgecolor="black", alpha=0.7, color="#f39c12")
    ax3.set_xlabel("Path Length (number of steps)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Distribution of Path Lengths", fontsize=12, fontweight="bold")
    ax3.set_xticks(lengths)

    # 4. Top banks usage
    ax4 = axes[1, 1]
    top_banks = bank_usage.most_common(10)
    if top_banks:
        banks, counts = zip(*top_banks)
        y_pos = range(len(banks))
        ax4.barh(y_pos, counts, edgecolor="black", alpha=0.7, color="#9b59b6")
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(banks, fontsize=9)
        ax4.set_xlabel("Usage Count", fontsize=11)
        ax4.set_title("Most Used Banks in Routes", fontsize=12, fontweight="bold")
        ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{base_filename}_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved statistical plots to {base_filename}_distributions.png")

    # Additional plot: FX edges usage
    if fx_edge_usage:
        plt.figure(figsize=(12, 6))
        top_fx = fx_edge_usage.most_common(15)
        if top_fx:
            labels = [f"{e[0]}\n{e[1]}→{e[2]}" for e, _ in top_fx]
            counts = [c for _, c in top_fx]
            y_pos = range(len(labels))
            plt.barh(y_pos, counts, edgecolor="black", alpha=0.7, color="#e74c3c")
            plt.yticks(y_pos, labels, fontsize=8)
            plt.xlabel("Usage Count", fontsize=11)
            plt.title("Most Used FX Conversions", fontsize=12, fontweight="bold")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{base_filename}_fx_usage.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved FX usage plot to {base_filename}_fx_usage.png")


# ============================================================================
# USER INPUT AND VALIDATION
# ============================================================================

def get_user_directive(institutions: Dict[str, Institution]) -> Tuple[str, str, str, str, float]:
    """
    Prompt user for routing directive with validation.
    Returns (source_bank, source_currency, dest_bank, dest_currency, amount).
    """
    inst_names = sorted(institutions.keys())

    print("\n" + "=" * 60)
    print("CROSS-BORDER PAYMENT ROUTING SIMULATOR")
    print("=" * 60)

    print("\nAvailable Institutions:")
    for i, name in enumerate(inst_names, 1):
        inst = institutions[name]
        currencies = ", ".join(inst.account_currencies)
        fx_str = ", ".join([f"{a}→{b}" for a, b in inst.fx_pairs[:3]])
        if len(inst.fx_pairs) > 3:
            fx_str += "..."
        print(f"  {i:2d}. {name:20s} | Currencies: {currencies:15s} | FX: {fx_str}")

    print("\n" + "-" * 60)

    # Source bank
    while True:
        source_bank = input("\nEnter SOURCE bank name (or number): ").strip()
        if source_bank.isdigit():
            idx = int(source_bank) - 1
            if 0 <= idx < len(inst_names):
                source_bank = inst_names[idx]
                break
        elif source_bank in institutions:
            break
        print(f"Invalid bank. Choose from: {', '.join(inst_names)}")

    # Source currency
    source_currencies = institutions[source_bank].account_currencies
    while True:
        print(f"Available currencies at {source_bank}: {', '.join(source_currencies)}")
        source_currency = input("Enter SOURCE currency: ").strip().upper()
        if source_currency in source_currencies:
            break
        print(f"Invalid currency. Choose from: {', '.join(source_currencies)}")

    # Destination bank
    while True:
        dest_bank = input("\nEnter DESTINATION bank name (or number): ").strip()
        if dest_bank.isdigit():
            idx = int(dest_bank) - 1
            if 0 <= idx < len(inst_names):
                dest_bank = inst_names[idx]
                break
        elif dest_bank in institutions:
            break
        print(f"Invalid bank. Choose from: {', '.join(inst_names)}")

    # Destination currency
    dest_currencies = institutions[dest_bank].account_currencies
    while True:
        print(f"Available currencies at {dest_bank}: {', '.join(dest_currencies)}")
        dest_currency = input("Enter DESTINATION currency: ").strip().upper()
        if dest_currency in dest_currencies:
            break
        print(f"Invalid currency. Choose from: {', '.join(dest_currencies)}")

    # Amount
    while True:
        try:
            amount_str = input(f"\nEnter AMOUNT to send ({source_currency}): ").strip()
            amount = float(amount_str.replace(",", ""))
            if amount > 0:
                break
            print("Amount must be positive.")
        except ValueError:
            print("Invalid amount. Please enter a number.")

    print("\n" + "-" * 60)
    print(f"Directive: Send {amount:,.2f} {source_currency}")
    print(f"  From: {source_bank}")
    print(f"  To:   {dest_bank} ({dest_currency})")
    print("-" * 60)

    return source_bank, source_currency, dest_bank, dest_currency, amount


def check_route_feasibility(
    institutions: Dict[str, Institution],
    transfer_graph: Dict[str, List[Tuple[str, str]]],
    source_bank: str,
    source_currency: str,
    dest_bank: str,
    dest_currency: str,
) -> bool:
    """Check if a route is theoretically possible."""
    G = build_reachability_graph(institutions, transfer_graph)
    source_node = (source_bank, source_currency)
    dest_node = (dest_bank, dest_currency)

    if source_node not in G:
        print(f"ERROR: Source node {source_node} not in network!")
        return False
    if dest_node not in G:
        print(f"ERROR: Destination node {dest_node} not in network!")
        return False

    if not can_reach(G, source_node, dest_node):
        print(f"ERROR: No path exists from {source_node} to {dest_node}!")
        print("The requested route is not feasible with the current network configuration.")
        return False

    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Configuration
    FX_DATA_PATH = "fx_changes.csv"
    NUM_RUNS = 500
    MAX_HOPS = 8
    NUM_SAMPLES_PER_CANDIDATE = 20

    # Load FX data
    print("Loading FX data...")
    try:
        df = load_fx_data(FX_DATA_PATH)
        print(f"Loaded {len(df)} rows of FX data.")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"ERROR: FX data file '{FX_DATA_PATH}' not found!")
        print("Please ensure the CSV file exists in the current directory.")
        sys.exit(1)

    # === Phase 1: MLE on corridor volatility ====================================
    print("\nFitting Gaussian and Log-Normal to corridor volatility (MLE)...")
    fits = fit_all_corridors(df)
    print(f"  Fit {len(fits)} corridors.")
    fit_summary = summarize_fits(fits)

    # Print compact summary
    cols = ["from", "to", "n", "gauss_sigma_bps",
            "lognorm_SD[X]_bps", "delta_logL", "winner"]
    print("\nMLE corridor fit summary:")
    print(fit_summary[cols].to_string(index=False))
    print(f"\n  Winners: {dict(fit_summary['winner'].value_counts())}")

    # Diagnostic plot for the writeup
    plot_corridor_fits(
        df,
        pairs_to_plot=[("USD", "BRL"), ("USD", "PYG"), ("EUR", "BRL")],
        filename="mle_corridor_fits.png",
    )

    # Build institutions and apply MLE-driven sigmas
    print("\nInitializing payment network...")
    institutions = get_institutions()
    transfer_graph = get_transfer_graph()
    institutions = apply_mle_estimates(institutions, fits, verbose=True)
    print(f"Created {len(institutions)} institutions with MLE-driven spreads.")
    # ============================================================================


    # === Phase 2: OU process on corridor log-VOLATILITY =========================
    print("\nFitting OU process on corridor log-volatility (MLE via AR(1) OLS)...")
    ou_fits = fit_ou_all_corridors(df, dt=1.0, mode="log_volatility")  # CHANGED
    print(f"  Fit OU on {len(ou_fits)} corridors.")

    ou_summary = summarize_ou_fits(ou_fits)
    print("\nOU fit summary (log-volatility):")
    print(ou_summary[["from", "to", "n", "theta", "stat_std",
                      "a_raw", "clipped", "half_life_days"]]   # CHANGED columns
          .to_string(index=False))

    plot_ou_diagnostics(
        df, ou_fits,
        pairs_to_plot=[("USD", "BRL"), ("USD", "PYG"), ("BRL", "EUR")],
        filename="ou_diagnostics.png",
    )
    # ============================================================================

    # Draw the full network graph
    print("\nGenerating network visualization...")
    draw_network_graph(institutions, transfer_graph, filename="network_all_nodes.png")

    # Get user directive
    source_bank, source_currency, dest_bank, dest_currency, amount = get_user_directive(institutions)

    # Check feasibility
    if not check_route_feasibility(institutions, transfer_graph, source_bank, source_currency,
                                   dest_bank, dest_currency):
        sys.exit(1)

    # Run simulations
    print(f"\nRunning {NUM_RUNS} simulations...")
    results = run_greedy_experiments(
        df=df,
        institutions=institutions,
        transfer_graph=transfer_graph,
        source_bank=source_bank,
        source_currency=source_currency,
        dest_bank=dest_bank,
        dest_currency=dest_currency,
        initial_amount=amount,
        num_runs=NUM_RUNS,
        max_hops=MAX_HOPS,
        num_samples_per_candidate=NUM_SAMPLES_PER_CANDIDATE,
        ou_fits=ou_fits,
        tier_table=INSTITUTION_TIER,
    )

    # Summarize results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    summary = summarize_results(results)
    print(f"\nDirective: {amount:,.2f} {source_currency} from {source_bank}")
    print(f"       to: {dest_currency} at {dest_bank}")
    print()
    print(f"  Success Rate:              {summary['success_rate'] * 100:.1f}%")
    print(f"  Mean Final Amount:         {summary['mean_final_amount']:,.2f} {dest_currency}")
    print(f"  Std Final Amount:          {summary['std_final_amount']:,.2f} {dest_currency}")
    print(f"  Mean Fairness Ratio:       {summary['mean_ratio_vs_commercial']:.4f}")
    print(f"  Std Fairness Ratio:        {summary['std_ratio_vs_commercial']:.4f}")
    print(f"  Fraction with Ratio >= 1:  {summary['fraction_ratio_ge_1'] * 100:.1f}%")
    print(f"  Mean Path Length:          {summary['mean_path_length']:.2f} steps")
    print(f"  Max Path Length:           {summary['max_path_length']} steps")

    # Find example path (median fairness ratio among successful runs)
    successful_results = [r for r in results if r.success]
    if successful_results:
        # Sort by fairness ratio and pick median
        sorted_results = sorted(successful_results, key=lambda x: x.ratio_vs_commercial)
        median_idx = len(sorted_results) // 2
        example_result = sorted_results[median_idx]

        print("\n" + "-" * 60)
        print("EXAMPLE PATH (median fairness ratio):")
        print("-" * 60)
        print(f"Commercial Benchmark: {example_result.commercial_benchmark:,.2f} {dest_currency}")
        print(f"Actual Received:      {example_result.final_amount:,.2f} {dest_currency}")
        print(f"Fairness Ratio:       {example_result.ratio_vs_commercial:.4f}")
        print()

        state_amount = example_result.initial_amount
        print(f"START: ({example_result.source_bank}, {example_result.source_currency}) "
              f"Amount: {state_amount:,.2f}")

        for i, step in enumerate(example_result.path, 1):
            if step.step_type == "transfer":
                print(f"  Step {i}: TRANSFER {step.inst_from} -> {step.inst_to} "
                      f"[{step.currency_from}]")
                print(f"          Amount: {step.amount_before:,.2f} -> {step.amount_after:,.2f} "
                      f"(fee: {step.fee:,.2f})")
            else:
                print(f"  Step {i}: FX at {step.inst_from}: "
                      f"{step.currency_from} -> {step.currency_to}")
                print(f"          Amount: {step.amount_before:,.2f} -> {step.amount_after:,.2f} "
                      f"(rate: {step.rate_eff:.6f}, fee: {step.fee:,.2f})")

        print(f"END: ({example_result.dest_bank}, {example_result.dest_currency}) "
              f"Amount: {example_result.final_amount:,.2f}")

        # Draw path-highlighted network
        print("\nGenerating path visualization...")
        draw_network_graph(
            institutions,
            transfer_graph,
            filename="network_example_path.png",
            highlight_path=example_result.path,
            source_node=(example_result.source_bank, example_result.source_currency),
            dest_node=(example_result.dest_bank, example_result.dest_currency),
        )
    else:
        print("\nNo successful routes found!")

    # Generate statistical plots
    print("\nGenerating statistical plots...")
    plot_statistics(results, dest_currency, base_filename="stats")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - network_all_nodes.png (full network graph)")
    if successful_results:
        print("  - network_example_path.png (example path highlighted)")
    print("  - stats_distributions.png (statistical distributions)")
    print("  - stats_fx_usage.png (FX conversion usage)")