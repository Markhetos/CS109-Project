"""
sim_utils.py — Output folder management and fit caching.

Two quality-of-life improvements:
  (1) Per-simulation output folder so every run's figures land in their own
      directory instead of clobbering the project root.
  (2) Pickle cache for MLE and OU fits so re-running for a demo video doesn't
      repeat 10-20 seconds of fitting that hasn't changed.

Usage in routing_sim_v2.py
---------------------------
from sim_utils import make_output_dir, save_fits, load_fits

# At startup:
cached = load_fits(CACHE_PATH)
if cached:
    mle_fits, ou_fits = cached
    print("Loaded cached fits.")
else:
    mle_fits = fit_mle(...)
    ou_fits  = fit_ou(...)
    save_fits(CACHE_PATH, mle_fits, ou_fits)

# When user enters a directive:
out_dir = make_output_dir("graphs", source_bank, source_currency,
                           dest_bank, dest_currency)
# Then prefix every plt.savefig call with out_dir, e.g.:
plt.savefig(os.path.join(out_dir, "efficient_frontier.png"), ...)
"""

import os
import pickle
from typing import Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Per-simulation output directory
# ---------------------------------------------------------------------------

def make_output_dir(
    base: str,
    source_bank: str,
    source_currency: str,
    dest_bank: str,
    dest_currency: str,
) -> str:
    """Create and return a per-directive output directory.

    Pattern: {base}/{source_bank}_{source_currency}-{dest_bank}_{dest_currency}/

    Example: graphs/Itau_BR_EUR-Continental_PY_PYG/
    """
    name = f"{source_bank}_{source_currency}-{dest_bank}_{dest_currency}"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Fit cache (MLE + OU fits)
# ---------------------------------------------------------------------------

CACHE_PATH = ".fits_cache.pkl"   # lives in project root, git-ignored


def save_fits(path: str, mle_fits: Any, ou_fits: Any) -> None:
    """Persist MLE and OU fits to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump({"mle_fits": mle_fits, "ou_fits": ou_fits}, f)
    print(f"Cached fits → {path}")


def load_fits(path: str) -> Optional[Tuple[Any, Any]]:
    """Load cached fits if the file exists. Returns (mle_fits, ou_fits) or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cached fits from {path}  (delete to re-fit)")
        return data["mle_fits"], data["ou_fits"]
    except Exception as e:
        print(f"Cache load failed ({e}), re-fitting.")
        return None


def clear_cache(path: str = CACHE_PATH) -> None:
    """Delete the cache so the next run re-fits from scratch."""
    if os.path.exists(path):
        os.remove(path)
        print(f"Cache cleared: {path}")
    else:
        print(f"No cache found at {path}")