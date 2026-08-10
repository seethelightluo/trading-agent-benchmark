"""Trader 2026-08-13: validate the 9-factor ensemble target before live cycle."""
import json
import sys
sys.path.insert(0, '.')
from strategy import build_target, _load_ensemble, EMBEDDED

ENSEMBLE_PATH = 'factor_ensemble.json'
DATE_PATH = '../persistent/date.json'

assets = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

date_state = json.load(open(DATE_PATH))
print("current:", date_state["current_date"], "visible:", date_state.get("visible_through"))

ensemble = _load_ensemble()
print("ensemble factors:", len(ensemble))
for f in ensemble:
    print("  ", f["factor_id"], round(f["weight"], 4), "dir", f["direction"])

built = build_target(assets, date_state, ensemble)
assert built is not None, "build_target returned None"
weights, forecast, used, meta = built
total = sum(weights.values())
print("used factors:", used)
print("n_used:", len(used), "(cap 10 OK)" if len(used) <= 10 else "(EXCEEDS CAP)")
print("sum weights:", round(total, 8))
print("meta:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()})
print("--- weights ---")
for a in assets:
    print(f"  {a:12s} {weights[a]:.4f}  fwd={forecast[a]:+.3f}")
assert abs(total - 1.0) < 1e-6, "weights do not sum to 1"
assert all(w >= 0 and w == w for w in weights.values()), "negative or NaN weight"
assert set(weights.keys()) == set(assets), "asset set mismatch"
print("VALID: full-investment 15-asset target, sum=1, no cash sleeve.")
