"""Trader read-only sanity check of build_target at 2034-02-16 block start.

Verifies: ensemble loads, strategy.py LIVE_FIDS matches ensemble factor IDs,
build_target returns a valid 15-asset target (sum=1, non-negative, finite),
and reports meta (risk/vix/m20) + one-way turnover vs current weights.
No account mutation: this script only reads data and computes.
"""
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import strategy as st

# 1. Ensemble consistency
ens = st._load_ensemble()
print("ensemble n =", len(ens))
for f in ens:
    print("  ", f["factor_id"], "w=", f["weight"], "dir=", f["direction"])
ens_ids = {f["factor_id"] for f in ens}
live_ids = set(st.LIVE_FIDS)
print("LIVE_FIDS == ensemble IDs:", live_ids == ens_ids)
if live_ids != ens_ids:
    print("  extra in LIVE_FIDS:", live_ids - ens_ids)
    print("  missing from LIVE_FIDS:", ens_ids - live_ids)

# 2. Date state
date_state = json.loads((BASE.parent / "persistent" / "date.json").read_text())
print("current_date:", date_state.get("current_date"),
      "| visible_through:", date_state.get("visible_through"))

# 3. Account & current weights
account = json.loads((BASE.parent / "persistent" / "account.json").read_text())
assets = list(account.get("watch_list", []))
print("n assets:", len(assets))
cur_w = st._current_weights(account, assets)
print("current weights sum:", round(sum(cur_w.values()), 6))
for a in assets:
    print("  ", a, round(cur_w[a], 4))

# 4. Build target (pure compute; no orders)
built = st.build_target(assets, date_state, ens, current_weights=cur_w)
if built is None:
    print("build_target returned None -- CHECK")
    sys.exit(1)
weights, forecast, used, meta = built
total = sum(weights.values())
finite = all(math.isfinite(weights[a]) for a in assets)
nonneg = all(weights[a] >= 0.0 for a in assets)
print("\nTARGET: sum=", round(total, 6), "finite=", finite, "nonneg=", nonneg)
print("used factors:", used)
print("meta: risk=", round(meta["risk"], 3), "vix=", round(meta["vix"], 1),
      "m20=", round(meta["m20"], 4), "disp=", round(meta["disp"], 4))
for a in assets:
    print("  ", a, "w=", round(weights[a], 4), "f10=", round(forecast[a], 4),
          "r20=", round(meta["r20"].get(a, 0.0), 4),
          "cap=", meta["cap_map"].get(a, st.CAP))
turn = sum(abs(weights[a] - cur_w[a]) for a in assets)
print("one-way turnover vs current:", round(turn, 4), "(MAX_TURNOVER", st.MAX_TURNOVER, ")")
