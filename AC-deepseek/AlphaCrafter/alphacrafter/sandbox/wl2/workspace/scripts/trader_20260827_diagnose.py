"""Trader diagnostic for 2026-08-27 block start (cycle 4).

Compute the target that strategy.py would produce with data visible through
2026-08-26, plus regime metrics, per-asset z-scores and weights, and check the
COPPER concern flagged in the previous cycle.
"""
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strategy as S
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

# --- date state -------------------------------------------------------------
date_state = json.loads(Path("../persistent/date.json").read_text())
current = date_state["current_date"]
visible = date_state["visible_through"]
td = date_state["trading_days"]
print(f"current_date={current} visible_through={visible}")

account = __import__("alphacrafter.sim.utils", fromlist=["get_account_dict"]).get_account_dict()
assets = list(account.get("watch_list", []))
print(f"n_assets={len(assets)}")

# --- ensemble ---------------------------------------------------------------
ensemble = S._load_ensemble()
print(f"ensemble factors: {len(ensemble)}")
for f in ensemble:
    print(f"  {f['factor_id']} w={f['weight']:.4f} dir={f['direction']:+d}")

# --- regime -----------------------------------------------------------------
closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
print(f"\nREGIME risk={risk:.3f} vix={vix:.2f} m20={m20:.4f} disp20={disp:.4f}")

# 20d per-asset returns for context
rets20 = {}
for a in assets:
    c = closes.get(a)
    if c is not None and len(c) >= 21:
        rets20[a] = (c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0
    else:
        rets20[a] = float("nan")
for a in sorted(rets20, key=lambda x: rets20[x], reverse=True):
    print(f"  {a:8s} 20d ret {rets20[a]:+7.2f}%")

# --- target computation -----------------------------------------------------
built = S.build_target(assets, date_state, ensemble)
if built is None:
    print("\nbuild_target returned None - cannot trade")
    sys.exit(1)
weights, forecast, used, meta = built
print(f"\nTARGET weights (sum={sum(weights.values()):.6f}):")
for a in sorted(weights, key=lambda x: weights[x], reverse=True):
    print(f"  {a:8s} w={weights[a]:.4f}  fcast={forecast[a]*100:+.2f}%  z={meta['z'][a]:+.2f}")

# --- factor resolution check ------------------------------------------------
row_idx = td.index(visible) - td.index(S.ARTIFACT_START)
print(f"\nrow_idx for {visible} = {row_idx}")
for fac in ensemble:
    row = S._signal_row(fac["factor_id"], row_idx, len(assets))
    n_valid = sum(1 for x in row if x == x) if row else 0
    print(f"  {fac['factor_id']:24s} resolved={row is not None} valid={n_valid}/15")

# COPPER specifics
print("\nCOPPER check:")
c = closes.get("COPPER")
if c is not None and len(c) >= 70:
    last60 = c.iloc[-61:-1]
    print(f"  close {c.iloc[-1]:.1f}, 20d {rets20.get('COPPER', float('nan')):+.2f}%, "
          f"60d { (c.iloc[-1]/c.iloc[-61]-1)*100:+.2f}%")
    print(f"  z-score in target: {meta['z']['COPPER']:+.3f}, weight {weights['COPPER']:.4f}")
