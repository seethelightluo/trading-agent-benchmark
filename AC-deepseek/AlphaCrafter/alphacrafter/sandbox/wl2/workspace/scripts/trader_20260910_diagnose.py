"""Trader diagnostic for 2026-09-10 block start (cycle 5).

Compute the target strategy.py would produce with data visible through
2026-09-09, plus regime metrics, per-asset z-scores, weights, and stale-artifact
impact check.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strategy as S
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

date_state = json.loads(Path("../persistent/date.json").read_text())
current = date_state["current_date"]
visible = date_state["visible_through"]
td = date_state["trading_days"]
print(f"current_date={current} visible_through={visible}")

account = get_account_dict()
assets = list(account.get("watch_list", []))
print(f"n_assets={len(assets)}  NAV={account['net_assets']:.2f}")

ensemble = S._load_ensemble()
print(f"ensemble factors: {len(ensemble)}")
for f in ensemble:
    print(f"  {f['factor_id']} w={f['weight']:.4f} dir={f['direction']:+d}")

closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
print(f"\nREGIME risk={risk:.3f} vix={vix:.2f} m20={m20:.4f} disp20={disp:.4f}")

rets20, rets60 = {}, {}
for a in assets:
    c = closes.get(a)
    rets20[a] = (c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0 if (c is not None and len(c) >= 21) else float("nan")
    rets60[a] = (c.iloc[-1] / c.iloc[-61] - 1.0) * 100.0 if (c is not None and len(c) >= 61) else float("nan")
print("\n20d returns:")
for a in sorted(rets20, key=lambda x: rets20[x], reverse=True):
    print(f"  {a:8s} 20d {rets20[a]:+7.2f}%   60d {rets60[a]:+7.2f}%")

built = S.build_target(assets, date_state, ensemble)
if built is None:
    print("\nbuild_target returned None - cannot trade")
    sys.exit(1)
weights, forecast, used, meta = built
print(f"\nTARGET weights (sum={sum(weights.values()):.6f}):")
for a in sorted(weights, key=lambda x: weights[x], reverse=True):
    print(f"  {a:8s} w={weights[a]:.4f}  fcast={forecast[a]*100:+.2f}%  z={meta['z'][a]:+.2f}  r20={meta['r20'][a]*100:+.2f}%")
print(f"\ncap_map (trend cap): {meta['cap_map']}")
print(f"n_factors used: {meta['n_factors']}")

row_idx = td.index(visible) - td.index(S.ARTIFACT_START)
print(f"\nrow_idx for {visible} = {row_idx} (artifact last row {2398-1})")
for fac in ensemble:
    row = S._signal_row(fac["factor_id"], row_idx, len(assets))
    n_valid = sum(1 for x in row if x == x) if row else 0
    print(f"  {fac['factor_id']:24s} resolved={row is not None} valid={n_valid}/15")

# current holdings vs target migration
print("\nCurrent weights vs target:")
cur = {p["symbol"]: p["market_value"] / account["net_assets"] for p in account["positions"]}
one_way = 0.0
for a in assets:
    diff = abs(weights[a] - cur.get(a, 0.0))
    one_way += diff
    print(f"  {a:8s} cur={cur.get(a,0):.4f} tgt={weights[a]:.4f} delta={weights[a]-cur.get(a,0):+.4f}")
print(f"\nOne-way turnover approx: {one_way/2:.4f}")
