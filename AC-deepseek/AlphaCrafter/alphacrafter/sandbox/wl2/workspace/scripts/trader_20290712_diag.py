"""Trader read-only diagnostic for cycle 57 (2029-07-12 decision).

Computes what build_target would produce for the next block start using data
visible at the decision date. Does NOT mutate account/date state.
"""
import json
import sys
sys.path.insert(0, ".")
from pathlib import Path

from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
import strategy as S

acct = get_account_dict()
assets = list(acct.get("watch_list", []))
print("n assets:", len(assets))
print("account: net_assets", acct.get("net_assets"), "cash", acct.get("available_cash"),
      "gross_pos_rate", acct.get("gross_position_rate"))
for p in acct.get("positions", []):
    print("  pos", p["symbol"], round(p.get("quantity", 0), 4), "mv", round(p.get("market_value", 0), 2))

# ensemble weights actually loaded by strategy
ens = S._load_ensemble()
print("\nloaded ensemble (%d):" % len(ens))
for f in ens:
    print("  ", f["factor_id"], "w", round(f["weight"], 4), "dir", f["direction"])

date_state = json.loads((S.BASE.parent / "persistent" / "date.json").read_text())
print("\ncurrent_date:", date_state.get("current_date"))
td = date_state.get("trading_days", [])
print("last 3 trading days:", td[-3:])
print("visible_through:", date_state.get("visible_through"))

# 20d returns & close levels
print("\n20d returns & levels (visible at decision):")
closes = S._closes(assets)
r20 = {}
for a in assets:
    c = closes.get(a)
    if c is not None and len(c) >= 21:
        r20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0)
        print(f"  {a:12s} close={c.iloc[-1]:12.4f}  20d={r20[a]*100:7.2f}%")

risk, vix, m20, disp = S._regime(closes, assets)
print(f"\nregime: risk={risk:.4f} vix={vix:.2f} m20={m20*100:.3f}% disp={disp*100:.3f}%")

# live factor snapshot
live = S._live_factors(assets)
print("\nlive factors finite counts:")
for fid, vals in live.items():
    print(f"  {fid:24s} finite={sum(1 for v in vals if v==v)}/15")

# build target as the strategy would
cur_w = S._current_weights(acct, assets)
built = S.build_target(assets, date_state, ens, current_weights=cur_w)
if built is None:
    print("\nbuild_target returned None")
else:
    w, fc, used, meta = built
    print("\nused factors:", used)
    print("meta: risk", round(meta["risk"], 4), "vix", round(meta["vix"], 2),
          "n_factors", meta["n_factors"])
    print("cap_map:", {k: v for k, v in meta["cap_map"].items()})
    print("\nproposed target vs current:")
    tot = 0
    for a in sorted(assets, key=lambda x: -w[x]):
        print(f"  {a:12s} cur={cur_w.get(a,0)*100:6.2f}%  tgt={w[a]*100:6.2f}%  fcast={fc[a]*100:+6.2f}%  z={meta['z'][a]:+5.2f}  r20={r20.get(a,0)*100:+6.2f}%")
        tot += w[a]
    print("sum weights:", round(tot, 6))
    turn = sum(abs(w[a] - cur_w.get(a, 0)) for a in assets)
    print("one-way turnover vs current book:", round(turn * 100, 2), "%")
