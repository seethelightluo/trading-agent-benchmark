"""Trader pre-cycle inspection (2029-06-28): account state, regime, and
simulated next-block target from the live v6 strategy (no step/backtest calls)."""
import json
import math
import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

import strategy as S

acct = get_account_dict()
print("=== ACCOUNT ===")
print("net_assets:", acct.get("net_assets"), "total_assets:", acct.get("total_assets"),
      "cash:", acct.get("available_cash"), "gross_pos_rate:", acct.get("gross_position_rate"))
pos = {p["symbol"]: p for p in acct.get("positions", [])}
print("positions:", {k: round(v.get("quantity", 0), 2) for k, v in pos.items()})
print("orders:", acct.get("orders"))

assets = list(acct.get("watch_list", []))
print("watch_list:", assets)

# regime + price data
closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
print("\n=== REGIME ===")
print(f"risk={risk:.3f} vix={vix:.2f} m20(60d mean daily ret)={m20*100:.3f}% disp20={disp*100:.3f}%")

print("\n=== 20d / 60d RETURNS ===")
for a in assets:
    c = closes.get(a)
    if c is not None and len(c) >= 61:
        r20 = c.iloc[-1] / c.iloc[-21] - 1.0
        r60 = c.iloc[-1] / c.iloc[-61] - 1.0
        print(f"{a:12s} r20={r20*100:8.2f}%  r60={r60*100:8.2f}%")
    else:
        print(f"{a:12s} no data")

# simulate build_target with the current date state
date_state = json.load(open(S.DATE_PATH))
cur = date_state.get("current_date")
td = date_state.get("trading_days", [])
print("\ncurrent_date:", cur, "visible in td:", cur in td)
if cur in td:
    ds = {"trading_days": td, "visible_through": cur, "current_date": cur}
    cur_w = S._current_weights(acct, assets)
    built = S.build_target(assets, ds, S._load_ensemble(), current_weights=cur_w)
    if built is None:
        print("build_target -> None")
    else:
        w, fc, used, meta = built
        print("\n=== SIMULATED TARGET (visible through", cur, ") ===")
        print("used factors:", used)
        print("meta: risk=%.3f vix=%.1f m20=%.4f disp=%.4f lam=%.1f n_factors=%d" %
              (meta["risk"], meta["vix"], meta["m20"], meta["disp"], meta["lam"], meta["n_factors"]))
        print("cap_map:", meta["cap_map"])
        print("r20 map:", {a: round(v * 100, 2) for a, v in meta["r20"].items()})
        tot = sum(w.values())
        print("sum weights:", round(tot, 6))
        # turnover and gross edge
        turn = sum(abs(w[a] - cur_w.get(a, 0.0)) for a in assets)
        edge = sum((w[a] - cur_w.get(a, 0.0)) * fc[a] for a in assets)
        print(f"one-way turnover vs current: {turn*100:.2f}%  threshold(turn*3bp): {turn*3e-4*10000:.2f}bp")
        print(f"gross edge (signed): {edge*10000:.2f}bp  abs-edge: {sum(abs(w[a]-cur_w.get(a,0.0))*abs(fc[a]) for a in assets)*10000:.2f}bp")
        print("\nasset    target%  current%  delta%   forecast%")
        for a in sorted(assets, key=lambda x: -w[x]):
            print(f"{a:10s} {w[a]*100:7.2f} {cur_w.get(a,0)*100:8.2f} {(w[a]-cur_w.get(a,0))*100:+7.2f} {fc[a]*100:+8.2f}")
        print("\nz-scores:", {a: round(meta['z'][a], 2) for a in assets})
