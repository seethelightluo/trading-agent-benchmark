"""Trader diagnostic: regime metrics as of block start 2027-11-04 (data through prev day)."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(symbol, days=300):
    try:
        if symbol in OBS_ONLY:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None

acct = get_account_dict()
assets = acct["watch_list"]
print("watch_list:", assets)
print("total_assets:", acct.get("total_assets"), "net:", acct.get("net_assets"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("cash:", acct.get("available_cash"))
print("n_positions:", len(acct.get("positions", [])))
print("pending orders:", len(acct.get("orders", [])))

frames = {a: get_df(a) for a in assets}
close = {a: frames[a]["close"].astype(float) for a in assets}
ret = {a: close[a].pct_change() for a in assets}
panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
print("\npanel rows:", len(panel), "last date:", panel.index[-1])

market = panel.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd20 = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
mkt10 = float(market.tail(10).mean())
vol20 = float(panel.tail(20).std().mean())
vol_med = float(panel.tail(120).std().median(axis=0))
print(f"\nmkt20={mkt20*100:.3f}%  mkt10={mkt10*100:.3f}%  mdd20={mdd20*100:.3f}%")
print(f"vol20={vol20*100:.3f}%  vol_med={vol_med*100:.3f}%  ratio={vol20/max(vol_med,1e-6):.3f}")

risk_off = (mkt20 < 0.0 and mdd20 < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd20 > -0.015
print("risk_off:", risk_off, " risk_on:", risk_on,
      " -> def_floor:", 0.16 if risk_off else (0.11 if risk_on else 0.13),
      " spread:", 2.0 if risk_off else (3.0 if risk_on else 2.0))

# per-asset recent returns
print("\nasset 5d/10d/20d returns (%):")
for a in assets:
    r = ret[a].dropna()
    if len(r) >= 20:
        print(f"  {a:10s} {r.tail(5).sum()*100:7.2f} {r.tail(10).sum()*100:7.2f} {r.tail(20).sum()*100:7.2f}")

# defensive assets recent
print("\ndefensive (XAU/US10Y/CN10Y) 20d rets:", {
    a: round(float(ret[a].dropna().tail(20).sum())*100, 2) for a in ["XAU", "US10Y", "CN10Y"]})
