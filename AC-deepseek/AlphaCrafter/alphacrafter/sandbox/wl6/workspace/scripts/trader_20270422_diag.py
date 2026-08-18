"""Trader diagnostic 2027-04-22: check VIX state, regime, and factor raws."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
assets = acct.get("watch_list", [])
print("date-implied account net_assets:", acct.get("net_assets"))
print("watch_list:", assets)

frames = {a: get_stock_daily_data(a, days=300) for a in assets}
closes = {a: f.close.astype(float) for a, f in frames.items() if f is not None and len(f) > 140}

# VIX level over last 70 days
vf = get_index_daily_data("VIX", days=120)
if vf is not None:
    vix = vf.close.astype(float)
    print("\nVIX last 70d: min=%.3f max=%.3f last=%.3f std=%.5f" % (
        vix.tail(70).min(), vix.tail(70).max(), vix.iloc[-1], vix.tail(70).std()))
    print("VIX last 5:", [round(x, 2) for x in vix.tail(5).tolist()])

# Regime: return-based 20d cross-asset drift
panel = pd.concat([closes[a].rename(a) for a in closes], axis=1, join="inner")
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
print("\nregime trend t-stat=%.2f -> %s" % (trend, regime))
print("20d mkt mean daily ret=%.4f%%" % (r20 * 100))

# 20d returns per asset
print("\n20d returns:")
for a in closes:
    r = closes[a].pct_change().tail(20).sum()
    print("  %-10s %+6.2f%%" % (a, r * 100))

# Factor raw values for the 3 active factors
def rank_series(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and pd.notna(v))
    out = {a: 0.5 for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out

vix_ret = vix.pct_change() if vf is not None else None
print("\nFactor raws:")
for a in closes:
    c = closes[a]
    ret = c.pct_change()
    bv = None
    if vix_ret is not None:
        z = pd.concat([ret.rename("a"), vix_ret.rename("v")], axis=1).dropna().tail(60)
        varv = float(z["v"].var()) if len(z) else 0.0
        if len(z) >= 30 and varv > 1e-14:
            bv = -float(z["a"].cov(z["v"]) / varv)
    lv = -float(ret.rolling(20, min_periods=10).std().iloc[-1])
    vov = float(ret.rolling(20).std().rolling(60).std().iloc[-1])
    print("  %-10s beta_vix_neg=%+7.3f low_vol20=%+8.5f volofvol=%+8.5f" % (a,
          bv if bv is not None else float("nan"), lv, vov))

# Current positions
print("\npositions:")
for p in acct.get("positions", []):
    print("  %-10s qty=%10.2f val=%12.2f pnl_rate=%+7.2f%%" % (
        p["symbol"], p.get("quantity", 0), p.get("market_value", 0),
        p.get("profit_loss_rate", 0) * 100))
