"""Trader 2027-10-07: inspect current regime metrics at block start."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
acct = get_account_dict()
assets = acct["watch_list"]
print("watch_list:", assets)
print("net_assets:", acct.get("net_assets"), "cash:", acct.get("available_cash"),
      "gross_pos_rate:", acct.get("gross_position_rate"))
print("positions:", [(p["symbol"], round(p.get("quantity", 0), 2)) for p in acct.get("positions", [])])

def get_df(sym, days=300):
    try:
        return get_index_daily_data(sym, days=days) if sym in OBS else get_stock_daily_data(sym, days=days)
    except Exception:
        return None

panel = None
for a in assets:
    df = get_df(a)
    if df is None or len(df) < 100:
        print("short data:", a, len(df) if df is not None else None)
        continue
    s = df["close"].astype(float)
    s.index = pd.to_datetime(df["date"])
    r = s.pct_change().rename(a)
    panel = r if panel is None else pd.concat([panel, r], axis=1, join="inner")
panel = panel.dropna()
print("panel rows:", len(panel), "last date:", panel.index[-1].date())

market = panel.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(panel.tail(20).std().mean())
vol_med = float(panel.tail(120).std().median(axis=0))
print(f"mkt20={mkt20:.5f} mdd20={mdd:.5f} vol20={vol20:.5f} vol_med={vol_med:.5f} ratio={vol20/vol_med:.3f}")
risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.015
print("risk_off:", risk_off, "risk_on:", risk_on,
      "-> def_floor:", 0.16 if risk_off else (0.11 if risk_on else 0.13),
      "spread:", 2.0 if risk_off else (3.0 if risk_on else 2.0))

# asset-level 20d vol and last close for context
print("\nasset 20d vol (annualized-ish, daily std):")
for a in assets:
    r = panel[a]
    print(f"  {a:10s} vol20={r.tail(20).std():.5f} last_ret20={r.tail(20).sum()*100:+.2f}%")
