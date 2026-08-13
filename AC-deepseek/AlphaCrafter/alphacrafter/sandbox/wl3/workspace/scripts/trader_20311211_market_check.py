"""Trader pre-step market check: 2031-12-11 block start.

Prints recent returns for commodity/crypto/equity complexes and regime flags
to decide whether the planned cap re-tune (COMM_CAP/COPPER/ETH) should fire.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}

def get_df(sym, days=160):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=days)
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

close = {}
for a in ASSETS + list(OBS):
    df = get_df(a)
    if df is None or len(df) < 30:
        print(a, "NO DATA")
        continue
    s = df["close"].astype(float)
    s.index = pd.to_datetime(df["date"])
    close[a] = s

print("last obs date:", max(s.index.max() for s in close.values()).date())

def ret(a, n):
    s = close[a]
    if len(s) < n + 1:
        return np.nan
    return s.iloc[-1] / s.iloc[-1 - n] - 1.0

print("\n--- 10d / 21d / 60d returns (pct) ---")
for a in ASSETS:
    print(f"{a:10s} {ret(a,10)*100:8.2f} {ret(a,21)*100:8.2f} {ret(a,60)*100:8.2f}")

# regime flags (mirror strategy.py)
panel = pd.concat([close[a].pct_change().rename(a) for a in ASSETS], axis=1, join="inner").dropna()
lp = panel
market = lp.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(lp.tail(20).std().mean())
vol_med = float(lp.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.015
print(f"\nregime: mkt20={mkt20*100:.2f}% mdd20={mdd*100:.2f}% vol20={vol20*100:.2f}% "
      f"vol_med={vol_med*100:.2f}% risk_off={risk_off} risk_on={risk_on}")

vix = close["VIX"]
print(f"VIX last={vix.iloc[-1]:.1f}  21d chg={vix.iloc[-1]/vix.iloc[-22]-1:.2%}" if len(vix) >= 22 else "VIX short")
eq = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"]
eq21 = np.mean([close[a].iloc[-1] / close[a].iloc[-22] - 1.0 for a in eq])
print(f"live-eq 21d mean={eq21*100:.2f}%")

# commodity complex trajectory over last 5 blocks
print("\n--- commodity/crypto recent 10d windows (last 30 trading days) ---")
for a in ["WTI", "COPPER", "XAU", "ETH", "NDX"]:
    s = close[a]
    w = s.pct_change().iloc[-30:]
    print(f"{a:6s}", " ".join(f"{x*100:6.1f}" for x in w.iloc[::3].values))
