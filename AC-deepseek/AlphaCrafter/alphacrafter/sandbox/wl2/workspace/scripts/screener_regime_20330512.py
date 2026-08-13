"""Screener analysis for 2033-05-12 decision (data visible through 2033-05-11).
Computes market regime stats and candidate factor cross-sections on the
15-asset tradable universe. Analysis only - no backtest/step, no account writes.
"""
import pandas as pd
import numpy as np
import json, glob, os

BASE = "../persistent/stock_data"
IDX = "../persistent/index_data"
AS_OF = "2033-05-11"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(sym):
    p = f"{BASE}/{sym}.csv" if os.path.exists(f"{BASE}/{sym}.csv") else f"{IDX}/{sym}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= AS_OF].set_index("date").sort_index()
    return df

closes = {}
for a in ASSETS:
    try:
        df = load(a)
        closes[a] = df["close"].astype(float)
    except Exception as e:
        print("load err", a, e)

panel = pd.DataFrame(closes).dropna(how="all")
rets = panel.pct_change()
print("panel dates:", panel.index.min().date(), "->", panel.index.max().date(), "n=", len(panel))

# ---- regime ----
vix = load("VIX")["close"]
spx = closes["SPX"]
mkt = rets.mean(axis=1)  # equal-weight cross-asset
print("\n=== REGIME (through", AS_OF, ") ===")
print("VIX last:", round(float(vix.iloc[-1]), 2), "| 20d ago:", round(float(vix.iloc[-21]), 2) if len(vix) > 21 else None)
print("VIX 60d mean:", round(float(vix.tail(60).mean()), 2), "| 60d max:", round(float(vix.tail(60).max()), 2))
for w in (10, 20, 60, 120, 180):
    if len(panel) > w:
        r = panel.iloc[-1] / panel.iloc[-1 - w] - 1
        print(f"median {w}d return: {r.median()*100:+.2f}% | SPX {w}d: {r['SPX']*100:+.2f}%")
# trend strength: SPX MA slopes
for w in (20, 60, 120):
    ma = spx.rolling(w).mean()
    slope = (ma.iloc[-1] / ma.iloc[-1 - 10] - 1) if len(ma) > 10 else np.nan
    print(f"SPX MA{w} 10d slope: {slope*100:+.2f}% | price vs MA{w}: {spx.iloc[-1]/ma.iloc[-1]-1:+.2%}")
# avg pairwise corr 60d
c60 = rets.tail(60).corr()
mask = np.triu(np.ones(c60.shape, dtype=bool), 1)
print("avg pairwise corr 60d:", round(float(c60.values[mask].mean()), 3))
# realized vol
rv20 = rets.tail(20).std() * np.sqrt(252)
print("RV20 annualized median:", round(float(rv20.median()), 3), "| max:", round(float(rv20.max()), 3))
print("\n20d returns by asset:")
r20 = (panel.iloc[-1] / panel.iloc[-21] - 1).sort_values()
for a, v in r20.items():
    print(f"  {a:10s} {v*100:+7.2f}%")
print("\n60d returns by asset:")
r60 = (panel.iloc[-1] / panel.iloc[-61] - 1).sort_values()
for a, v in r60.items():
    print(f"  {a:10s} {v*100:+7.2f}%")
print("\n180d returns by asset:")
r180 = (panel.iloc[-1] / panel.iloc[-181] - 1).sort_values()
for a, v in r180.items():
    print(f"  {a:10s} {v*100:+7.2f}%")
