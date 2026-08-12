"""miner_3 probe (2030-01-14) - data availability & recent tape state."""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | elapsed {time.time()-t0:.1f}s")
print("last completed trading day:", closes.index.max().date())

# coverage: non-null counts per asset
print("\n=== per-asset coverage (n obs / total dates) ===")
for a in closes.columns:
    n = closes[a].notna().sum()
    print(f"{a:10s} {n:5d} / {len(closes)}  {n/len(closes)*100:5.1f}%")

# recent 60d returns and vol
print("\n=== last 60d cumulative returns ===")
r60 = (1 + rets.tail(60)).prod() - 1
r20 = (1 + rets.tail(20)).prod() - 1
for a in closes.columns:
    print(f"{a:10s} r60={r60[a]*100:+7.2f}%  r20={r20[a]*100:+7.2f}%")

mkt = rets.mean(axis=1)
print("\nmkt 60d cum:", f"{(1+mkt.tail(60)).prod()-1:+.2%}", " mkt 20d cum:", f"{(1+mkt.tail(20)).prod()-1:+.2%}")

for m in MACRO:
    df = panels.get(m)
    if df is not None:
        c = df["close"].astype(float)
        print(f"{m}: last {c.iloc[-1]:.2f} | 20d ago {c.iloc[-21]:.2f} | 60d ago {c.iloc[-61]:.2f} | 250d ago {c.iloc[-251]:.2f}")
