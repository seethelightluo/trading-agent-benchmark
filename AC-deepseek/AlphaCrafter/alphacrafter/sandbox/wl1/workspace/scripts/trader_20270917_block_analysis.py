"""Trader analysis for block 2027-09-17 -> 2027-10-01.

1) Replicate the 09-17 proposal (data sliced to <= 09-16, i.e. last completed
   day before the decision) to reconstruct the proposed target and estimate
   why the execution gate skipped migration.
2) Per-asset block return / contribution using close 09-16 -> close 09-30.
"""
import json
import numpy as np
import pandas as pd
import strategy as st

CUR = "2027-09-17"
LAST_COMPLETED = "2027-09-16"
BLOCK_END_LAST = "2027-09-30"

# 09-03 executed target weights (cost basis, from memory.txt) - actual holdings
T0 = {
    "BTC": 0.114, "ETH": 0.108, "NDX": 0.101, "WTI": 0.094, "SOX": 0.087,
    "000688.SH": 0.074, "CN10Y": 0.067, "HSI": 0.064, "XAU": 0.060,
    "000300.SH": 0.057, "SPX": 0.050, "N225": 0.042, "COPPER": 0.035,
    "SX5E": 0.028, "US10Y": 0.021,
}

assets = list(T0.keys())
frames = st._fetch(assets)

# ---- 1) replicate 09-17 proposal on sliced data ----
frames_sliced = {}
for a, df in frames.items():
    if df is None:
        frames_sliced[a] = None
        continue
    frames_sliced[a] = df[df.index <= pd.Timestamp(LAST_COMPLETED)].copy()

scores, used = st._scores(frames_sliced, assets, CUR)
print("factors used:", used)
regime = st._regime(frames_sliced, assets)
print("regime:", regime)
w_prop = st._weights(scores, assets, regime)
w_prop = st._ma_guard(w_prop, frames_sliced, assets, CUR)
f = st._forecasts(scores, assets)

print("\nproposed target (09-17):")
order = sorted(assets, key=lambda a: -w_prop[a])
for a in order:
    print(f"  {a:>8} {w_prop[a]*100:6.2f}%")

# turnover vs current holdings (valued at 09-16 close)
turn = 0.0
for a in assets:
    df = frames_sliced.get(a)
    if df is None or len(df) == 0:
        continue
    px = float(df["close"].iloc[-1])
    cur_w = T0[a]
    turn += abs(w_prop[a] - cur_w)
turn *= 0.5  # one-way migrated notional
gross_edge = sum(f[a] * w_prop[a] for a in assets) * 10  # 10-day horizon edge
print(f"\nregime={regime} one-way turnover vs 09-03 holdings: {turn*100:.2f}%")
print(f"10d gross edge of proposed target: {gross_edge*100:.3f}%")
print(f"gate threshold (turnover*3bp): {turn*0.0003*100:.4f}%")
print("gate would execute:", gross_edge > turn * 0.0003)

# ---- 2) block returns 09-16 close -> 09-30 close ----
print("\nblock per-asset returns (09-16 -> 09-30 close):")
total_contrib = 0.0
for a in assets:
    df = frames.get(a)
    if df is None or len(df) == 0:
        print(f"  {a}: no data")
        continue
    sub = df[df.index <= pd.Timestamp(BLOCK_END_LAST)]
    p0 = float(sub[sub.index <= pd.Timestamp(LAST_COMPLETED)]["close"].iloc[-1])
    p1 = float(sub["close"].iloc[-1])
    r = p1 / p0 - 1.0
    contrib = T0[a] * r
    total_contrib += contrib
    print(f"  {a:>8} ret={r*100:7.2f}%  w0={T0[a]*100:5.2f}%  contrib={contrib*100:6.3f}%")
print(f"sum contributions (approx): {total_contrib*100:.3f}%")
