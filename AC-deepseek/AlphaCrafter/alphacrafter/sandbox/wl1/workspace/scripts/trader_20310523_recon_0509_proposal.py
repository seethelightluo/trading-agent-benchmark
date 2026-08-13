"""Reconstruct the 2031-05-09 proposal target (decision data visible <= 05-08)
to explain why the execution gate skipped and give factor feedback."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import strategy as st

DECISION = "2031-05-09"
VISIBLE = "2031-05-08"  # decision sees data through previous completed day

wl = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
      'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

frames = {}
for a in wl:
    df = st.get_stock_daily_data(symbol=a, days=st.DATA_DAYS)
    if df is None or len(df) < st.MIN_ROWS:
        frames[a] = None
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[df.index <= pd.Timestamp(VISIBLE)]
    frames[a] = df

scores, used = st._scores(frames, wl, DECISION)
print("factors used:", used)
scores = st._de_rank_value_traps(scores, frames, wl, DECISION)
regime = st._regime(frames, wl)
print("regime:", regime)
below = st._below_ma(frames, wl)
print("below MA20:", sorted(below))
w = st._weights(scores, wl, regime)
w = st._composite_top2_cap(w, wl, scores)
w = st._composite_ma_guard(w, frames, wl)
w = st._ma_guard(w, frames, wl, DECISION)
for _ in range(6):
    w = st._commod_cap(w, wl)
    w = st._crypto_cap(w, wl)
    w = st._china_cap(w, wl)
f = st._forecasts(scores, wl)

# current holdings weights (04-25 exec, qty fixed) at 05-08 close
qty = {'000300.SH': 15.5006, 'SPX': 6.5828, 'HSI': 5.2905, 'N225': 1.6398,
       'SX5E': 12.0998, '000688.SH': 84.4236, 'SOX': 18.8982, 'NDX': 2.403,
       'XAU': 10.6973, 'COPPER': 16856.9467, 'WTI': 384.4859, 'BTC': 0.8575,
       'ETH': 43.3749, 'US10Y': 18869.9567, 'CN10Y': 37398.8701}
mv = {}
for a in wl:
    df = frames[a]
    if df is None or len(df) == 0:
        mv[a] = 0.0
        continue
    mv[a] = qty[a] * float(df["close"].iloc[-1])
tmv = sum(mv.values())
w_old = {a: mv[a] / tmv for a in wl}

order = sorted(wl, key=lambda a: -w[a])
print("\nproposed target @0509 (recon):")
for a in order:
    print(f"  {a:10s} new {w[a]*100:6.2f}%  old {w_old[a]*100:6.2f}%  d {100*(w[a]-w_old[a]):+6.2f}pp  f {f[a]:+.4f}")

# gate approximation: gross edge = f . (w_new - w_old), turnover = 0.5 sum|d|
d = {a: w[a] - w_old[a] for a in wl}
gross_edge = sum(f[a] * d[a] for a in wl)
turnover = 0.5 * sum(abs(d[a]) for a in wl)
print(f"\ngross_edge {gross_edge:.5f}  turnover {turnover:.5f}  thresh {turnover*0.0003:.6f}")
print("gate pass:", gross_edge > turnover * 0.0003)
print("sum new w:", sum(w.values()))
print("crypto pair:", round(100*(w['BTC']+w['ETH']),2), "old", round(100*(w_old['BTC']+w_old['ETH']),2))
print("comm pair:", round(100*(w['WTI']+w['COPPER']),2), "old", round(100*(w_old['WTI']+w_old['COPPER']),2))
print("china pair:", round(100*(w['000300.SH']+w['000688.SH']),2), "old", round(100*(w_old['000300.SH']+w_old['000688.SH']),2))
