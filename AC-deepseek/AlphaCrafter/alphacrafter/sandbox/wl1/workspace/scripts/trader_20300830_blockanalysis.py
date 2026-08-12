"""Trader block analysis: reconstruct 2030-08-30 proposal weights and per-asset
block contribution 08-30 -> 09-13 (close-to-close)."""
import json
import numpy as np
import pandas as pd
import strategy as st
from alphacrafter.sim.utils import get_account_dict

DEC = "2030-08-30"
END = "2030-09-13"

acc = get_account_dict()
assets = list(acc["watch_list"])

# Fetch frames (through current sim date) and truncate to DEC for the decision
frames = st._fetch(assets)
dec_frames = {}
for a, df in frames.items():
    if df is None:
        dec_frames[a] = None
        continue
    dec_frames[a] = df[df.index <= pd.Timestamp(DEC)]

# Reconstruct the decision exactly as strategy_hook would
scores, used = st._scores(dec_frames, assets, DEC)
print("factors used:", used)
scores = st._de_rank_value_traps(scores, dec_frames, assets, DEC)
regime = st._regime(dec_frames, assets)
print("regime:", regime)
w = st._weights(scores, assets, regime)
w = st._composite_top2_cap(w, assets, scores)
w = st._composite_ma_guard(w, frames, assets)  # MA uses full frames (same as live)
w = st._ma_guard(w, frames, assets, DEC)
for _ in range(6):
    w = st._commod_cap(w, assets)
    w = st._crypto_cap(w, assets)

tot = sum(w.values())
print("sum(w):", tot)

order = sorted(assets, key=lambda a: (scores[a], a))
print("\nScore rank -> weight (executed target @08-30):")
for i, a in enumerate(order):
    print(f"  rank{i+1:2d} {a:8s} score={scores[a]:+.4f} w={w[a]*100:6.2f}%")

# Block returns close-to-close DEC -> END
print("\nBlock returns and contribution (w * ret):")
contrib = 0.0
rows = []
for a in assets:
    df = frames.get(a)
    if df is None:
        print(f"  {a:8s} NO DATA")
        continue
    c_dec = float(df[df.index <= pd.Timestamp(DEC)]["close"].iloc[-1])
    c_end = float(df[df.index <= pd.Timestamp(END)]["close"].iloc[-1])
    r = c_end / c_dec - 1.0
    c = w[a] * r
    contrib += c
    rows.append((a, r, c))
for a, r, c in sorted(rows, key=lambda x: -x[2]):
    print(f"  {a:8s} ret={r*100:+6.2f}% w={w[a]*100:6.2f}% contrib={c*100:+6.3f}%")
print(f"\nEstimated block contribution (sum w*ret): {contrib*100:+.3f}%")
print(f"NAV check: 890798 * (1+{contrib:.5f}) = {890798*(1+contrib):.0f} vs actual {acc['net_assets']:.0f}")

# Report block-end weights (drift) from account
print("\nBlock-end weights (@09-13):")
mv = {p["symbol"]: p["market_value"] for p in acc["positions"]}
nav = acc["net_assets"]
for a in sorted(assets, key=lambda x: -mv.get(x, 0)):
    print(f"  {a:8s} {mv.get(a,0)/nav*100:6.2f}%  pnl={acc['positions'][[p['symbol'] for p in acc['positions']].index(a)]['profit_loss'] if a in [p['symbol'] for p in acc['positions']] else 0:+.0f}")
