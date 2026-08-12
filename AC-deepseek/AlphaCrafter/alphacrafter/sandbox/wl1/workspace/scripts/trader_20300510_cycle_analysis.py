"""Trader cycle analysis for block 2030-04-26 -> 2030-05-10.

Reconstruct the 0426 executed target (slice data to <= 2030-04-26 to avoid
leakage) and compute per-asset block returns / contribution estimates.
"""
import json
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import strategy as st

DECISION = "2030-04-26"
BLOCK_END = "2030-05-10"

acc = json.load(open("../persistent/account.json"))
assets = list(acc.get("watch_list", []))
print("n assets:", len(assets))

# ---- Reconstruct 0426 proposal weights (data sliced to decision date) ----
frames = st._fetch(assets)
for a, df in frames.items():
    if df is not None:
        frames[a] = df[df.index <= pd.Timestamp(DECISION)]

scores, used = st._scores(frames, assets, DECISION)
print("factors used:", used)
scores = st._de_rank_value_traps(scores, frames, assets, DECISION)
regime = st._regime(frames, assets)
print("regime:", regime)
w = st._weights(scores, assets, regime)
w = st._composite_top2_cap(w, assets, scores)
w = st._composite_ma_guard(w, frames, assets)
w = st._ma_guard(w, frames, assets, DECISION)
w = st._crypto_cap(w, assets)
w = st._commod_cap(w, assets)
print("\nReconstructed 0426 executed target weights:")
for a in sorted(assets, key=lambda x: -w[x]):
    print(f"  {a:8s} {100*w[a]:6.2f}%")
print("sum:", round(sum(w.values()), 6), "min:", min(w.values()), "max:", max(w.values()))

# ---- Block returns 0426 -> 0510 (via live data API, last row = 0510) ----
frames2 = st._fetch(assets)
rets = {}
for a, df in frames2.items():
    if df is None or len(df) < 2:
        rets[a] = None
        continue
    c = df["close"].astype(float)
    # last close = block end (0510); find close on/before decision date
    sub = c[c.index <= pd.Timestamp(DECISION)]
    if len(sub) == 0:
        rets[a] = None
        continue
    p0 = float(sub.iloc[-1])
    p1 = float(c.iloc[-1])
    rets[a] = p1 / p0 - 1.0

print("\nPer-asset block return and contribution (w_end * r estimate):")
tot = 0.0
for a in sorted(assets, key=lambda x: -(rets.get(x) or 0.0)):
    r = rets.get(a)
    if r is None:
        print(f"  {a:8s} n/a")
        continue
    cont = w[a] * r
    tot += cont
    print(f"  {a:8s} r={100*r:7.2f}%  w_start={100*w[a]:5.2f}%  cont~{100*cont:+6.2f}%")
print(f"sum contribution ~ {100*tot:+.2f}%")

# ---- Factor snapshot at decision: which factors drove top picks ----
print("\nFactor values at 0426 (rank 0=worst .. 1=best):")
for fid, wf, direction in st.FACTORS:
    vals = st._factor_values(frames, fid, DECISION)
    r = st._ranks(vals, assets)
    top = sorted(assets, key=lambda a: r[a], reverse=True)[:4]
    bot = sorted(assets, key=lambda a: r[a])[:3]
    print(f"  {fid} (dir {direction:+d}, w {wf:.2f}): top {[(a, round(r[a],2)) for a in top]} | bot {[(a, round(r[a],2)) for a in bot]}")
