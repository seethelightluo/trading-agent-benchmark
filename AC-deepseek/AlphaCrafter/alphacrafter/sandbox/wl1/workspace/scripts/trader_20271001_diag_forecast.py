"""Diagnostic: replicate strategy score/forecast computation on latest data.

Checks whether composite-score dispersion / forecast magnitude is sufficient
for the rebalance gate (gross edge > one-way turnover * 3bp). Only touches
strategy.py logic + public data APIs; does not mutate account/date state.
"""
import json
import math
import numpy as np
import pandas as pd

from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

import sys
sys.path.insert(0, ".")
import strategy as S  # reuse v7 functions (read-only)

acc = get_account_dict()
assets = list(acc["watch_list"])
cur, tds = S._today_and_calendar()
print("current_date:", cur, "| n_assets:", len(assets))

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print("factors used:", used, "of", len(S.FACTORS))
for a in sorted(assets, key=lambda x: -scores[x]):
    print(f"  {a:10s} score={scores[a]:+.4f}")

regime = S._regime(frames, assets)
w = S._weights(scores, assets, regime)
w = S._ma_guard(w, frames, assets, cur)
f = S._forecasts(scores, assets)

# turnover vs currently executed target
exec_w = acc.get("last_executed_target_weights") or acc.get("last_target_weights")
if exec_w:
    t2 = sum(abs(w[a] - exec_w.get(a, 0.0)) for a in assets) / 2.0
    print(f"one-way turnover vs executed target: {t2*100:.2f}%")
    # crude signed edge if forecasts align with weight deltas
    edge = sum(f[a] * (w[a] - exec_w.get(a, 0.0)) for a in assets)
    cost = t2 * 0.0003
    print(f"crude signed edge: {edge*100:.2f}%  cost: {cost*100:.4f}%  pass: {edge > cost}")

print("forecast range:", min(f.values()), max(f.values()))
print("n |f|>0.005:", sum(1 for v in f.values() if abs(v) > 0.005))
print("n |f|>0.02:", sum(1 for v in f.values() if abs(v) > 0.02))
# score dispersion
sc = list(scores.values())
print(f"score mean={np.mean(sc):+.4f} std={np.std(sc):.4f} min={min(sc):+.4f} max={max(sc):+.4f}")
