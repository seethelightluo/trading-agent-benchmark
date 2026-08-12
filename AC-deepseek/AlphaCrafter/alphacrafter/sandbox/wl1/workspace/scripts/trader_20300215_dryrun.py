"""Trader 2030-02-15 read-only dry-run: compute proposal weights exactly as
strategy.py would at the 0215 decision, WITHOUT submitting any orders or
advancing state. Also print 10d/20d returns to assess extension risk."""
import json, math
import numpy as np
import pandas as pd
import strategy as S
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
cur, tds = S._today_and_calendar()
print("current_date:", cur, "| n_assets:", len(assets))
print("watchlist:", assets)

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print("factors used:", used, "| FACTORS:", [(f, w, d) for f, w, d in S.FACTORS])

# 10d and 20d returns per asset (for extension analysis)
ret10, ret20, ma20rel = {}, {}, {}
for a, df in frames.items():
    if df is None or len(df) < 25:
        continue
    c = df["close"].astype(float)
    ret10[a] = float(c.iloc[-1] / c.iloc[-11] - 1.0)
    ret20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0)
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma20rel[a] = float(c.iloc[-1] / ma20 - 1.0)

scores = S._de_rank_value_traps(scores, frames, assets, cur)
regime = S._regime(frames, assets)
w = S._weights(scores, assets, regime)
w = S._composite_top2_cap(w, assets, scores)
w = S._composite_ma_guard(w, frames, assets)
w = S._ma_guard(w, frames, assets, cur)
w = S._crypto_cap(w, assets)
w = S._commod_cap(w, assets)

print("\nregime:", regime)
print("\n%-8s %8s %8s %8s %8s %8s" % ("asset", "weight", "score", "ret10", "ret20", "ma20rel"))
for a in sorted(assets, key=lambda x: -w[x]):
    print("%-8s %8.4f %8.4f %8.3f %8.3f %8.3f" % (
        a, w[a], scores[a],
        ret10.get(a, float("nan")), ret20.get(a, float("nan")), ma20rel.get(a, float("nan"))))
print("\nsum w:", sum(w.values()))
top = sorted(assets, key=lambda x: -w[x])[:3]
print("top3 weights:", [(a, round(w[a], 4)) for a in top])
ext = {a: r for a, r in ret10.items() if r > 0.10}
print("extended >+10% over 10d:", {a: round(r, 3) for a, r in sorted(ext.items(), key=lambda x: -x[1])})
