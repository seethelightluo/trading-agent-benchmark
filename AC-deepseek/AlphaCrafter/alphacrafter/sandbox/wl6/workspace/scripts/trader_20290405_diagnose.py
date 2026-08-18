"""Diagnose trader decision as of 2029-04-05 (decision day).

Replicates strategy.py factor/regime/target computation using only data
visible through the last completed bar (<= 2029-04-04). Read-only.
"""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import strategy as S

acc = get_account_dict()
assets = list(acc["watch_list"])
frames = {a: get_stock_daily_data(a, days=300) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
# Truncate to the last completed bar before decision day 2029-04-05.
cut = pd.Timestamp("2029-04-04")
closes_cut = {}
for a, c in closes.items():
    if c is None:
        closes_cut[a] = None
        continue
    cc = c[c.index <= cut] if hasattr(c.index, "max") else c[c["date"] <= cut]
    closes_cut[a] = cc

usable = [c.rename(a) for a, c in closes_cut.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join="inner")
print("panel dates:", panel.index.min(), "->", panel.index.max(), "rows:", len(panel))

regime = S.regime_from_market(panel)
print("regime at decision:", regime)

rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean()); v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
print("trend t-stat:", round(trend, 3))

vf = get_index_daily_data("VIX", days=300)
vix_close = vf.close.astype(float) if vf is not None else None
vix_cut = vix_close[vix_close.index <= cut] if vix_close is not None else None

raw = S.compute_raw_factors(closes_cut, vix_cut, assets)
factors = S.load_ensemble()
print("ensemble factors:", [f["factor_id"] for f in factors])

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()}

vol20 = S.vol20_map(closes_cut, assets)
valid_vol = {a: v for a, v in vol20.items() if v is not None and v > 0}
if valid_vol:
    vmin = min(valid_vol.values())
    inv = {a: (vmin / valid_vol[a] if a in valid_vol else 0.0) for a in assets}
    inv_top_sum = sum(inv.get(a, 0.0) for a in top)
    if inv_top_sum > 1e-12:
        blended = {a: ((1.0 - S.VOL_BLEND) * w.get(a, 0.0)
                       + S.VOL_BLEND * (inv.get(a, 0.0) / inv_top_sum if a in top else 0.0))
                   for a in assets}
        bsum = sum(blended.values())
        if bsum > 1e-12:
            w = {a: v / bsum for a, v in blended.items()}

frozen = S.frozen_set(closes_cut, assets)
print("frozen:", frozen)
w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_min_xau(w, assets)
for _ in range(50):
    prev = dict(w)
    w = S.apply_cap(w, assets)
    w = S.apply_crypto_cap(w, assets)
    w = S.apply_single_cap(w, assets, "WTI", S.WTI_CAP)
    w = S.apply_frozen_cap(w, assets, frozen)
    w = S.apply_min_xau(w, assets)
    if sum(abs(w.get(a, 0.0) - prev.get(a, 0.0)) for a in assets) < 1e-11:
        break

total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
           for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\nProposed target (as of 2029-04-05 decision):")
for a in sorted(assets, key=lambda x: -weights[x]):
    print(f"  {a:10s} {weights[a]*100:6.2f}%")
print("sum:", round(sum(weights.values()), 6))

# Block performance of each leg 2029-04-04 -> 2029-04-18 (last completed bar of block)
print("\nPer-asset block return (2029-04-04 close -> 2029-04-18 close):")
for a in assets:
    c = closes.get(a)
    if c is None or len(c) < 2:
        continue
    c0 = c[c.index <= cut]
    if len(c0) < 1:
        continue
    p0 = float(c0.iloc[-1])
    p1 = float(c.iloc[-1])
    print(f"  {a:10s} {p0:12.4f} -> {p1:12.4f}  {100*(p1/p0-1):+7.2f}%")
