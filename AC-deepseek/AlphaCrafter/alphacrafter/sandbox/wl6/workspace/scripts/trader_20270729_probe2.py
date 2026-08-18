"""Trader probe 2027-07-29: verify mom_120d_skip5 admission + target vector.

Replicates strategy.py internals WITHOUT calling rebalance_to_weights/step.
Checks: ensemble load, factor coverage, regime, weight invariants.
"""
import json
from pathlib import Path
from math import isfinite

from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import strategy as S

assets = list(get_account_dict()["watch_list"])
print("assets:", len(assets), assets)

frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = __import__("pandas").concat(usable, axis=1, join="inner")
print("usable:", len(usable), "panel rows:", len(panel))

factors = S.load_ensemble()
print("ensemble:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in factors])

vf = S.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

fid = "mom_120d_skip5"
vals = {a: raw[fid][a] for a in assets}
finite_vals = {a: v for a, v in vals.items() if v is not None and isfinite(v)}
print(f"{fid}: {len(finite_vals)}/15 finite")
for a, v in sorted(finite_vals.items(), key=lambda kv: -kv[1]):
    print(f"   {a:10s} {v:+.4f}")

for f in factors:
    fid2 = f["factor_id"]
    n = sum(1 for a in assets if raw.get(fid2, {}).get(a) is not None and isfinite(raw[fid2][a]))
    print(f"coverage {fid2}: {n}/15")

regime = S.regime_from_market(panel)
print("regime:", regime)

# Replicate scoring pipeline
score = {a: 0.0 for a in assets}
for f in factors:
    fid3, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid3, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values())
span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()} if wsum > 1e-9 else {a: 1.0 / K for a in top}

vol20 = S.vol20_map(closes, assets)
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

w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_cap(w, assets)
w = S.apply_min_xau(w, assets)
w = S.apply_crypto_cap(w, assets)

total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
           for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\nTARGET WEIGHTS (sum=%.6f):" % sum(weights.values()))
for a, v in sorted(weights.items(), key=lambda kv: -kv[1]):
    print(f"   {a:10s} {v:.4f}")
print("sum:", sum(weights.values()))
print("crypto:", weights.get("ETH", 0) + weights.get("BTC", 0))
print("XAU:", weights.get("XAU", 0))
