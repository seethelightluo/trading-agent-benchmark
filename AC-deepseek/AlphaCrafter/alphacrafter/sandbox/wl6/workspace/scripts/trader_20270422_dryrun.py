"""Trader dry-run 2027-04-22: replicate strategy.py target construction exactly."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import strategy as S

acct = get_account_dict()
assets = list(acct["watch_list"])
frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None) for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join="inner")

factors = S.load_ensemble()
print("ensemble:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in factors])

vf = S.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)
    print("\n%s w=%.2f d=%+d" % (fid, w, d))
    for a in sorted(assets, key=lambda x: -r[x]):
        print("   %-10s rank=%5.2f contrib=%+6.3f" % (a, r[a], (w*d)*(r[a]-0.5)))

regime = S.regime_from_market(panel)
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
print("\nregime=%s K=%d" % (regime, K))
lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()}

vol20 = S.vol20_map(closes, assets)
valid_vol = {a: v for a, v in vol20.items() if v is not None and v > 0}
if valid_vol:
    vmin = min(valid_vol.values())
    inv = {a: (vmin / valid_vol[a] if a in valid_vol else 0.0) for a in assets}
    inv_top_sum = sum(inv.get(a, 0.0) for a in top)
    if inv_top_sum > 1e-12:
        blended = {a: ((1 - S.VOL_BLEND) * w.get(a, 0.0) + S.VOL_BLEND * (inv.get(a, 0.0) / inv_top_sum if a in top else 0.0)) for a in assets}
        bsum = sum(blended.values())
        if bsum > 1e-12:
            w = {a: v / bsum for a, v in blended.items()}

w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_cap(w, assets)
total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0/len(assets)) for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\nproposed target weights:")
for a in sorted(weights, key=lambda x: -weights[x]):
    print("  %-10s %6.2f%%  score=%+6.3f" % (a, weights[a]*100, score[a]))
print("sum=%.6f" % sum(weights.values()))
print("\nXAU weight: %.2f%%" % (weights["XAU"]*100))
print("US10Y weight: %.2f%%" % (weights["US10Y"]*100))
