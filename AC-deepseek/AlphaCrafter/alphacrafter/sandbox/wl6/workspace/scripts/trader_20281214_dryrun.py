"""Trader dry-run: replicate strategy.py scoring with the new ensemble and
print the proposed target (NO rebalance_to_weights call - read-only)."""
import json
from pathlib import Path
from math import isfinite
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

import sys
sys.path.insert(0, ".")
import strategy as S

assets = list(get_account_dict()["watch_list"])
frames = {a: get_stock_daily_data(a, days=300) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None) for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join="inner")

factors = S.load_ensemble()
print("Ensemble:", [(f["factor_id"], f["weight"], f["direction"]) for f in factors])

vf = get_index_daily_data("VIX", days=300)
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

# Check new factor validity
cn_vals = raw.get("beta_cn10y_60d", {})
n_valid = sum(1 for v in cn_vals.values() if v is not None)
print(f"\nbeta_cn10y_60d valid values: {n_valid}/15 (CN10Y frozen -> likely 0)")

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    contrib = {a: (w * d) * (r[a] - 0.5) for a in assets}
    nz = sum(1 for v in raw.get(fid, {}).values() if v is not None)
    print(f"{fid:>24} w={w:.2f} d={d:+d} n_valid={nz:>2} score_range=[{min(contrib.values()):+.4f},{max(contrib.values()):+.4f}]")
    for a in assets:
        score[a] += contrib[a]

regime = S.regime_from_market(panel)
print(f"\nRegime: {regime}")
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]

lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()} if wsum > 1e-9 else {a: 1.0/K if a in top else 0.0 for a in assets}

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

frozen = S.frozen_set(closes, assets)
print("Frozen set:", sorted(frozen))
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
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets)) for a in assets}
print("\nProposed target weights (sum=%.4f):" % sum(weights.values()))
for a in sorted(assets, key=lambda x: -weights[x]):
    print(f"  {a:>10} {weights[a]:.4f}")
