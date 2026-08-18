"""Trader probe 2030-07-11: verify ensemble match + compute would-be target.

Read-only: does NOT call rebalance_to_weights, step, or backtest.
"""
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import strategy as S

assets = list(get_account_dict()["watch_list"])
print("watch_list:", assets)
print("n_assets:", len(assets))

# 1. Ensemble check
ens = json.loads(Path("factor_ensemble.json").read_text())
factors = [dict(f) for f in ens.get("selected_factors", [])]
wsum = sum(f.get("weight", 0.0) for f in factors)
print("\nensemble:", ens.get("generated_at"), "n_factors:", len(factors), "sum_w:", round(wsum, 6))
for f in factors:
    print("  ", f["factor_id"], "w=", f.get("weight"), "dir=", f.get("direction"))

known = ["beta_vix_60d_neg", "beta_cn10y_60d", "beta_chi_60d", "vol_of_vol20x60",
         "low_vol_20d", "mom_10d_skip5", "mom_120d_skip5", "down_vol_ratio_20x120",
         "sign_ewma_60d", "vol_beta_spx_60d", "skew_20d_neg"]
missing = [f["factor_id"] for f in factors if f["factor_id"] not in known]
print("factors missing from compute_raw_factors:", missing if missing else "NONE - all computable")

# 2. Build frames as strategy does
frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
print("\nusable series:", len(usable), "of", len(assets))
panel = pd.concat(usable, axis=1, join="inner")
print("panel tail date:", panel.index[-1], "rows:", len(panel))

vf = S.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

# 3. Composite score + target (same math as strategy_hook, no submission)
score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

regime = S.regime_from_market(panel)
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values())
span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()} if wsum > 1e-9 else {a: 1.0 / K if a in top else 0.0 for a in assets}

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

frozen = S.frozen_set(closes, assets)
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

print("\nregime:", regime, "K:", K, "frozen:", sorted(frozen))
print("target weights (sum=%.6f):" % sum(weights.values()))
for a in sorted(assets, key=lambda x: -weights[x]):
    print("  %-10s %6.2f%%" % (a, 100 * weights[a]))
assert abs(sum(weights.values()) - 1.0) < 1e-9, "sum-to-one violated"
assert all(v >= 0 for v in weights.values()), "negative weight"
print("\nOK: sum-to-one, non-negative, all factors computable.")
