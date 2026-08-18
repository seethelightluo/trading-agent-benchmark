"""Trader read-only probe 2030-08-22: regime, factor scores, would-be target.

Replicates strategy.py computation WITHOUT calling rebalance_to_weights so the
live account proposal state is untouched. For next-block planning only.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strategy as S
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = list(get_account_dict()["watch_list"])
frames = {a: get_stock_daily_data(a, days=300) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None) for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = __import__("pandas").concat(usable, axis=1, join="inner")

print("=== DATA VISIBILITY ===")
for a in assets:
    c = closes.get(a)
    if c is not None and len(c):
        print(f"  {a:10s} last={str(c.index[-1])[:10]} n={len(c)} close={c.iloc[-1]:.2f}")
    else:
        print(f"  {a:10s} NO DATA")

print("\n=== REGIME (decision 2030-08-22, visible through 08-21) ===")
regime = S.regime_from_market(panel)
print("regime:", regime)
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20 ** 0.5) if v20 and v20 > 1e-12 else 0.0
print(f"mkt 20d mean ret {r20*100:.3f}%/d  std {v20*100:.3f}%  trend t {trend:.2f}")

# 20d cross-asset returns for a feel of the last block
print("\n=== last 20d per-asset return (visible) ===")
for a in assets:
    c = closes.get(a)
    if c is None or len(c) < 25:
        continue
    r = c.iloc[-1] / c.iloc[-21] - 1.0
    print(f"  {a:10s} {r*100:+7.2f}%")

print("\n=== FACTOR ENSEMBLE ===")
factors = S.load_ensemble()
print("n factors:", len(factors))
for f in factors:
    print(f"  {f['factor_id']} w={f.get('weight')} dir={f.get('direction')}")

print("\n=== RAW FACTOR VALUES (last bar) ===")
vf = get_index_daily_data("VIX", days=300)
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
if vix_close is not None:
    print("VIX last:", round(float(vix_close.iloc[-1]), 2))
raw = S.compute_raw_factors(closes, vix_close, assets)
for fid in ["beta_vix_60d_neg", "sign_ewma_60d", "vol_beta_spx_60d", "mom_10d_skip5",
            "mom_120d_skip5", "down_vol_ratio_20x120", "skew_20d_neg"]:
    vals = raw.get(fid, {})
    top = sorted(vals.items(), key=lambda kv: (-(kv[1] if kv[1] is not None else -9e9)))
    print(f"  {fid:22s} top3: " + ", ".join(f"{a}={v:.3f}" for a, v in top[:3] if v is not None))

# Replicate score -> weights (strategy logic, read-only)
score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values())
span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
if wsum < 1e-9:
    w = {a: (1.0 / K if a in top else 0.0) for a in assets}
else:
    w = {a: v / wsum for a, v in w.items()}

vol20 = S.vol20_map(closes, assets)
valid_vol = {a: v for a, v in vol20.items() if v is not None and v > 0}
if valid_vol:
    vmin = min(valid_vol.values())
    inv = {a: (vmin / valid_vol[a] if a in valid_vol else 0.0) for a in assets}
    inv_top_sum = sum(inv.get(a, 0.0) for a in top)
    if inv_top_sum > 1e-12:
        blended = {a: ((1.0 - S.VOL_BLEND) * w.get(a, 0.0)
                       + S.VOL_BLEND * (inv.get(a, 0.0) / inv_top_sum if a in top else 0.0)) for a in assets}
        bsum = sum(blended.values())
        if bsum > 1e-12:
            w = {a: v / bsum for a, v in blended.items()}

frozen = S.frozen_set(closes, assets)
print("\nfrozen set:", frozen)
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
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\n=== WOULD-BE TARGET WEIGHTS (next block; NOT submitted) ===")
for a in sorted(weights, key=lambda x: -weights[x]):
    print(f"  {a:10s} {weights[a]*100:6.2f}%")
print("sum:", round(sum(weights.values()), 6))

# current live weights for comparison
acc = get_account_dict()
na = acc.get("net_assets", 0.0)
print("\n=== CURRENT LIVE WEIGHTS (post 08-08 rebalance + drift) ===")
for p in sorted(acc.get("positions", []), key=lambda x: -x["market_value"]):
    print(f"  {p['symbol']:10s} {p['market_value']/na*100:6.2f}%")
