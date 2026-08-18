"""Trader probe: compute the target weights strategy.py would submit today
(2029-05-31) with the current 6-factor ensemble, WITHOUT calling
rebalance_to_weights (no account mutation). Mirrors strategy_hook logic.
"""
import sys, json
sys.path.insert(0, '.')
import strategy as S
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
assets = list(acct["watch_list"])
frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = S.pd.concat(usable, axis=1, join="inner")

factors = S.load_ensemble()
print("ensemble factors:", [(f["factor_id"], round(f.get("weight",0),4), f.get("direction")) for f in factors])
assert len(factors) <= 10, "ensemble > 10 factors!"

vf = S.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

regime = S.regime_from_market(panel)
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
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

print("regime:", regime, "| K:", K, "| frozen:", sorted(frozen))
print("target weights:")
for a in assets:
    print(f"  {a:10s} {weights[a]*100:7.2f}%  (score {score[a]:+.4f})")
print("sum:", round(sum(weights.values()), 12))
print("XAU+US10Y:", round(weights.get('XAU',0)+weights.get('US10Y',0), 4))
print("crypto combined:", round(weights.get('BTC',0)+weights.get('ETH',0), 4))
