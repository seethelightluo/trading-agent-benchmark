"""Trader diagnostic: verify guardrail pipeline produces capped weights.

Replicates strategy_hook's weight construction EXACTLY using the same
functions from strategy.py, on the current decision date (2028-06-01, data
visible through 2028-05-31). Checks every trader-side cap/floor invariant.
"""
import json
import pandas as pd

import strategy as S
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
assets = list(acc["watch_list"])
print("assets:", assets)

frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
print("usable:", len(usable), usable)
panel = pd.concat(usable, axis=1, join="inner")
print("panel shape:", panel.shape, "last date:", panel.index[-1])

factors = S.load_ensemble()
print("ensemble:", [(f["factor_id"], f["weight"], f.get("direction")) for f in factors])

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
print("regime:", regime)
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values())
span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()} if wsum >= 1e-9 else {a: 1.0 / K if a in top else 0.0 for a in assets}

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
print("frozen:", frozen)
w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_cap(w, assets)
w = S.apply_min_xau(w, assets)
w = S.apply_crypto_cap(w, assets)
w = S.apply_single_cap(w, assets, "WTI", S.WTI_CAP)
w = S.apply_frozen_cap(w, assets, frozen)

total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
           for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\n=== FINAL TARGET WEIGHTS (guardrail pipeline) ===")
for a in sorted(weights, key=lambda x: -weights[x]):
    flag = ""
    if weights[a] > S.CAP + 1e-9:
        flag += " CAP_VIOLATION>12%"
    if a in ("BTC", "ETH") and weights[a] > S.CRYPTO_EACH + 1e-9:
        flag += " CRYPTO_VIOLATION"
    if a == "WTI" and weights[a] > S.WTI_CAP + 1e-9:
        flag += " WTI_VIOLATION"
    if a in frozen and weights[a] > S.FROZEN_CAP + 1e-9:
        flag += " FROZEN_VIOLATION"
    print(f"  {a:10s} {weights[a]*100:6.2f}%{flag}")

print("\n=== INVARIANTS ===")
print("sum:", round(sum(weights.values()), 10))
print("max weight:", max(weights.values()))
print("crypto total:", round(weights.get("BTC",0)+weights.get("ETH",0), 4))
print("def floor (XAU+US10Y):", round(weights.get("XAU",0)+weights.get("US10Y",0), 4),
      "floor for regime:", S.FLOOR[regime])
print("XAU min:", round(weights.get("XAU",0), 4), ">= 0.04:", weights.get("XAU",0) >= 0.04 - 1e-9)
print("WTI:", round(weights.get("WTI",0), 4))
print("all non-negative:", all(v >= 0 for v in weights.values()))

# forecast returns (same as strategy)
score_mean = sum(score.values()) / len(assets)
score_std = (sum((x - score_mean) ** 2 for x in score.values()) / len(assets)) ** 0.5
_rets = panel.pct_change().dropna()
ret_scale = float(_rets.tail(252).std(axis=1, ddof=0).median()) if len(_rets) else 0.0
forecast = {a: (score[a] / score_std if score_std > 1e-12 else 0.0) * ret_scale for a in assets}
print("\nforecast sample:", {k: round(v, 6) for k, v in list(forecast.items())[:5]})
