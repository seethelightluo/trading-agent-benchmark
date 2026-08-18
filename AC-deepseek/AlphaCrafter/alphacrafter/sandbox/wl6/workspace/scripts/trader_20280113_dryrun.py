"""Trader dry-run: replicate strategy_hook weights WITHOUT calling rebalance_to_weights.

Purpose: preview the target vector the live hook would submit on 2028-01-13
(block start), confirm frozen-name exposure, and sanity-check guardrail output.
"""
import json
import sys
import pandas as pd
sys.path.insert(0, ".")
import strategy as S
from alphacrafter.sim.utils import get_account_dict

assets = list(get_account_dict()["watch_list"])
frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
print("usable assets:", len(usable), "/", len(assets))

# --- frozen-name check: 60d realized vol and 120d price range ---
print("\n=== frozen-name scan (last 120 bars) ===")
for a in assets:
    c = closes.get(a)
    if c is None or len(c) < 140:
        print(f"{a:10s} no-data"); continue
    ret = c.pct_change().dropna()
    v60 = float(ret.tail(60).std())
    rng = float(c.tail(120).max() - c.tail(120).min())
    rng_pct = rng / float(c.tail(120).mean()) * 100
    flag = "FROZEN" if v60 < 1e-9 or rng_pct < 0.01 else ""
    print(f"{a:10s} vol60={v60:10.2e} range120={rng_pct:8.4f}% {flag}")

# --- replicate hook computation (no rebalance call) ---
factors = S.load_ensemble()
print("\nensemble:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in factors])
panel = pd.concat(usable, axis=1, join="inner") if usable else None
import pandas as pd
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
lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
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
w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_cap(w, assets)
w = S.apply_min_xau(w, assets)
w = S.apply_crypto_cap(w, assets)
w = S.apply_single_cap(w, assets, "WTI", S.WTI_CAP)
w = S.apply_frozen_cap(w, assets, S.frozen_set(closes, assets))
total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
           for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\n=== proposed target (dry-run) ===")
for a in sorted(weights, key=lambda x: -weights[x]):
    print(f"{a:10s} w={weights[a]:6.3f}")
print("sum:", round(sum(weights.values()), 6))

# rank detail for frozen names
print("\n=== frozen-name score detail ===")
for a in assets:
    if closes.get(a) is not None and len(closes[a]) >= 140:
        ret = closes[a].pct_change().dropna()
        v60 = float(ret.tail(60).std())
        if v60 < 1e-9:
            print(f"{a:10s} score={score[a]:+.4f} raw_w={raw_w[a]:.4f} in_top={a in top} w={weights[a]:.4f}")

# current holdings for turnover estimate
acc = get_account_dict()
pos = {p["symbol"]: p.get("market_value", 0) for p in acc.get("positions", [])}
tot_mv = sum(pos.values()) or 1
print("\n=== current holdings ===")
for a in sorted(pos, key=lambda x: -pos[x]):
    print(f"{a:10s} w={pos[a]/tot_mv:6.3f}")
