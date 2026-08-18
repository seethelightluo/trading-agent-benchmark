"""Dry-run of strategy target (no rebalance_to_weights call) to document cycle."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import strategy as st
from alphacrafter.sim.utils import get_account_dict

assets = list(get_account_dict()["watch_list"])
frames = {a: st.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None) for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join="inner")
factors = st.load_ensemble()
vf = st.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = st.compute_raw_factors(closes, vix_close, assets)

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = st.rank_series(raw.get(fid, {}), assets)
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

regime = st.regime_from_market(panel)
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean()); v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
wsum = sum(w.values())
w = {a: v / wsum for a, v in w.items()} if wsum > 1e-9 else {a: 1.0/K for a in top}
vol20 = st.vol20_map(closes, assets)
valid_vol = {a: v for a, v in vol20.items() if v is not None and v > 0}
if valid_vol:
    vmin = min(valid_vol.values())
    inv = {a: (vmin / valid_vol[a] if a in valid_vol else 0.0) for a in assets}
    inv_top_sum = sum(inv.get(a, 0.0) for a in top)
    if inv_top_sum > 1e-12:
        blended = {a: ((1.0 - st.VOL_BLEND) * w.get(a, 0.0)
                       + st.VOL_BLEND * (inv.get(a, 0.0) / inv_top_sum if a in top else 0.0)) for a in assets}
        bsum = sum(blended.values())
        if bsum > 1e-12:
            w = {a: v / bsum for a, v in blended.items()}
frozen = st.frozen_set(closes, assets)
w = st.apply_floor(w, assets, [a for a in st.DEF if a in assets], st.FLOOR[regime])
w = st.apply_min_xau(w, assets)
for _ in range(50):
    prev = dict(w)
    w = st.apply_cap(w, assets)
    w = st.apply_crypto_cap(w, assets)
    w = st.apply_single_cap(w, assets, "WTI", st.WTI_CAP)
    w = st.apply_frozen_cap(w, assets, frozen)
    w = st.apply_min_xau(w, assets)
    if sum(abs(w.get(a, 0.0) - prev.get(a, 0.0)) for a in assets) < 1e-11:
        break
total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0/len(assets)) for a in assets}
rem = 1.0 - sum(weights.values()); weights[assets[0]] += rem

print("regime:", regime, "trend=%.3f" % trend, "K =", K)
print("frozen:", frozen)
for a in assets:
    if weights[a] >= 0.005:
        print(f"  {a}: {weights[a]*100:.2f}%")
print("sum:", round(sum(weights.values()), 6))
