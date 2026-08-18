"""Trader diagnostic 2028-05-18: replicate strategy_hook logic up to rebalance
call and print regime, factor ranks, target weights (no live mutation)."""
import json, sys
sys.path.insert(0, ".")
import strategy as S
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = list(get_account_dict()["watch_list"])
frames = {a: get_stock_daily_data(a, days=S.N_DAYS) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
print("n usable:", len(usable), "of", len(assets))
panel = __import__("pandas").concat(usable, axis=1, join="inner")
print("panel rows:", len(panel))

regime = S.regime_from_market(panel)
print("regime:", regime)

factors = S.load_ensemble()
print("factors:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in factors])

vf = get_index_daily_data("VIX", days=S.N_DAYS)
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

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
w = {a: v / wsum for a, v in w.items()} if wsum > 1e-9 else {a: (1.0/K if a in top else 0.0) for a in assets}

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
print("frozen:", sorted(frozen))
w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_cap(w, assets)
w = S.apply_min_xau(w, assets)
w = S.apply_crypto_cap(w, assets)
w = S.apply_single_cap(w, assets, "WTI", S.WTI_CAP)
w = S.apply_frozen_cap(w, assets, frozen)

total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0/len(assets)) for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\n=== TARGET WEIGHTS (proposal) ===")
for a in sorted(assets, key=lambda x: -weights[x]):
    print(f"  {a:10s} {weights[a]*100:6.2f}%  score={score[a]:+.4f}")

# factor rank table
print("\n=== FACTOR RANKS (rank in [0,1], higher=better for direction) ===")
for f in factors:
    fid, d = f["factor_id"], f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    sorted_r = sorted(assets, key=lambda a: -r[a])
    print(f"  {fid:22s} dir={d:+d} top: " + ", ".join(f"{a}({r[a]:.2f})" for a in sorted_r[:5]))

# current account state
acc = get_account_dict()
print("\naccount net_assets:", acc.get("net_assets"), "cash:", acc.get("available_cash"))
pos = {p["symbol"]: p.get("market_value", 0) for p in acc.get("positions", [])}
tot = acc.get("net_assets", 1)
print("current weights:", {k: round(v/tot, 4) for k, v in sorted(pos.items(), key=lambda x: -x[1])})
