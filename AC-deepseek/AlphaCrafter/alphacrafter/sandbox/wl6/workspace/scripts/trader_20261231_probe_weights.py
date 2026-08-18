"""Trader probe: replicate strategy.py scoring to inspect proposed weights (read-only)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import strategy as S
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
assets = list(acct["watch_list"])
print("watch_list:", assets)
print("cash:", acct.get("available_cash"), "net:", acct.get("net_assets"))
print("positions:")
for p in acct.get("positions", []):
    print("  ", p["symbol"], p.get("quantity"), "mv=", round(p.get("market_value", 0), 2))
print("pending orders:", acct.get("orders", []))

frames = {a: S.stock(a) for a in assets}
closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
          for a, f in frames.items()}
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
print("usable frames:", len(usable), "min len:", min(len(c) for c in usable) if usable else 0)

factors = S.load_ensemble()
print("ensemble:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in factors])

panel = __import__("pandas").concat(usable, axis=1, join="inner")
regime = S.regime_from_market(panel)
print("regime:", regime)

vf = S.index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
raw = S.compute_raw_factors(closes, vix_close, assets)

score = {a: 0.0 for a in assets}
for f in factors:
    fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
    r = S.rank_series(raw.get(fid, {}), assets)
    print(f"  factor {fid} w={w} d={d}: top=",
          sorted(assets, key=lambda a: r[a], reverse=True)[:4],
          "bottom=", sorted(assets, key=lambda a: r[a])[:3])
    for a in assets:
        score[a] += (w * d) * (r[a] - 0.5)

K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
lo = min(score.values()); span = max(max(score.values()) - lo, 1e-9)
raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
if sum(w.values()) < 1e-9:
    w = {a: (1.0 / K if a in top else 0.0) for a in assets}
def_assets = [a for a in S.DEF if a in assets]
w = S.apply_floor(w, assets, def_assets, S.FLOOR[regime])
w = S.apply_cap(w, assets)
total = sum(w.values())
weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
           for a in assets}
rem = 1.0 - sum(weights.values())
weights[assets[0]] += rem

print("\nproposed weights (sum=%.6f):" % sum(weights.values()))
for a in sorted(weights, key=lambda x: -weights[x]):
    print("  %-10s %6.2f%%" % (a, weights[a] * 100))
print("\nregime:", regime, "K:", K)
