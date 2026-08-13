"""Trader preview: compute the 2033-03-25 proposal target (no execution).

Replicates strategy.py's pipeline using only its internal helpers so we can
report the intended 15-asset target, scores and regime without submitting a
rebalance. Does NOT call the registered hook and does NOT touch account state.
"""
import json
import sys
sys.path.insert(0, ".")
import strategy as S

cur, tds = S._today_and_calendar()
print("sim date:", cur, "| in tds:", cur in tds)

assets = [a.strip() for a in open("watch_list.txt").read().split(",")] if False else None
# use the same source of truth as the hook
from alphacrafter.sim.utils import get_account_dict
acc = get_account_dict()
assets = list(acc.get("watch_list", []))
print("n assets:", len(assets))
assert len(assets) == 15

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print("factors used:", used)
scores = S._de_rank_value_traps(scores, frames, assets, cur)
regime = S._regime(frames, assets)
print("regime:", regime)

w = S._weights(scores, assets, regime)
w = S._composite_top2_cap(w, assets, scores)
w = S._composite_ma_guard(w, frames, assets)
w = S._ma_guard(w, frames, assets, cur)
for _ in range(6):
    w = S._commod_cap(w, assets)
    w = S._crypto_cap(w, assets)
    w = S._china_cap(w, assets)
    w = S._composite_top2_cap(w, assets, scores)

f = S._forecasts(scores, assets)

print("\n=== TARGET WEIGHTS (2033-03-25 proposal) ===")
tot = 0.0
for a in sorted(assets, key=lambda x: -w[x]):
    print(f"{a:10s} w={w[a]*100:6.2f}%  score={scores[a]:.4f}  fcast={f[a]*100:+.2f}%")
    tot += w[a]
print("sum:", round(tot, 6))

crypto = w.get("BTC", 0) + w.get("ETH", 0)
comm = w.get("WTI", 0) + w.get("COPPER", 0)
cn = w.get("000300.SH", 0) + w.get("000688.SH", 0)
top2 = sorted(w.values(), reverse=True)[:2]
print(f"BTC+ETH={crypto*100:.2f}% (cap 12) WTI+COPPER={comm*100:.2f}% (cap 12) "
      f"000300+000688={cn*100:.2f}% (cap 12) top2={[round(x*100,2) for x in top2]} (cap 9.0)")

# factor value diagnostics
print("\n=== FACTOR VALUES (cross-sectional) ===")
for fid, wgt, direction in S.FACTORS:
    vals = S._factor_values(frames, fid, cur)
    valid = {a: v for a, v in vals.items() if v is not None}
    print(f"{fid} (w={wgt}, dir={direction:+d}) n={len(valid)}")
    for a, v in sorted(valid.items(), key=lambda kv: -kv[1])[:5]:
        print(f"   top {a}: {v:.4f}")
    for a, v in sorted(valid.items(), key=lambda kv: kv[1])[:3]:
        print(f"   low {a}: {v:.4f}")

# VIX check
vix = S._vix_close(cur)
if vix is not None and len(vix):
    last = vix.iloc[-1]
    v20 = vix.iloc[-21] if len(vix) > 21 else None
    v60 = vix.iloc[-61] if len(vix) > 61 else None
    print("\nVIX last:", round(float(last), 2),
          "20d ago:", round(float(v20), 2) if v20 is not None else None,
          "60d ago:", round(float(v60), 2) if v60 is not None else None)
