"""Trader dry-run for 2028-06-23 decision: compute current v10 proposal and a
v12 variant (top-2 momentum hard cap) WITHOUT calling rebalance_to_weights.
Reads ensemble, fetches data, replicates strategy_hook order. No state mutation.
"""
import json
import numpy as np
import strategy as S

CUR = "2028-06-23"

def main():
    acc = S.get_account_dict()
    assets = list(acc["watch_list"])
    frames = S._fetch(assets)
    scores, used = S._scores(frames, assets, CUR)
    print("factors used:", used, "of", len(S.FACTORS))
    for fid, w, d in S.FACTORS:
        vals = S._factor_values(frames, fid, CUR)
        valid = {a: round(v, 4) for a, v in vals.items() if v is not None}
        print(f"  {fid} w={w} dir={d}: {valid}")

    scores = S._de_rank_value_traps(scores, frames, assets, CUR)
    regime = S._regime(frames, assets)
    print("regime:", regime)

    # --- current holdings weights (market value / total) ---
    tot = acc["total_assets"]
    hold_w = {p["symbol"]: p["market_value"] / tot for p in acc["positions"]}
    print("\ncurrent holdings weights:")
    for a in sorted(assets, key=lambda x: -hold_w.get(x, 0)):
        print(f"  {a:10s} {hold_w.get(a, 0)*100:6.2f}%")

    # --- v10 proposal ---
    w10 = S._weights(scores, assets, regime)
    w10 = S._composite_ma_guard(w10, frames, assets)
    w10 = S._ma_guard(w10, frames, assets, CUR)
    w10 = S._crypto_cap(w10, assets)
    print("\nv10 target (current strategy.py):")
    for a in sorted(assets, key=lambda x: -w10[x]):
        print(f"  {a:10s} {w10[a]*100:6.2f}%")
    print("  sum:", round(sum(w10.values()), 6))

    # --- v12 variant: top-2 momentum hard cap at 6% regardless of MA ---
    mom_vals = S._factor_values(frames, "mom_120d_skip5", CUR)
    mom_rank = S._ranks(mom_vals, assets)
    print("\nmomentum 120d values & ranks:")
    for a in sorted(assets, key=lambda x: -(mom_vals.get(x) or -9)):
        print(f"  {a:10s} val={mom_vals.get(a)} rank={mom_rank[a]:.3f}")

    w12 = dict(w10)
    GUARD_CAP = 0.06
    for _ in range(80):
        penalized = {a for a in assets if w12[a] > GUARD_CAP + 1e-9 and mom_rank[a] >= 0.90}
        if not penalized:
            break
        excess = sum(w12[a] - GUARD_CAP for a in penalized)
        for a in penalized:
            w12[a] = GUARD_CAP
        room = [a for a in assets if w12[a] < GUARD_CAP - 1e-12 and a not in penalized]
        if not room:
            room = [a for a in assets if a not in penalized]
        den = sum(w12[a] for a in room) + 1e-12
        for a in room:
            w12[a] += excess * w12[a] / den
    totw = sum(w12.values())
    w12 = {a: x / totw for a, x in w12.items()}
    w12[assets[-1]] += 1.0 - sum(w12.values())
    print("\nv12 variant target (top-2 momentum cap 6%):")
    for a in sorted(assets, key=lambda x: -w12[x]):
        print(f"  {a:10s} {w12[a]*100:6.2f}%")
    print("  sum:", round(sum(w12.values()), 6))

    # --- turnover & edge estimate for v10 vs v12 ---
    for tag, w in (("v10", w10), ("v12", w12)):
        oneway = sum(abs(w[a] - hold_w.get(a, 0)) for a in assets) / 2.0
        f = S._forecasts(scores, assets)
        edge = sum(f[a] * w[a] for a in assets)
        print(f"\n{tag}: one-way turnover {oneway*100:.2f}% | gross edge {edge*100:.3f}% | gate thr {oneway*0.03*100:.3f}% | pass={edge > oneway*0.0003}")

if __name__ == "__main__":
    main()
