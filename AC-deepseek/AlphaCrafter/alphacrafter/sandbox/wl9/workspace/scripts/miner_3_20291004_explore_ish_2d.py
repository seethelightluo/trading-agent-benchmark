"""Miner3 2029-10-04: Explore interaction-sign factor IS(slo,s) = sign(r_slo) * r_s.

Motivation: existing library is dominated by single-series statics (vol, skew, kurt,
momentum). A factor that conditions a short-horizon return on the sign of a longer-
horizon return (regime-consistent beta) may add orthogonal cross-asset tilt.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import (
    load_panel, fwd_ret_panel, ic_series, rank_turnover, validate,
)

P = load_panel().loc[: "2029-10-03"]
print("panel:", P.shape, P.index.min().date(), "->", P.index.max().date())

FR = {h: fwd_ret_panel(P, h) for h in [1, 2, 3, 5, 10, 20]}


def factor_from(panel, slo, s, tol=0.0):
    out = {}
    for a in panel.columns:
        s_ = panel[a].dropna()
        r_long = s_ / s_.shift(slo) - 1.0
        r_short = s_ / s_.shift(s) - 1.0
        sl_ = np.sign(r_long)
        if tol > 0:
            sl_ = sl_.where(r_long.abs() > tol, 0.0)
        out[a] = sl_ * r_short
    return pd.DataFrame(out).sort_index()


def rank_turnover_v(fv):
    r = fv.rank(axis=1)
    return float(r.diff(10).abs().mean().mean())


results = []
for slo, s in [(20, 5), (40, 5), (60, 5), (20, 10), (60, 10), (5, 5), (120, 10)]:
    for tol in [0.0, 1e-5]:
        try:
            fv = factor_from(P, slo, s, tol)
            fwd10 = FR[10].reindex(fv.index)
            res = validate(fv, fwd10, f"IS_{slo}_{s}_t{tol}")
            res.update({"slo": slo, "s": s, "tol": tol})
            results.append(res)
            print(f"IS_{slo}_{s}_t{tol}: ic={res['ic']} icir={res['icir']} "
                  f"hit={res['ic_hit_ratio']} n={res['n_ic_dates']} "
                  f"cov={res['coverage']} turn={res['turnover_10d_rank']} "
                  f"pass={res['passes']}")
        except Exception as e:
            print(f"IS_{slo}_{s}_t{tol} ERROR {e}")

# decay for best candidate
best = max(results, key=lambda r: (abs(r["ic"] or 0), abs(r["icir"] or 0)))
print("\nBEST:", best["label"])
fv = factor_from(P, best["slo"], best["s"], best["tol"])
dec = {}
for h in [1, 2, 3, 5, 10, 20]:
    ih = ic_series(fv, FR[h].reindex(fv.index))
    dec[str(h)] = round(float(ih.mean()), 4) if len(ih) else None
print("decay IC:", dec)
print("rank_turnover_10d:", round(rank_turnover_v(fv), 3))

print("\nSUMMARY")
for r in results:
    print(f"{r['label']:18s} ic={r['ic']} icir={r['icir']} hit={r['ic_hit_ratio']} pass={r['passes']}")