"""miner_3 (2026-08-21): Sweep V - cross-asset breadth / idiosyncrasy dimensions.

Most existing factors are per-asset (price/vol) or macro-beta (vs SPX/VIX).
Here I build signals from each asset's relationship to the equal-weighted
cross-asset mean return (breadth center), which is a new center of reference:
  - cs_beta_20  : rolling 20d beta of asset return vs equal-weighted cross-asset mean
  - idio_mom_20 : asset 5d momentum minus 5d CS-mean momentum (idiosyncratic drift)
  - idio_vol_20 : idiosyncratic vol (std of CS-mean regression residual) over 20d
  - cs_mom_rank20: cross-sectional rank of 20d momentum (0..1), centered
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes


def rolling_beta(a, m, w=20, minp=12):
    df = pd.concat([a.rename("a"), m.rename("m")], axis=1)
    out = []
    for i in range(len(df)):
        if i < w - 1:
            out.append(np.nan); continue
        sub = df.iloc[i-w+1:i+1]
        mm = sub["m"].to_numpy(); aa = sub["a"].to_numpy()
        fm = np.isfinite(mm) & np.isfinite(aa)
        if fm.sum() < minp or np.nanstd(mm[fm]) == 0:
            out.append(np.nan); continue
        out.append(np.cov(aa[fm], mm[fm])[0, 1] / np.var(mm[fm]))
    return pd.Series(out, index=df.index)


def rolling_resid_std(a, m, w=20, minp=12):
    df = pd.concat([a.rename("a"), m.rename("m")], axis=1)
    out = []
    for i in range(len(df)):
        if i < w - 1:
            out.append(np.nan); continue
        sub = df.iloc[i-w+1:i+1]
        mm = sub["m"].to_numpy(); aa = sub["a"].to_numpy()
        fm = np.isfinite(mm) & np.isfinite(aa)
        if fm.sum() < minp or np.nanstd(mm[fm]) == 0:
            out.append(np.nan); continue
        beta = np.cov(aa[fm], mm[fm])[0, 1] / np.var(mm[fm])
        resid = aa[fm] - beta * mm[fm]
        out.append(np.std(resid))
    return pd.Series(out, index=df.index)


def main():
    closes = load_closes()
    print("assets:", len(closes))
    ret = {a: closes[a].pct_change() for a in closes}
    retf = pd.DataFrame(ret)
    cs_mean = retf.mean(axis=1)
    cs_med = retf.median(axis=1)

    cand = {
        "cs_beta_20": {a: rolling_beta(ret[a], cs_mean, 20) for a in closes},
        "idio_vol_20": {a: rolling_resid_std(ret[a], cs_mean, 20) for a in closes},
    }
    # idiosyncratic momentum vs CS median
    momf = pd.DataFrame({a: (closes[a]/closes[a].shift(5)-1.0) for a in closes})
    mommed = momf.median(axis=1)
    cand["idio_mom_20"] = {a: momf[a] - mommed for a in closes}
    # cross-sectional rank of 20d momentum centered at 0
    momf20 = pd.DataFrame({a: (closes[a]/closes[a].shift(20)-1.0) for a in closes})
    cs_rank = momf20.rank(axis=1, pct=True) - 0.5
    cand["cs_mom_rank20"] = {a: cs_rank[a] for a in closes}

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()