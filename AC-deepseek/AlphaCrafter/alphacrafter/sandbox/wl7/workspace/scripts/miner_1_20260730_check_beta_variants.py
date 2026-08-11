"""miner_1: precise re-check of beta_ew_60d and vix_beta_cond_60x20 variants."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import load_panel, load_macro, fwd_returns, MIN_ASSETS, FACTOR_LAST


def ic_series_fast(factor, panel, h):
    fwd = fwd_returns(panel, h)
    dates = factor.index.intersection(fwd.index)
    dates = dates[dates <= pd.Timestamp(FACTOR_LAST)]
    F = factor.reindex(dates).values.astype(float)
    R = fwd.reindex(dates).values.astype(float)
    A = np.argsort(np.argsort(F, axis=1), axis=1).astype(float)
    B = np.argsort(np.argsort(R, axis=1), axis=1).astype(float)
    out, idx = [], []
    for i in range(len(dates)):
        m = np.isfinite(F[i]) & np.isfinite(R[i])
        if int(m.sum()) < MIN_ASSETS:
            continue
        a_, b_ = A[i][m], B[i][m]
        ma, mb = a_.mean(), b_.mean()
        num = float(((a_ - ma) * (b_ - mb)).sum())
        den = float(np.sqrt(((a_ - ma) ** 2).sum() * ((b_ - mb) ** 2).sum()))
        out.append(num / den if den > 0 else 0.0)
        idx.append(dates[i])
    return pd.Series(out, index=idx)


def report(name, fv, panel):
    ic10 = ic_series_fast(fv, panel, 10)
    raw = ic10.mean()
    direction = float(np.sign(raw)) if np.isfinite(raw) and raw != 0 else 1.0
    ic = ic10 * direction
    icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    hit = float((ic > 0).mean())
    gate = abs(ic.mean()) >= 0.007 and abs(icir) >= 0.084
    print(f"{name:28s} rawIC10={raw:+.4f} dir={direction:+.0f} IC10={ic.mean():+.4f} "
          f"ICIR10={icir:+.4f} hit={hit:.3f} n={len(ic)} -> {'PASS' if gate else 'fail'}")


def per_asset(fn):
    def wrapper(panel, *macro_series):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            if macro_series:
                idx = s.index
                for ms in macro_series:
                    idx = idx.intersection(ms.dropna().index)
                s = s.loc[idx]
                args = tuple(ms.reindex(idx) for ms in macro_series)
                cols[a] = fn(s, *args)
            else:
                cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


panel = load_panel()
macro = load_macro()
mkt = panel.pct_change().mean(axis=1)

# beta_ew: market returns reindexed to asset calendar (NO extra pct_change)
bv1 = per_asset(lambda s: (s.pct_change().rolling(60).cov(mkt.reindex(s.index))
                           / mkt.reindex(s.index).rolling(60).var().replace(0, np.nan)))(panel)
# beta_ew: union-calendar computation
rets = panel.pct_change()
bv2 = rets.rolling(60).cov(mkt) / mkt.rolling(60).var().replace(0, np.nan)
# beta_ew flipped-sign variant used by miner_2? stored IC was +0.0565 with direction +1 (high beta good)
print("--- beta_ew_60d variants ---")
report("beta_ew_60d per-asset", bv1, panel)
report("beta_ew_60d union", bv2, panel)

# vix_beta_cond variants
vix = macro["VIX"]
vixr = vix.pct_change()
# A: per-asset aligned (my v3)
va = per_asset(lambda s, m: -((s.pct_change().rolling(60).cov(m.pct_change())
                               / m.pct_change().rolling(60).var().replace(0, np.nan))
                              * (m / m.shift(20) - 1.0)))(panel, vix)
# B: union-calendar (miner_2 library_signals style)
vb = -rets.rolling(60).cov(vixr) / vixr.rolling(60).var() * (vix / vix.shift(20) - 1.0)
print("--- vix_beta_cond_60x20 variants ---")
report("vix_beta per-asset", va, panel)
report("vix_beta union", vb, panel)

# dxy_beta sanity (already clean pass)
dxy = macro["DXY"]
bd = per_asset(lambda s, m: ((s.pct_change().rolling(60).cov(m.pct_change())
                              / m.pct_change().rolling(60).var().replace(0, np.nan))
                             * (m / m.shift(20) - 1.0)))(panel, dxy)
report("dxy_beta_60d per-asset", bd, panel)
