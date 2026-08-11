"""miner_2 2026-07-30 -- decorrelation experiments for gate-passing candidates.
Goal: keep |IC|>=0.007, |ICIR|>=0.084 on horizon-10, while pushing max pooled
Spearman |rho| vs library factors clearly below 0.5 (target <= ~0.40).
Variants tested:
  - hl_pos_120 / hl_pos_150 / hl_pos_180 (range position, longer windows)
  - hl_pos_120_orth: daily cross-sectional residual of hl_pos_120 ranks vs mom ranks
  - gl20_orth: same residualization for gain_loss_20
  - gl20_mom2: gain_loss on de-trended returns (close vs close.shift(3)) to cut mom overlap
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   coverage, turnover_rank, fwd_returns,
                                   ic_series, IC_GATE, ICIR_GATE, ASSETS)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()

# ---------- library panels (recomputed identically to persisted artifacts) ----------
def _beta(ar, mr, win):
    return ar.rolling(win).cov(mr) / mr.rolling(win).var()

def _reindex_macro(c, m):
    return m.reindex(c.index).ffill()

lib_panels = {}
lp = {}
for a in close.columns:
    c = close[a].dropna()
    lp[a] = (c.shift(5) / c.shift(15) - 1.0).reindex(close.index)
lib_panels["mom_10d_skip5"] = pd.DataFrame(lp)
lp = {}
for a in close.columns:
    c = close[a].dropna()
    vs = _reindex_macro(c, vix)
    lp[a] = (-_beta(c.pct_change(), vs.pct_change(), 60) * (vs / vs.shift(20) - 1.0)).reindex(close.index)
lib_panels["vix_beta_cond_60x20"] = pd.DataFrame(lp)
lp = {}
for a in close.columns:
    c = close[a].dropna()
    ys = _reindex_macro(c, macro["US10Y"])
    lp[a] = (_beta(c.pct_change(), ys.pct_change(), 60) * (ys / ys.shift(20) - 1.0)).reindex(close.index)
lib_panels["yield_beta_cond_60x20"] = pd.DataFrame(lp)

def per_factor_rho(panel):
    out = {}
    for fid, lp in lib_panels.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            out[fid] = (float("nan"), int(m.sum())); continue
        rho, _ = spearmanr(a[m], b[m])
        out[fid] = (float(rho), int(m.sum()))
    return out

def orth_vs_mom(panel, mom_panel):
    """Daily cross-sectional rank residualization vs mom ranks."""
    pr = panel.rank(axis=1)
    mr = mom_panel.rank(axis=1)
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for dt in panel.index:
        x = pr.loc[dt].values.astype(float)
        y = mr.loc[dt].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            continue
        xv, yv = x[m], y[m]
        xc, yc = xv - xv.mean(), yv - yv.mean()
        if yc.std() == 0:
            continue
        b = np.dot(xc, yc) / np.dot(yc, yc)
        res = x - b * y
        out.iloc[dt == panel.index] = np.nan
        out.loc[dt, panel.columns] = res
    return out

def report(name, panel):
    cov_ad, cov_ge8 = coverage(panel)
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr)
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic10 = ic_series(panel, fwd_returns(close, 10))
    ic = float(ic10.mean()); icir = float(ic10.mean() / ic10.std())
    hit = float((ic10 > 0).mean()) if ic >= 0 else float((ic10 < 0).mean())
    to = turnover_rank(panel)
    rhos = per_factor_rho(panel)
    maxr = max([abs(r) for r, _ in rhos.values() if np.isfinite(r)] or [0])
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"=== {name} ===", flush=True)
    print(f"  ic={ic:.4f} icir={icir:.4f} hit={hit:.3f} n={len(ic10)} cov={cov_ad:.3f}/{cov_ge8:.2f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(k): round(v,4) for k,v in decay.items()} }", flush=True)
    for fid, (r, nn) in rhos.items():
        print(f"  spearman vs {fid}: {r:.4f} (n={nn})", flush=True)
    print(f"  GATE={'PASS' if ok else 'FAIL'}  max|rho|={maxr:.4f}  {'OK<0.5' if maxr < 0.5 else 'TOO CORRELATED'}", flush=True)
    return ic, icir, maxr

# ---------- factor definitions ----------
def hl_pos(c, v, o, h, l, m, w):
    hi, lo = h.rolling(w).max(), l.rolling(w).min()
    return (c - lo) / (hi - lo)

def gain_loss(c, v, o, h, l, m, w):
    r = c.pct_change()
    pos = r.clip(lower=0).rolling(w).mean()
    neg = (-r.clip(upper=0)).rolling(w).mean()
    return pos / neg.replace(0, np.nan)

def gain_loss_detrend(c, v, o, h, l, m, w):
    r = c / c.shift(3) - 1.0  # 3-day returns: less overlap with 5/10d mom
    pos = r.clip(lower=0).rolling(w).mean()
    neg = (-r.clip(upper=0)).rolling(w).mean()
    return pos / neg.replace(0, np.nan)

print(">>> decorrelation experiments (validation to 2026-07-30)", flush=True)
res = {}
for name, fn, prm in [
    ("hl_pos_120", hl_pos, dict(w=120)),
    ("hl_pos_150", hl_pos, dict(w=150)),
    ("hl_pos_180", hl_pos, dict(w=180)),
    ("gl20_detrend", gain_loss_detrend, dict(w=20)),
    ("gl20_detrend60", gain_loss_detrend, dict(w=60)),
]:
    p = factor_panel(fn, close, vol, open_, high, low, macro, **prm)
    r = report(name, p)
    res[name] = (p, r)

# orthogonalized variants (vs mom_10d_skip5)
for name, fn, prm in [
    ("hl_pos_120", hl_pos, dict(w=120)),
    ("hl_pos_150", hl_pos, dict(w=150)),
    ("gl20_detrend", gain_loss_detrend, dict(w=20)),
]:
    p = factor_panel(fn, close, vol, open_, high, low, macro, **prm)
    po = orth_vs_mom(p, lib_panels["mom_10d_skip5"])
    r = report(name + "_orth", po)
    res[name + "_orth"] = (po, r)
