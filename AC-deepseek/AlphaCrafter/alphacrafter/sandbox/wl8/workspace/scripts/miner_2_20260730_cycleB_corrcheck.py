"""miner_2 2026-07-30 -- detailed correlation check + variants for passing
candidates (hl_pos_60, gain_loss_20). Computes Spearman rho vs each library
factor and tests variants with different windows/definitions to keep
|rho| < 0.5 with margin while retaining IC/ICIR gate pass.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   coverage, turnover_rank, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()

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
            out[fid] = (float("nan"), int(m.sum()))
            continue
        rho, _ = spearmanr(a[m], b[m])
        out[fid] = (float(rho), int(m.sum()))
    return out

H = [1, 2, 3, 5, 10, 20]
fwd_rank = {}
for h in H:
    out = {}
    for a in close.columns:
        c = close[a].dropna()
        out[a] = (c.shift(-h) / c - 1.0).reindex(close.index)
    fwd_rank[h] = pd.DataFrame(out).rank(axis=1)

def fast_ic(panel, h):
    pr = panel.rank(axis=1)
    fr = fwd_rank[h]
    ics = []
    for dt in panel.index:
        x = pr.loc[dt].values
        y = fr.loc[dt].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8:
            xv, yv = x[m], y[m]
            if xv.std() == 0 or yv.std() == 0:
                continue
            ics.append(float(np.corrcoef(xv, yv)[0, 1]))
    return np.array(ics)

def report(name, panel):
    cov_ad, cov_ge8 = coverage(panel)
    ic10 = fast_ic(panel, 10)
    ic = float(ic10.mean()); icir = float(ic10.mean() / ic10.std())
    hit = float((ic10 > 0).mean()) if ic >= 0 else float((ic10 < 0).mean())
    decay = {str(h): round(float(fast_ic(panel, h).mean()), 4) for h in H}
    to = turnover_rank(panel)
    rhos = per_factor_rho(panel)
    print(f"\n=== {name} ===", flush=True)
    print(f"  ic={ic:.4f} icir={icir:.4f} hit={hit:.3f} n={len(ic10)} cov={cov_ad:.3f}/{cov_ge8:.2f} to={to:.2f}")
    print(f"  decay={decay}")
    for fid, (r, nn) in rhos.items():
        print(f"  spearman rho vs {fid}: {r:.4f} (n={nn})")
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    maxr = max([abs(r) for r, _ in rhos.values() if np.isfinite(r)] or [0])
    print(f"  GATE={'PASS' if ok else 'FAIL'}  max|spearman|={maxr:.4f} -> {'OK<0.5' if maxr < 0.5 else 'TOO CORRELATED'}")

# baseline candidates
def gain_loss(c, v, o, h, l, m, w):
    r = c.pct_change()
    pos = r.clip(lower=0).rolling(w).mean()
    neg = (-r.clip(upper=0)).rolling(w).mean()
    return pos / neg.replace(0, np.nan)

def hl_pos(c, v, o, h, l, m, w):
    hi, lo = h.rolling(w).max(), l.rolling(w).min()
    return (c - lo) / (hi - lo)

cands = {
    "gain_loss_20": (gain_loss, dict(w=20)),
    "gain_loss_60": (gain_loss, dict(w=60)),
    "hl_pos_60": (hl_pos, dict(w=60)),
    "hl_pos_120": (hl_pos, dict(w=120)),
    "hl_pos_90": (hl_pos, dict(w=90)),
}
for name, (fn, prm) in cands.items():
    panel = factor_panel(fn, close, vol, open_, high, low, macro, **prm)
    report(name, panel)
