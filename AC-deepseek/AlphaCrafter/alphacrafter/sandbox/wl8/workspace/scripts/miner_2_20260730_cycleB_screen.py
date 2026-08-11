"""miner_2 2026-07-30 -- cycle B screen (fast vectorized). New candidate
families orthogonal to effective library (mom_10d_skip5, vix_beta_cond_60x20,
yield_beta_cond_60x20). Admission horizon = 10. Gate |IC|>=0.007, |ICIR|>=0.084.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   coverage, turnover_rank, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets", flush=True)

def _beta(ar, mr, win):
    cov = ar.rolling(win).cov(mr)
    var = mr.rolling(win).var()
    return cov / var

def _reindex_macro(c, m):
    return m.reindex(c.index).ffill()

# ---------------- library panels ----------------
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
    b = _beta(c.pct_change(), vs.pct_change(), 60)
    lp[a] = (-b * (vs / vs.shift(20) - 1.0)).reindex(close.index)
lib_panels["vix_beta_cond_60x20"] = pd.DataFrame(lp)
lp = {}
for a in close.columns:
    c = close[a].dropna()
    ys = _reindex_macro(c, macro["US10Y"])
    b = _beta(c.pct_change(), ys.pct_change(), 60)
    lp[a] = (b * (ys / ys.shift(20) - 1.0)).reindex(close.index)
lib_panels["yield_beta_cond_60x20"] = pd.DataFrame(lp)
print("library panels reconstructed", flush=True)

def lib_corr(panel):
    best = 0.0
    for fid, lp in lib_panels.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            continue
        rho = float(np.corrcoef(a[m], b[m])[0, 1])
        best = max(best, abs(rho))
    return round(best, 4)

# ---------------- forward rank panels (once) ----------------
H = [1, 2, 3, 5, 10, 20]
fwd_rank = {}
for h in H:
    out = {}
    for a in close.columns:
        c = close[a].dropna()
        fr = (c.shift(-h) / c - 1.0).reindex(close.index)
        out[a] = fr
    fdf = pd.DataFrame(out)
    fwd_rank[h] = fdf.rank(axis=1)
print("forward rank panels ready", flush=True)

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

def rv(c, w):
    return c.pct_change().rolling(w).std()

def mean_hl(c, o, h, l, w):
    return ((h - l) / c).rolling(w).mean()

cands = {}
def F(name):
    def deco(fn):
        cands[name] = fn
        return fn
    return deco

@F("vol_ratio_5x60")
def f(c, v, o, h, l, m): return rv(c, 5) / rv(c, 60) - 1.0

@F("vol_ratio_10x60")
def f(c, v, o, h, l, m): return rv(c, 10) / rv(c, 60) - 1.0

@F("vol_ratio_20x60")
def f(c, v, o, h, l, m): return rv(c, 20) / rv(c, 60) - 1.0

@F("range_ratio_5x60")
def f(c, v, o, h, l, m): return mean_hl(c, o, h, l, 5) / mean_hl(c, o, h, l, 60) - 1.0

@F("max_dd_60")
def f(c, v, o, h, l, m): return c / c.rolling(60).max() - 1.0

@F("dist_high_60")
def f(c, v, o, h, l, m): return c / h.rolling(60).max() - 1.0

@F("hl_pos_60")
def f(c, v, o, h, l, m):
    hi, lo = h.rolling(60).max(), l.rolling(60).min()
    return (c - lo) / (hi - lo)

@F("autocorr_20")
def f(c, v, o, h, l, m):
    r = c.pct_change()
    cov = r.rolling(20).cov(r.shift(1))
    var = r.rolling(20).var()
    return cov / var

@F("gain_loss_20")
def f(c, v, o, h, l, m):
    r = c.pct_change()
    pos = r.clip(lower=0).rolling(20).mean()
    neg = (-r.clip(upper=0)).rolling(20).mean()
    return pos / neg.replace(0, np.nan)

@F("skew_20")
def f(c, v, o, h, l, m): return c.pct_change().rolling(20).skew()

@F("vol_adj_mom_20")
def f(c, v, o, h, l, m): return (c / c.shift(20) - 1.0) / rv(c, 20)

@F("vol_adj_mom_60")
def f(c, v, o, h, l, m): return (c / c.shift(60) - 1.0) / rv(c, 60)

@F("vol_trend_5x60")
def f(c, v, o, h, l, m):
    if v is None: return pd.Series(np.nan, index=c.index)
    return v.rolling(5).mean() / v.rolling(60).mean() - 1.0

@F("pv_corr_20")
def f(c, v, o, h, l, m):
    if v is None: return pd.Series(np.nan, index=c.index)
    return c.pct_change().rolling(20).corr(v.pct_change())

@F("oc_ratio_20")
def f(c, v, o, h, l, m):
    intra = (o - c).abs()
    over = (c - c.shift(1)).abs()
    return intra.rolling(20).mean() / over.rolling(20).mean()

@F("body_ratio_60")
def f(c, v, o, h, l, m):
    return (c - o).abs().rolling(60).mean() / (h - l).rolling(60).mean()

@F("upper_shadow_20")
def f(c, v, o, h, l, m):
    upper = (h - np.maximum(o, c))
    return (upper / (h - l).replace(0, np.nan)).rolling(20).mean()

@F("rev_1d")
def f(c, v, o, h, l, m): return -(c.pct_change())

@F("rev_3d_skip1")
def f(c, v, o, h, l, m): return -(c.shift(1) / c.shift(4) - 1.0)

@F("cn10y_beta_cond_60x20")
def f(c, v, o, h, l, m):
    cs = _reindex_macro(c, m["CN10Y"])
    b = _beta(c.pct_change(), cs.pct_change(), 60)
    return b * (cs / cs.shift(20) - 1.0)

@F("mom_10_skip1")
def f(c, v, o, h, l, m): return c.shift(1) / c.shift(11) - 1.0

print(f"{len(cands)} candidates", flush=True)
rows = []
for name, fn in cands.items():
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    cov_ad, cov_ge8 = coverage(panel)
    if cov_ge8 < 0.05:
        print(f"{name:24s} skip cov_ge8={cov_ge8:.3f}", flush=True)
        continue
    ic10 = fast_ic(panel, 10)
    ic = float(ic10.mean()) if len(ic10) else np.nan
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic10 < 0).mean())
    decay = {str(h): round(float(fast_ic(panel, h).mean()), 4) for h in H}
    to = turnover_rank(panel)
    lc = lib_corr(panel)
    gate = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    rows.append((name, ic, icir, hit, len(ic10), cov_ad, cov_ge8, to, lc, gate, decay))
    print(f"{name:24s} ic={ic: .4f} icir={icir: .4f} hit={hit:.3f} n={len(ic10):4d} "
          f"cov={cov_ad:.3f}/{cov_ge8:.2f} to={to:.2f} libcorr={lc:.3f} {'PASS' if gate else 'fail'}", flush=True)

print("\n--- PASSING gate, corr<0.5 ---")
for r in rows:
    if r[9] and r[8] < 0.5:
        print(f"{r[0]:24s} ic={r[1]: .4f} icir={r[2]: .4f} libcorr={r[8]:.3f} decay={r[10]}", flush=True)
