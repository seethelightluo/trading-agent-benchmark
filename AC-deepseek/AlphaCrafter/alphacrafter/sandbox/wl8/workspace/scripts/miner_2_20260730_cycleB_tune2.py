"""miner_2 2026-07-30 -- final tuning: hl_pos window/decay variants to maximize
IC/ICIR while keeping max pooled Spearman |rho| vs library low; then regime
breakdown + Pearson cross-check for the chosen factor.
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

def per_factor_rho(panel, method="spearman"):
    out = {}
    for fid, lp in lib_panels.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            out[fid] = (float("nan"), int(m.sum())); continue
        if method == "spearman":
            rho, _ = spearmanr(a[m], b[m])
        else:
            rho = float(np.corrcoef(a[m], b[m])[0, 1])
        out[fid] = (float(rho), int(m.sum()))
    return out

def full_report(name, panel, do_regime=True):
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
    rho_sp = per_factor_rho(panel, "spearman")
    rho_pe = per_factor_rho(panel, "pearson")
    maxr = max([abs(r) for r, _ in rho_sp.values() if np.isfinite(r)] or [0])
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"=== {name} ===", flush=True)
    print(f"  ic={ic:.4f} icir={icir:.4f} hit={hit:.3f} n={len(ic10)} cov={cov_ad:.3f}/{cov_ge8:.2f} to={to:.2f}", flush=True)
    print(f"  decay={ {str(k): round(v,4) for k,v in decay.items()} }", flush=True)
    for fid in lib_panels:
        rs, _ = rho_sp[fid]; rp, _ = rho_pe[fid]
        print(f"  vs {fid}: spearman={rs:.4f} pearson={rp:.4f}", flush=True)
    print(f"  GATE={'PASS' if ok else 'FAIL'}  max|spearman|={maxr:.4f}", flush=True)
    if do_regime:
        # regime breakdown of horizon-10 IC
        ic_s = ic10
        regs = {
            "2020 (COVID)": ("2020-01-01", "2020-12-31"),
            "2021 (bull)": ("2021-01-01", "2021-12-31"),
            "2022 (tightening)": ("2022-01-01", "2022-12-31"),
            "2023 (recovery)": ("2023-01-01", "2023-12-31"),
            "2024": ("2024-01-01", "2024-12-31"),
            "2025": ("2025-01-01", "2025-12-31"),
            "2026H1": ("2026-01-01", "2026-06-30"),
            "2026 recent": ("2026-04-01", "2026-07-30"),
        }
        print("  regime IC (h=10):", flush=True)
        for rname, (a, b) in regs.items():
            sub = ic_s.loc[(ic_s.index >= a) & (ic_s.index <= b)]
            if len(sub):
                print(f"    {rname}: ic={sub.mean():.4f} icir={sub.mean()/sub.std():.3f} n={len(sub)}", flush=True)
    return ic, icir, maxr

# variants
def hl_pos(c, v, o, h, l, m, w, skip=0):
    hi = h.rolling(w).max().shift(skip)
    lo = l.rolling(w).min().shift(skip)
    rng = (hi - lo).replace(0, np.nan)
    return (c.shift(skip) - lo) / rng

cands = [
    ("hl_pos_150", dict(w=150, skip=0)),
    ("hl_pos_180", dict(w=180, skip=0)),
    ("hl_pos_200", dict(w=200, skip=0)),
    ("hl_pos_150_skip5", dict(w=150, skip=5)),
    ("hl_pos_180_skip5", dict(w=180, skip=5)),
    ("hl_pos_150_skip10", dict(w=150, skip=10)),
]
print(">>> tuning round 2", flush=True)
for name, prm in cands:
    p = factor_panel(hl_pos, close, vol, open_, high, low, macro, **prm)
    full_report(name, p, do_regime=False)
