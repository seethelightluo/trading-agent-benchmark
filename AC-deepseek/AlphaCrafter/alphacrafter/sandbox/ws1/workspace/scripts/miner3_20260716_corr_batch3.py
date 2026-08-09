"""miner_3: correlation of batch-3 passing candidates vs existing library factors.

Existing library (4): mom_10d_skip5, mom_120d_skip5, vix_beta_cond_60x20, vol_of_vol20x60.
Gate: max_abs_library_correlation < 0.5 for admission.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_harness import get_panels, evaluate, WATCH

closes, rets, ohlc, macro = get_panels()
vix = macro["VIX"]
dxy = macro["DXY"]
us10y = closes["US10Y"]


def roll_beta(panel, x, n=60):
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    dx = x.diff()
    for a in panel.columns:
        y = panel[a].diff()
        cov = y.rolling(n).cov(dx)
        var = dx.rolling(n).var()
        out[a] = cov / var
    return out


def max_dd(px, n=60):
    def mdd(y):
        if len(y) < n or np.any(~np.isfinite(y)):
            return np.nan
        peak = np.maximum.accumulate(y)
        return float((y / peak - 1.0).min())
    return px.rolling(n).apply(lambda y: mdd(y.values), raw=False)


# ---- existing library factors (recomputed exactly) ----
lib = {
    "mom_10d_skip5": closes.shift(5) / closes.shift(15) - 1.0,
    "mom_120d_skip5": closes.shift(5) / closes.shift(125) - 1.0,
    "vix_beta_cond_60x20": (-roll_beta(closes, vix, 60)).mul((vix / vix.shift(20) - 1.0).reindex(closes.index), axis=0),
    "vol_of_vol20x60": rets.rolling(20).std().rolling(60).std(),
}

# ---- batch-3 gate passing candidates ----
cand = {
    "vix_beta_60d": roll_beta(closes, vix, 60),
    "vix_beta_120d": roll_beta(closes, vix, 120),
    "dxy_beta_60d": roll_beta(closes, dxy, 60),
    "spx_beta_60d": roll_beta(closes, closes["SPX"], 60),
    "us10y_beta_60d": roll_beta(closes, us10y, 60),
    "maxdd_60d": max_dd(closes, 60),
    "mom_z_composite": None,  # build below
    "mom_z_60_180": None,
}

def zscore(px):
    mu = px.mean(axis=1)
    sd = px.std(axis=1)
    return px.sub(mu, axis=0).div(sd, axis=0)

mom20 = closes.pct_change(20)
mom60 = closes.pct_change(60)
mom120 = closes.pct_change(120)
mom180 = closes.pct_change(180)
z20, z60, z120, z180 = (zscore(m) for m in (mom20, mom60, mom120, mom180))
cand["mom_z_composite"] = (z20 + z60 + z120 + z180) / 4.0
cand["mom_z_60_180"] = (z60 + z120 + z180) / 3.0


def mean_rank_corr(a, b):
    """mean cross-sectional spearman rank corr over common dates with >=8 valid."""
    idx = a.index.intersection(b.index)
    cs = []
    for t in idx:
        fa, fb = a.loc[t], b.loc[t]
        mask = fa.notna() & fb.notna() & np.isfinite(fa) & np.isfinite(fb)
        if mask.sum() >= 8:
            r = pd.Series(fa[mask]).corr(pd.Series(fb[mask]), method="spearman")
            if np.isfinite(r):
                cs.append(r)
    return float(np.mean(cs)) if cs else np.nan, len(cs)


print("=== candidate vs library max abs correlation ===")
for cname, cf in cand.items():
    cf = cf.reindex(closes.index)
    row = []
    for lname, lf in lib.items():
        lf = lf.reindex(closes.index)
        v, n = mean_rank_corr(cf, lf)
        row.append((lname, round(v, 4) if np.isfinite(v) else None))
    mx = max((abs(v) for _, v in row if v is not None), default=np.nan)
    status = "ADMISSIBLE" if mx < 0.5 else "REJECT-corr"
    print(f"{cname:22s} max_abs_corr={mx:.4f} {status}")
    for lname, v in row:
        print(f"    vs {lname:24s}: {v}")
