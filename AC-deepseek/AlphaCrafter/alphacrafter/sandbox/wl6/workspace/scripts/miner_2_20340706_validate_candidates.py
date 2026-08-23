"""miner_2 2034-07-06: validate candidate factors over full sample.
Methodology calibrated to beta_vix_60d_neg artifact (ICIR = mean/std of daily ICs).
Min 8 valid instruments per cross-section.
"""
import json, io
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

VISIBLE = "2034-07-05"
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

def load(sym, ddir="../persistent/stock_data", cutoff=VISIBLE):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    return df[df["date"] <= pd.Timestamp(cutoff)].set_index("date")["close"].astype(float).sort_index()

px = pd.DataFrame({s: load(s) for s in TRADABLE})
obs = {s: load(s, "../persistent/index_data/") for s in OBS}
ret = px.pct_change()
fwd = px.shift(-10)/px - 1

frozen = [s for s in TRADABLE if px[s].nunique() <= 1]
print("frozen:", frozen)

def ic_series(fac, fwdmat=fwd, min_valid=8):
    dates, ics = [], []
    common = fac.index.intersection(fwdmat.index)
    for dt in common:
        f = fac.loc[dt]; r = fwdmat.loc[dt]
        m = pd.notna(f) & pd.notna(r) & np.isfinite(f.values) & np.isfinite(r.values)
        m = m & ~f.index.isin(frozen)
        if m.sum() < min_valid: continue
        fv = f[m].values.astype(float); rv = r[m].values.astype(float)
        if np.nanstd(fv) < 1e-12 or np.nanstd(rv) < 1e-12: continue
        rho,_ = spearmanr(fv,rv)
        if np.isfinite(rho): dates.append(dt); ics.append(rho)
    s = pd.Series(ics, index=dates)
    return s

def report(name, fac):
    s = ic_series(fac)
    n = len(s)
    if n == 0:
        print(f"{name}: NO IC DATES"); return None
    mu, sd = s.mean(), s.std()
    icir = mu/sd if sd > 0 else np.nan
    hit = (s>0).mean()
    # decay
    decay = {}
    for H in [1,3,5,10,20]:
        fh = px.shift(-H)/px - 1
        sh = ic_series(fac, fh)
        decay[H] = round(sh.mean(),4) if len(sh) else np.nan
    print(f"{name}: n={n} IC={mu:.4f} ICIR={icir:.4f} hit={hit:.3f} decay={decay}")
    return dict(n=n, ic=mu, icir=icir, hit=hit)

# ---- Candidate factors ----
# 1) 40d momentum skip5
mom40 = px.pct_change(45)   # return over 40d skipping ~5 recent days
mom40 = mom40.replace([np.inf,-np.inf], np.nan)
print("\n=== mom_40d_skip5 (ret 45) ==="); report("mom_40d_skip5", mom40)

# 2) 60d momentum skip5
mom60 = px.pct_change(65)
print("\n=== mom_60d_skip5 (ret 65) ==="); report("mom_60d_skip5", mom60)

# 3) short-term 3d reversal (negative 3d return)
mom3 = px.pct_change(6)
print("\n=== rev_3d (ret 6) ==="); report("rev_3d", -mom3)

# 4) vol-adjusted 40d momentum (return / realized vol)
rv30 = ret.rolling(30).std()*np.sqrt(252)
mom40_av = mom40/rv30
print("\n=== mom40_av (ret45 / rv30) ==="); report("mom40_av", mom40_av)

# 5) USD strength carry: beta to DXY (risk-on) - observe only
dxy_ret = obs['DXY'].pct_change()
beta_dxy = ret.rolling(60).corr(dxy_ret) / dxy_ret.rolling(60).var()
print("\n=== beta_dxy_60d ==="); report("beta_dxy_60d", beta_dxy)