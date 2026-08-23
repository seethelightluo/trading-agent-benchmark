"""miner_2 2034-07-06 batch 2: fix beta_dxy + test defensive/rates/cross-asset factors."""
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

def ic_series(fac, fwdmat=fwd, min_valid=8):
    dates, ics = [], []
    for dt in fac.index.intersection(fwdmat.index):
        f = fac.loc[dt].astype(float); r = fwdmat.loc[dt].astype(float)
        m = pd.notna(f) & pd.notna(r) & np.isfinite(f) & np.isfinite(r)
        m = m & ~f.index.isin(frozen)
        if m.sum() < min_valid: continue
        fv=f[m].values; rv=r[m].values
        if np.nanstd(fv)<1e-12 or np.nanstd(rv)<1e-12: continue
        rho,_=spearmanr(fv,rv)
        if np.isfinite(rho): dates.append(dt); ics.append(rho)
    return pd.Series(ics,index=dates)

def report(name, fac):
    s=ic_series(fac); n=len(s)
    if n==0: print(f"{name}: NO"); return
    mu,sd=s.mean(),s.std(); icir=mu/sd if sd>0 else np.nan
    print(f"{name}: n={n} IC={mu:.4f} ICIR={icir:.4f} hit={(s>0).mean():.3f}")
    return mu,icir

print("\n=== beta_dxy_60d (risk-on) ===")
dxyr=obs['DXY'].pct_change()
beta={}
for s in TRADABLE:
    cov=ret[s].rolling(60).cov(dxyr); var=dxyr.rolling(60).var()
    beta[s]=cov/var
report("beta_dxy_60d", pd.DataFrame(beta))

print("\n=== beta_us10y_60d (rates) ===")
ur=px['US10Y'].pct_change()
beta10={}
for s in TRADABLE:
    cov=ret[s].rolling(60).cov(ur); var=ur.rolling(60).var()
    beta10[s]=cov/var
report("beta_us10y_60d", pd.DataFrame(beta10))

print("\n=== vol_term_20x60 ===")
v20=ret.rolling(20).std(); v60=ret.rolling(60).std()
report("vol_term_20x60", v20/v60)

print("\n=== down_vol_10x60 ===")
def dvol(w):
    neg=(ret<0)*ret
    return neg.rolling(w).std()*np.sqrt(252)
report("down_vol_10x60", dvol(10)/dvol(60))

print("\n=== dd_from_60high ===")
report("dd_from_60high", px/px.rolling(60).max()-1)

print("\n=== mom_20d_skip5 ===")
report("mom_20d_skip5", px.pct_change(25))