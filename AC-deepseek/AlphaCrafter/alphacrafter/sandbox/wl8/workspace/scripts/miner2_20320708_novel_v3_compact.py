"""
miner2_20320708_novel_v3_compact.py - Compact factor exploration
Explore: A) Risk-adj reversal B) Vol term structure C) Dispersion D) Accel E) Drawup
"""
import numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

W = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
     'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MAC = ['DXY','USDCNY','VIX']
H, MV = 10, 8

print("="*70)
print("MINER2 - Compact Factor Exploration (2032-07-07)")
print("="*70)

# Load
data = {}
for sym in W:
    df = get_stock_daily_data(sym, 600)
    if df is not None and len(df) >= 200: data[sym] = df
print(f"Assets: {len(data)}/15")

m_data = {}
for sym in MAC:
    df = get_index_daily_data(sym, 600)
    if df is not None and len(df) >= 200: m_data[sym] = df

# Common dates
c = None
for sym,df in data.items():
    ds = set(str(d)[:10] for d in df['date'].values)
    c = ds if c is None else c & ds
for sym,df in m_data.items():
    ds = set(str(d)[:10] for d in df['date'].values)
    c &= ds
cd = sorted(c)
print(f"Dates: {len(cd)}, {cd[0]} to {cd[-1]}")

di = {d:i for i,d in enumerate(cd)}
N, A = len(cd), len(data)
assets = sorted(data.keys())
close = np.full((N,A), np.nan)
pct = np.full((N,A), np.nan)
for j,sym in enumerate(assets):
    df = data[sym]
    for _,r in df.iterrows():
        d = str(r['date'])[:10]
        if d in di:
            i=di[d]; close[i,j]=r['close']; p[i,j]=r['pct_change'] if not np.isnan(r['pct_change']) else 0.0

mm = {}
for sym,df in m_data.items():
    arr=np.full(N,np.nan)
    for _,r in df.iterrows():
        d=str(r['date'])[:10]
        if d in di: arr[di[d]]=r['close']
    mm[sym]=arr

# Fwd returns
fwd = np.full((N,A), np.nan)
for j in range(A):
    for i in range(N-H):
        if ~np.isnan(close[i,j]) and ~np.isnan(close[i+H,j]) and close[i,j]>0:
            fwd[i,j]=close[i+H,j]/close[i,j]-1

def ic(fac, fwd_, minv=MV):
    ics=[]
    for t in range(N):
        fv=fac[t]; rv=fwd_[t]
        ok=~(np.isnan(fv)|np.isnan(rv))
        if np.sum(ok)>=minv:
            fvv=fv[ok]; rvv=rv[ok]
            if np.std(fvv)>1e-10 and np.std(rvv)>1e-10:
                icv,_=spearmanr(fvv,rvv)
                if ~np.isnan(icv): ics.append(icv)
    return np.array(ics)

def rep(name, fac, fd=fwd):
    ics=ic(fac,fd)
    if len(ics)<5: print(f"\n{name}: only {len(ics)} obs"); return
    mi=np.mean(ics); mai=np.mean(np.abs(ics)); sd=np.std(ics) or 1e-10
    ir=mi/sd; hit=np.mean(ics>0); p007=np.mean(np.abs(ics)>=0.007); cov=np.mean(~np.isnan(fac))
    gate=(mai>=0.007 and abs(ir)>=0.084)
    print(f"\n--- {name} ---")
    print(f"  Obs={len(ics)} IC={mi:.6f} |IC|={mai:.6f} SD={sd:.6f} ICIR={ir:.4f}")
    print(f"  Hit={hit:.3f} Pass007={p007:.3f} Cov={cov:.4f} GATE={gate}")