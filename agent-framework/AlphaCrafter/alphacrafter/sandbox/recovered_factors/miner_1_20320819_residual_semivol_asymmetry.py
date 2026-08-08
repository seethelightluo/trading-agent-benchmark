"""Miner-1: validate residual downside/upside semivolatility-asymmetry contraction."""
import os
import numpy as np
import pandas as pd

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT='2032-08-18'
# aligned completed-close panel; zero/absent prices are invalid rather than filled
panels=[]
for a in ASSETS:
    x=pd.read_csv(f'../persistent/stock_data/{a}.csv',usecols=['date','close'])
    x['date']=pd.to_datetime(x.date); x=x[x.date<=CUT].set_index('date').close.rename(a)
    panels.append(x)
px=pd.concat(panels,axis=1).sort_index()
r=px.pct_change().replace([np.inf,-np.inf],np.nan)
# preserve each asset's own availability; market is available cross-sectional mean
mkt=r.mean(axis=1,skipna=True)
# rolling beta uses only prior completed observations for signal t
beta=r.rolling(60,min_periods=42).cov(mkt).div(mkt.rolling(60,min_periods=42).var(),axis=0).shift(1)
res=r-beta.mul(mkt,axis=0)
# At t, compare short and long (shifted) downside/upside semi-deviation asymmetry.
# Ratio decreases => downside variability is easing relative to upside variability.
def semis(x,w,down):
    z=x.where(x<0 if down else x>0)
    return np.sqrt(z.pow(2).rolling(w,min_periods=int(w*.7)).mean())
d20,u20=semis(res,20,True),semis(res,20,False)
d60,u60=semis(res,60,True),semis(res,60,False)
a20=np.log((d20+1e-8)/(u20+1e-8)); a60=np.log((d60+1e-8)/(u60+1e-8))
signal=-(a20-a60).shift(1) # higher: reduced relative residual downside risk, fully visibility-safe

def cs_ic(sig, h):
    fwd=px.shift(-h).div(px)-1
    vals=[]; ns=[]; dates=[]
    for d in sig.index:
        z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(d)
    s=pd.Series(vals,index=dates).dropna()
    return s, float(np.mean(ns)) if ns else np.nan
print('FACTOR: inverse residual downside/upside semivolatility-asymmetry contraction (20d vs 60d)')
print('cutoff',CUT,'calendar_dates',len(px),'instruments',len(ASSETS),'valid_cells',int(signal.notna().sum().sum()),'coverage',round(signal.notna().mean().mean(),6))
all_ics={}
for h in [1,5,10,20]:
    s,n=cs_ic(signal,h); all_ics[h]=s
    print(f'h={h} IC={s.mean():.6f} ICIR={(s.mean()/s.std(ddof=1)):.6f} dates={len(s)} hit={(s>0).mean():.6f} mean_n={n:.3f}')
# cross-sectional rank turnover
rank=signal.rank(axis=1,pct=True)
tv=rank.diff().abs().mean(axis=1).dropna()
print(f'turnover={tv.mean():.6f} turnover_dates={len(tv)}')
for label,start,end in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_onward','2027-01-01','2032-12-31')]:
 s=all_ics[10].loc[start:end]
 print(f'regime={label} h10_dates={len(s)} IC={s.mean():.6f} ICIR={(s.mean()/s.std(ddof=1)):.6f} hit={(s>0).mean():.6f}')
# save candidate matrix only locally for potential correlation screening; no persistence admission claim
signal.to_pickle('scripts/miner_1_20320819_candidate_signal.pkl')
"""
