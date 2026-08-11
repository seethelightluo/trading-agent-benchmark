import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for a in assets:
    p=f'../persistent/stock_data/{a}.csv'
    d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
    frames[a]=d['close'].pct_change()
r=pd.DataFrame(frames).sort_index()
# use only dates with broad cross-section; factor is lagged 1 day
out=[]
for h in [5,10,20]:
    ics=[]; dates=[]; cov=[]
    for i in range(5,len(r)-h):
        dt=r.index[i]
        hist=r.iloc[i-5:i].sum(axis=0,min_count=4) # trailing 5-session returns (sum approximation)
        f=-(hist-hist.mean())
        f=f.shift(0)
        fr=r.iloc[i+1:i+1+h].sum(axis=0,min_count=h-2)
        z=pd.concat([f,fr],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); cov.append(len(z)/15)
    x=np.array(ics); mean=x.mean(); sd=x.std(ddof=1)
    print(f'h={h} dates={len(x)} avg_n={np.mean(np.array(cov)*15):.2f} coverage={np.mean(cov):.4f} IC={mean:.6f} ICIR={mean/sd:.6f} hit={np.mean(x>0):.4f}')
    for start in ['2025-01-01','2026-01-01','2027-01-01']:
        q=x[np.array(dates)>=pd.Timestamp(start)]
        print(' ',start,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# artifact full signal
sig=-(r.rolling(5,min_periods=4).sum()-r.rolling(5,min_periods=4).sum().mean(axis=1))
sig.to_csv('scripts/miner_1_20280113_cross_sectional_reversion_signal.csv',index_label='date')
