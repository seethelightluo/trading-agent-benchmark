import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    p=f'../persistent/stock_data/{s}.csv'
    x=pd.read_csv(p)
    x['date']=pd.to_datetime(x['date'])
    x=x.set_index('date')['close'].sort_index().astype(float)
    D[s]=x
px=pd.DataFrame(D).sort_index().loc[:'2029-02-07']
ret=px.pct_change()
# Relative strength: asset's 20-session return relative to contemporaneous cross-sectional median.
raw=px/px.shift(20)-1
fac=raw.sub(raw.median(axis=1),axis=0).shift(1)
print('cutoff',px.index.max().date(),'dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
    f=fac
    fr=px.shift(-h)/px-1
    vals=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    a=np.asarray(vals); ic=np.nanmean(a); sd=np.nanstd(a,ddof=1); icir=ic/sd if sd else np.nan
    print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={icir:.6f} hit={np.mean(a>0):.4f}')
# coverage and rank turnover (daily rank changes)
print('coverage',fac.notna().mean().mean(),'avg cross section',fac.notna().sum(axis=1).mean())
r=fac.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)).mean(); print('rank_turnover',turn)
