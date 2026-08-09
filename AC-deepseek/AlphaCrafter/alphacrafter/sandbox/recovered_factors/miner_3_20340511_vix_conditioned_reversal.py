import pandas as pd, numpy as np, glob, json, os
from scipy.stats import spearmanr

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
d={}
for a in assets:
    x=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'].rename(a)
    d[a]=x
px=pd.concat(d.values(),axis=1).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
# VIX regime-conditioned short-term reversal: mean-reversion when volatility is rising,
# continuation when volatility is falling. All inputs lagged one completed day.
r=px.pct_change()
vixchg=vix.pct_change(5)
raw=r.rolling(5).sum()
sig=raw.mul(np.where(vixchg.values[:,None]>0,-1.0,1.0))
# neutralize cross-sectional level each date, no future data
sig=sig.sub(sig.mean(axis=1),axis=0)
rows=[]
for h in [1,5,10,20]:
    f=sig
    fr=px.pct_change(h).shift(-h)
    ics=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    a=np.array(ics); rows.append({'h':h,'ic':np.nanmean(a),'icir':np.nanmean(a)/(np.nanstd(a,ddof=1)/np.sqrt(len(a))),'dates':len(a),'mean_n':np.mean(ns)})
# coverage and turnover
valid=sig.notna().sum().sum()/sig.size
ranks=sig.rank(axis=1,pct=True); turnover=(ranks-ranks.shift(10)).abs().mean().mean()
print(json.dumps({'factor':'vix_conditioned_5d_reversal','end':str(px.index.max().date()),'assets':len(assets),'dates':len(px),'coverage':valid,'turnover10':turnover,'results':rows},indent=2))
for yr in [(2020,2023),(2024,2027),(2028,2030),(2031,2034)]:
    mask=(sig.index.year>=yr[0])&(sig.index.year<=yr[1]); f=sig.loc[mask]; fr=px.pct_change(1).shift(-1).loc[mask]; aa=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
      if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    aa=np.array(aa); print('REGIME',yr,'ic',np.mean(aa),'icir',np.mean(aa)/(np.std(aa,ddof=1)/np.sqrt(len(aa))) if len(aa)>1 else np.nan,'dates',len(aa))
