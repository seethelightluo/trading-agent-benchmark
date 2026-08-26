import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_stock_daily_data(s, days=6000)
    except Exception: d=None
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
print('loaded',len(frames),sorted(frames))
p=pd.concat(frames,axis=1).sort_index().ffill(); r=p.pct_change()
ret60=p.pct_change(60); vol60=r.rolling(60).std()*np.sqrt(252); pos=r.gt(0).rolling(60).mean()
rollmax=p.rolling(60).max(); dd=(p/rollmax-1).rolling(60).min().abs()
f=((ret60/(vol60+0.02))*(0.5+pos)*(1-dd)).shift(1)
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a)); dates.append(dt)
 z=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); ic=z.mean(); print(f'H={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ic/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rank=f.rank(axis=1,pct=True); print(f'coverage={f.notna().mean(axis=1).mean():.4f} turnover={rank.diff().abs().mean(axis=1).mean():.6f} instruments={len(frames)} range={p.index.min().date()}:{p.index.max().date()}')
