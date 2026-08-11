import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000); d=d[d.date<=cut].drop_duplicates('date').set_index('date').sort_index(); P[s]=d.close
p=pd.concat(P,axis=1).sort_index(); r=p.pct_change(fill_method=None); up=r.gt(0).rolling(20,min_periods=15).mean()*2-1; vol=r.rolling(20,min_periods=15).std(); f=(p.pct_change(20,fill_method=None)*up/vol).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 fw=p.pct_change(h,fill_method=None).shift(-h); a=[]; ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8:a.append(q.x.corr(q.y));ns.append(len(q))
 a=pd.Series(a).dropna();print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()))
print('coverage',round(f.notna().sum().sum()/(len(f)*15),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
