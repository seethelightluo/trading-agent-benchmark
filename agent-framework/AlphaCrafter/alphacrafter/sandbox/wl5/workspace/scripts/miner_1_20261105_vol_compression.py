import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-11-04'
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
 x=x.loc[x.index<=END]; r=x.close.pct_change(fill_method=None)
 # volatility compression: prefer assets whose recent realized vol is low relative to their own long vol
 v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
 sig=-(v20/(v60+1e-12)).replace([np.inf,-np.inf],np.nan)
 D[s]=pd.DataFrame({'sig':sig,'r1':x.close.pct_change().shift(-1),'r5':x.close.pct_change(5).shift(-5),'r10':x.close.pct_change(10).shift(-10)})
dates=sorted(set().union(*[set(v.index) for v in D.values()]))
for h in ['r1','r5','r10']:
 vals=[]; ds=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({s:[D[s].at[dt,'sig'] if dt in D[s].index else np.nan,D[s].at[dt,h] if dt in D[s].index else np.nan] for s in U}).T
  z=z.replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z[0].nunique()>1 and z[1].nunique()>1:
   vals.append(spearmanr(z[0],z[1]).statistic);ds.append(dt);ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(ds)); print(h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
 for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-04')]:
  z=q.loc[a:b];print(' regime',a[:4]+'-'+b[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
S=pd.DataFrame({s:D[s].sig for s in U}); print('coverage',round(S.notna().mean().mean(),4),'turnover',round(S.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',S.index.min().date(),S.index.max().date())
out=S.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20261105_vol_compression_signal.csv',index=False)
