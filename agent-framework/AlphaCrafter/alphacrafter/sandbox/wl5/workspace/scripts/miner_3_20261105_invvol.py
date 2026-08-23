import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-05')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return d.loc[:cut]
D={s:load(s) for s in U}; p=pd.DataFrame({s:D[s].close for s in U}).sort_index(); r=p.pct_change()
# Cross-sectional defensive quality: inverse recent realized volatility, lagged and asset-local.
for w in [10,20,40,60]:
 fac=1/r.rolling(w,min_periods=max(8,w//2)).std()
 fw=p.pct_change(1).shift(-1); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); print('w',w,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None)
 rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
 if w==20:
  out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20261105_invvol20_signal.csv',index=False)
