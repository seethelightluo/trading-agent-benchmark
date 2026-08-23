import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-05')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:cut]
D={s:load(s) for s in U}; p=pd.DataFrame({s:D[s].close for s in U}).sort_index(); r=p.pct_change()
# Efficiency-ratio trend: signed 20d displacement divided by total absolute daily movement.
# It rewards persistent directional movement and penalizes choppy paths; completed-day only.
num=r.rolling(20,min_periods=15).sum(); den=r.abs().rolling(20,min_periods=15).sum()
fac=num/den.replace(0,np.nan)
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None)
rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20261105_efficiency_ratio_signal.csv',index=False)
