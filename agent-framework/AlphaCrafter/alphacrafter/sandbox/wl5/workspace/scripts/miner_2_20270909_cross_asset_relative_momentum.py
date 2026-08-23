import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-09-08')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return d.loc[:cut]
D={s:load(s) for s in U}; p=pd.DataFrame({s:D[s].close for s in U}).sort_index(); r=p.pct_change()
# Cross-asset relative momentum: residualize 20-day return by the same-date cross-sectional median.
# Positive signal favors assets with stronger medium-term relative trend; all inputs are completed-day data.
ret20=p.pct_change(20); fac=ret20.sub(ret20.median(axis=1),axis=0)
rows=[]
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20270909_cross_asset_relative_momentum_signal.csv',index=False)
