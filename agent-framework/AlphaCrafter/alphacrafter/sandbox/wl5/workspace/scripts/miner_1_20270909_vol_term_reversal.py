import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-09-08')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return d.loc[:cut]
D={s:load(s) for s in U}; p=pd.DataFrame({s:D[s].close for s in U}).sort_index(); r=p.pct_change()
# Interpretable term-structure signal: assets with compressed recent volatility
# relative to their 30d volatility are favored (negative short/long ratio).
fac=-(r.rolling(5,min_periods=4).std()/r.rolling(30,min_periods=20).std()).replace([np.inf,-np.inf],np.nan)
allres=[]
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); ic=a.mean(); ir=ic/a.std(ddof=1)
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None)
 allres.append((h,ic,ir))
rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270909_vol_term_reversal_signal.csv',index=False)
