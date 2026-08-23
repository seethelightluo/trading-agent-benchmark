import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-22')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); ret=p.pct_change(20); fac=-(ret-ret.median(axis=1,skipna=True).values[:,None]); fw={h:p.pct_change(h).shift(-h) for h in [1,5,10]}
for h,y in fw.items():
 vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None)
rank=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
fac.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20270923_relative_reversal_signal.csv',index=False)
