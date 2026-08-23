import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-11-04'
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
 P[s]=x.close[x.index<=END]
pd0=pd.concat(P,axis=1).sort_index(); r=pd0.pct_change(fill_method=None)
# Directional-consistency momentum: trailing return weighted by fraction of positive sessions.
ret20=pd0.pct_change(20,fill_method=None)
cons=(r>0).rolling(20,min_periods=15).mean()*2-1
fac=ret20*cons
fw=pd0.pct_change(1,fill_method=None).shift(-1)
ics=[]; dates=[]; ns=[]
for dt in fac.index:
 z=pd.DataFrame({'x':fac.loc[dt],'y':fw.loc[dt]}).dropna()
 if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
  ics.append(spearmanr(z.x,z.y).statistic); dates.append(dt); ns.append(len(z))
q=pd.Series(ics,index=pd.to_datetime(dates)); print('factor directional_consistency_momentum'); print('dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(fac.notna().mean().mean(),4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for lab,sub in [('2020-22',q[:'2022']),('2023-24',q['2023':'2024']),('2025-26',q['2025':])]: print(lab,'n',len(sub),'IC',round(sub.mean(),6),'ICIR',round(sub.mean()/sub.std(ddof=1),6))
ranks=fac.rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean().mean(),4),'period',fac.index.min().date(),fac.index.max().date())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20261105_directional_consistency_signal.csv',index=False)
