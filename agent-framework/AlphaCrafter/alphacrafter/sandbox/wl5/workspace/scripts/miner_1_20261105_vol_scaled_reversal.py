import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-11-04'
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
 x=x.loc[:END]
 P[s]=x.close
p=pd.DataFrame(P).sort_index(); r=p.pct_change(fill_method=None)
# Candidate: volatility-scaled short-term reversal, using only completed sessions.
vol=r.rolling(20,min_periods=15).std()
f=-(p.pct_change(5,fill_method=None)/vol)
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,5,10]:
  vals=[]; dates=[]; ns=[]
  fw=p.pct_change(h,fill_method=None).shift(-h)
  for dt in p.index:
    z=pd.DataFrame({'x':f.loc[dt], 'y':fw.loc[dt]}).dropna()
    if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
      vals.append(spearmanr(z.x,z.y).statistic); dates.append(dt); ns.append(len(z))
  q=pd.Series(vals,index=pd.to_datetime(dates))
  print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
  for lab,sub in [('2020-22',q.loc[:'2022']),('2023-24',q.loc['2023':'2024']),('2025-26',q.loc['2025':])]:
    print(' regime',lab,'n',len(sub),'IC',round(sub.mean(),6),'ICIR',round(sub.mean()/sub.std(ddof=1),6) if len(sub)>1 else np.nan)
  rows.append((h,q))
ranks=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(ranks.diff().abs().mean().mean(),4),'period',p.index.min().date(),p.index.max().date())
# artifact for admission horizon 1
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20261105_vol_scaled_reversal_signal.csv',index=False)
