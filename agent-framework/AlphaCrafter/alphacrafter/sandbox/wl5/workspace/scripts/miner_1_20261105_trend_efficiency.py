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
# Signed trend efficiency: net 20d move divided by path length; rewards persistent directional trends
net=p.pct_change(20); path=r.abs().rolling(20,min_periods=15).sum(); fac=net/(path+1e-12)
# strictly forward returns
for h in [1,5,10]:
 fw=p.pct_change(h,fill_method=None).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in p.index:
  z=pd.DataFrame({'x':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.x,z.y).statistic); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(dates)); print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
 for lab,a,b in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026')]:
  z=q.loc[a:b]; print(lab,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
rank=fac.rank(axis=1,pct=True)
print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20261105_trend_efficiency_signal.csv',index=False)
