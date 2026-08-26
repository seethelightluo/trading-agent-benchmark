import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2035-03-18'); px={}
for s in UNIV:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date').set_index('date'); px[s]=d.close.astype(float)
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change(); trend=prices.pct_change(20); path=rets.abs().rolling(20,min_periods=20).sum()
factor=-(trend/path); fwd=prices.shift(-10)/prices-1
rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor=-(20d return)/(20d absolute-return path), cutoff',cut.date())
print('dates',len(r),'avg_n',round(r.n.mean(),2),'min_n',r.n.min())
for label,sub in [('full',r),('recent500',r.tail(500)),('recent180',r.tail(180)),('2020-22',r.loc['2020':'2022']),('2023-26',r.loc['2023':'2026']),('2027-30',r.loc['2027':'2030']),('2031-34',r.loc['2031':'2034']),('2035',r.loc['2035':])]:
 if len(sub)>2:
  m=sub.ic.mean(); sd=sub.ic.std(ddof=1); print(label,'n_dates',len(sub),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((sub.ic>0).mean(),4))
rank=factor.rank(axis=1,pct=True); print('coverage',round(factor.notna().mean().mean(),4),'rank_turnover',round((rank-rank.shift()).abs().mean(axis=1).dropna().mean(),4))
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20350319_efficiency_trend_signal.csv',index=False)
