import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-09-08')
def ld(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut]
p=pd.DataFrame({s:ld(s).close for s in U}); r=p.pct_change();
# Trend consistency: fraction of positive daily returns over the prior 20 sessions, centered cross-sectionally.
fac=r.gt(0).rolling(20,min_periods=15).mean(); fac=fac.sub(fac.median(axis=1),axis=0)
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); a=[]; ns=[]; ds=[]
 for d in fac.index:
  z=pd.DataFrame({'f':fac.loc[d],'y':fw.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(d)
 a=np.array(a); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None)
print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270909_trend_consistency_signal.csv',index=False)
