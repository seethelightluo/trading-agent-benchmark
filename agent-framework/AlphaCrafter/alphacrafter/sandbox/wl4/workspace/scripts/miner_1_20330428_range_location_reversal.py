import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
p=pd.DataFrame(P).reindex(columns=A); r=p.pct_change()
# Contrarian range-location: buy assets near lower part of their 60d range, with volatility normalization.
lo=p.rolling(60,min_periods=40).min(); hi=p.rolling(60,min_periods=40).max()
loc=(p-lo)/(hi-lo+1e-12)
vol=r.rolling(20,min_periods=15).std()
F=(-(loc-.5)/(vol+1e-9)).shift(1)
for h in [5,10,20,30]:
 vals=[]; ns=[]; ds=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],p.shift(-h).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
 if h==10: pd.DataFrame({'date':ds,'n':ns,'ic':vals}).to_csv('scripts/artifacts/miner_1_20330428_range_location_reversal_ic.csv',index=False)
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for n in [260,520,780]:
 q=pd.Series(vals).tail(n);print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
F.to_csv('scripts/artifacts/miner_1_20330428_range_location_reversal_signal.csv')
