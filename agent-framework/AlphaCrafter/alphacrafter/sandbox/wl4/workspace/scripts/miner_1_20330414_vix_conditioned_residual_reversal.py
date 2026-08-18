import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; R={}; P={}
for a in A:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float); R[a]=P[a].pct_change()
r=pd.DataFrame(R); p=pd.DataFrame(P)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].astype(float).reindex(r.index).ffill()
vratio=(v/v.rolling(60,min_periods=30).median()).clip(.5,2.0)
res=r.sub(r.mean(axis=1),axis=0); vol=r.rolling(40,min_periods=20).std()*np.sqrt(15)
F=(-(res.rolling(15,min_periods=10).sum()).div(vol+1e-9).mul(0.5+0.5*vratio,axis=0)).shift(1)
allstats={}
for h in [5,10,20,30]:
 vals=[]; ns=[]; dates=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],p.shift(-h).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals); allstats[h]=s
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
 if h==10: pd.DataFrame({'date':dates,'n':ns,'ic':vals}).to_csv('scripts/artifacts/miner_1_20330414_vix_conditioned_residual_reversal_ic.csv',index=False)
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for n in [260,520,780]:
 s=allstats[10]; print('recent',n,'IC',round(s.tail(n).mean(),6),'ICIR',round(s.tail(n).mean()/s.tail(n).std(),6))
F.to_csv('scripts/artifacts/miner_1_20330414_vix_conditioned_residual_reversal_signal.csv')
