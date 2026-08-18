import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; P={}
for a in A:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
p=pd.DataFrame(P).reindex(columns=A); r=p.pct_change()
# 20d risk-adjusted momentum, strengthened by persistence of daily direction; all inputs lagged
mom=p.pct_change(20).div(r.rolling(40,min_periods=25).std()+1e-9)
cons=(r.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
F=(mom*cons).shift(1)
stats={}
for h in [5,10,20,30]:
 vals=[]; ns=[]; dates=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],p.shift(-h).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals); stats[h]=s
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
 if h==10: pd.DataFrame({'date':dates,'n':ns,'ic':vals}).to_csv('scripts/artifacts/miner_1_20330428_consistency_volscaled_momentum_ic.csv',index=False)
print('assets',len([x for x in A if x in P]),'coverage',round(F.notna().sum(axis=1).mean()/len(A),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for n in [260,520,780]:
 s=stats[10]; q=s.tail(n); print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
F.to_csv('scripts/artifacts/miner_1_20330428_consistency_volscaled_momentum_signal.csv')
