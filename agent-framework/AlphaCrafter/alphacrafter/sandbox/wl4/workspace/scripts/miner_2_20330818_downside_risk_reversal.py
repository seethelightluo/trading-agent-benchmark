import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); mom=P.pct_change(20)
down=r.where(r<0,0).pow(2).rolling(40,min_periods=20).mean().pow(.5)
F=(-mom/(down+1e-8)).shift(1); rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),4),'hit',round((s>0).mean(),4))
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),4),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
import os; os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20330818_downside_risk_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20330818_downside_risk_reversal_signal.csv')
