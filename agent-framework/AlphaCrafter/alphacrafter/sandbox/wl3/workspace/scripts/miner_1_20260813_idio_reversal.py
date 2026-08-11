import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:'2026-08-12']
r=p.pct_change(); cs=r.median(axis=1); vol=r.rolling('60D',min_periods=15).std(); f=-(r.sub(cs,axis='index'))/(vol+1e-12)
y=p.pct_change().shift(-1)
for h in [1,5,10]:
 yy=p.pct_change(h).shift(-h);q=[];ns=[]
 for d in f.index:
  a=pd.DataFrame({'f':f.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8:q.append(spearmanr(a.f,a.y).statistic);ns.append(len(a))
 q=np.array(q);print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2026-08')]:
 q=[]
 for d in f.loc[a:b].index:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 q=np.array(q);print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
f.to_csv('scripts/miner_1_20260813_idio_reversal_signal.csv')
