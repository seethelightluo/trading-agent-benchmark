import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-11-18'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); rv=r.rolling(30,min_periods=15).std(); raw=P.pct_change(20)/(rv*np.sqrt(20)+1e-8)
gate=(P.pct_change(5)>0).astype(float); f=(raw*gate).replace([np.inf,-np.inf],np.nan); f=f.clip(lower=f.quantile(.1,axis=1),upper=f.quantile(.9,axis=1),axis=0); y=P.pct_change().shift(-1)
def calc(yy, dates=f.index):
 v=[]; ns=[]
 for d in dates:
  z=pd.DataFrame({'f':f.loc[d],'y':yy.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   v.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 q=np.asarray(v); return q,ns
q,ns=calc(y);print('all dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2026-11')]:
 q,_=calc(y,f.loc[a:b].index); print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [3,5,10]:
 q,_=calc(P.pct_change(h).shift(-h));print('decay',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4),'period',P.index.min().date(),P.index.max().date());f.to_csv('scripts/miner_1_20261119_trend_gate_signal.csv')
