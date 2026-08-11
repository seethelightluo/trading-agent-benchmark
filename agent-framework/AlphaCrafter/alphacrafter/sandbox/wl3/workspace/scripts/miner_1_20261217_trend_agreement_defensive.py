import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-12-16'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(30,min_periods=15).std()*np.sqrt(20)
# Defensive low-frequency trend: 20d return, gated by agreement of 5/20/60d trends, volatility scaled.
mom=P.pct_change(20)/(vol+1e-8)
agree=((P.pct_change(5)>0).astype(int)+(P.pct_change(20)>0).astype(int)+(P.pct_change(60)>0).astype(int))
f=(mom*((agree-1)/2)).replace([np.inf,-np.inf],np.nan)
f=f.clip(lower=f.quantile(.1,axis=1),upper=f.quantile(.9,axis=1),axis=0)
def calc(yy, dates=f.index):
 v=[]; ns=[]
 for d in dates:
  z=pd.DataFrame({'f':f.loc[d],'y':yy.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   v.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(v);return q,ns
for label,yy in [('1d',P.pct_change().shift(-1)),('3d',P.pct_change(3).shift(-3)),('5d',P.pct_change(5).shift(-5)),('10d',P.pct_change(10).shift(-10))]:
 q,ns=calc(yy);print(label,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for a,b in [('2020-01','2022-12'),('2023-01','2024-12'),('2025-01','2026-12')]:
 q,ns=calc(P.pct_change().shift(-1),f.loc[a:b].index);print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_1_20261217_trend_agreement_defensive_signal.csv')
