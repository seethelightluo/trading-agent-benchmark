import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 D[s]=d
# Signed range shock: reverse the day's directional move, scaled by its high-low range
p=pd.DataFrame({s:d.close for s,d in D.items()}); r=p.pct_change()
f=pd.DataFrame({s:-(d.close/d.open-1)*(d.high-d.low)/d.close/(r[s].rolling(20).std()+1e-12) for s,d in D.items()})
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); q=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1:q.append(spearmanr(a.f,a.y).statistic);ns.append(len(a))
 q=np.array(q);print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
y=p.pct_change().shift(-1)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=[]
 for dt in f.loc[lo:hi].index:
  a=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(a)>=8:q.append(spearmanr(a.f,a.y).statistic)
 q=np.array(q);print('regime',lo,hi,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'period',p.index.min(),p.index.max())
