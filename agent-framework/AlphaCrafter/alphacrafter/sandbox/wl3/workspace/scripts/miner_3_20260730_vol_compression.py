import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  z=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date');D[s]=z.close
 except: pass
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
# low short/long realized vol, with enough cross-section
f=-(r.rolling(5,min_periods=5).std()/(r.rolling(60,min_periods=30).std()+1e-12))
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in f.index:
  a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: vals.append(spearmanr(a.f,a.y).statistic);ns.append(len(a))
 q=np.array(vals); print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for name,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-07')]:
 q=[]
 y=p.pct_change().shift(-1)
 for d in f.loc[lo:hi].index:
  a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(a)>=8:q.append(spearmanr(a.f,a.y).statistic)
 q=np.array(q);print(name,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'period',p.index.min(),p.index.max())
