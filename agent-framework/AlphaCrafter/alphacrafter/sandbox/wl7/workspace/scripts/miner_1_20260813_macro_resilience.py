import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change()
# Observation-only macro impulse, fully lagged by using rolling statistics through t.
m=pd.concat({'vix':load('VIX',1).pct_change(),'dxy':load('DXY',1).pct_change(),'fx':load('USDCNY',1).pct_change()},axis=1).reindex(px.index).ffill()
z=(m-m.rolling(60,min_periods=40).mean())/m.rolling(60,min_periods=40).std()
shock=(z.vix+z.dxy+z.fx)/3
# Cross-asset resilience: reward low sensitivity to a macro stress impulse, with robust shrinkage.
fac=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U:
 cov=r[s].rolling(60,min_periods=40).cov(shock)
 var=shock.rolling(60,min_periods=40).var()
 fac[s]=-cov/(var+1e-8)
# neutralize cross-sectional level; rank-equivalent but improves auditability
fac=fac.sub(fac.mean(axis=1),axis=0)
for h in [1,5,10]:
  vals=[]; ns=[]
  for i in range(len(px)-h):
    q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
    if len(q)>=8:
      vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
  a=np.asarray(vals); print('macro_resilience',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sum(ns)/(len(a)*15),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
# recent regime and turnover
q=fac.rank(axis=1,pct=True); turn=(q.diff().abs().mean(axis=1)).dropna().mean(); print('turnover_proxy',round(turn,6))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 vals=[]
 for i in range(len(px)-1):
  if str(px.index[i].year) not in []:
   if lo<=str(px.index[i].year)<=hi:
    x=pd.concat([fac.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
    if len(x)>=8: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 a=np.asarray(vals); print('regime',lo,hi,'dates',len(a),'ICIR',round(a.mean()/a.std(),6) if len(a) else None)
