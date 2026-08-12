import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change()
bench=r['SPX']; fac=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U:
 cov=r[s].rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var()
 beta=cov/var
 fac[s]=(r[s]-beta*bench).rolling(20,min_periods=15).sum()
# compare daily paper horizon and stability by years
for h in [1,5,10]:
 vals=[]; ns=[]; dates=[]
 for i in range(len(px)-h):
  q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(px.index[i])
 a=np.array(vals); print('residual_momentum',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0),'turnover_pending')
 for y in [2020,2021,2022,2023,2024,2025,2026]:
  b=a[[d.year==y for d in dates]]
  if len(b): print(y,len(b),round(b.mean(),5),round(b.mean()/b.std(),5))
# rank turnover
rank=fac.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna(); print('turnover',turn.mean())
# pooled corr with canonical momentum and reversal
m=r.rolling(20).sum(); rev=-r.rolling(5).sum()
print('corr_momentum',fac.stack().corr(m.stack()),'corr_reversal',fac.stack().corr(rev.stack()))
