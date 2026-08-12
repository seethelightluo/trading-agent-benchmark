import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in S:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): P[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
# Cross-asset residual trend: 20d return after removing rolling beta to the equal-weight benchmark, divided by idiosyncratic risk.
beta=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in r:
 beta[s]=r[s].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var().replace(0,np.nan)
idio=r-beta.mul(m,axis=0)
res=(1+idio.clip(-.99,.99)).rolling(20,min_periods=20).apply(np.prod,raw=True)-1
risk=idio.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(res/risk.replace(0,np.nan)).clip(-5,5).shift(1)
fwd=r.shift(-1)
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/len(o)/15)
print('IC %.8f ICIR %.8f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=o.loc[a:b].ic; print(a,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 y=(1+r).rolling(h).apply(np.prod,raw=True).shift(-h); zlist=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:zlist.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(zlist).dropna();print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
print('max_abs_library_correlation unavailable')
