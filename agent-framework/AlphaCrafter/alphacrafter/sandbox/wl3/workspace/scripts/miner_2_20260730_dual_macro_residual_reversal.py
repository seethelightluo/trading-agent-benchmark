import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2026-07-15'
def load(p): return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().query('date<=@CUT')
D={s:load('../persistent/stock_data/'+s+'.csv') for s in U}
mac=pd.concat([load('../persistent/index_data/'+q+'.csv').close.pct_change().rename(q) for q in ['DXY','VIX']],axis=1).mean(axis=1).rename('macro')
def fac(x):
 m=mac.reindex(x.index); r=x.close.pct_change(); b=r.rolling(60,min_periods=30).cov(m)/m.rolling(60,min_periods=30).var(); e=r-b*m
 return -e.rolling(3).sum()/(e.rolling(20,min_periods=15).std()*np.sqrt(3))
rows=[]
for s,x in D.items(): rows.append(pd.DataFrame({'date':x.index,'f':fac(x).to_numpy(),'y':(x.close.shift(-1)/x.close-1).to_numpy()}))
a=pd.concat(rows).dropna(); out=[]; ns=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
z=pd.Series(out); alln=sum(len(x) for x in D.values()); valid=sum(fac(x).notna().sum() for x in D.values())
r=pd.concat([fac(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True)
print('idea dual-macro residual 3d; cutoff',CUT,'dates',len(z),'avg_n',np.mean(ns),'coverage',valid/alln,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',r.diff().abs().mean(axis=1).mean())
for h in [3,5,10]:
 b=pd.concat([pd.DataFrame({'date':x.index,'f':fac(x).to_numpy(),'y':(x.close.shift(-h)/x.close-1).to_numpy()}) for x in D.values()]).dropna(); q=[]
 for d,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
