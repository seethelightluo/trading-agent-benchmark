import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
end=pd.Timestamp('2026-07-15'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
# Align close panel, and use only t information for regime and t+1 return.
C=pd.concat({s:load(s).close for s in A},axis=1).loc[:end]
R=C.pct_change(5); regime=R.median(axis=1)<0
rows=[]
for s in A:
 d=load(s).loc[:end]; r5=d.close.pct_change(5); f=(-r5).where(regime.reindex(d.index).fillna(False),0.0); fw=d.close.shift(-1)/d.close-1
 rows += [(dt,s,x,y) for dt,x,y in zip(d.index,f,fw) if pd.notna(x) and pd.notna(y)]
x=pd.DataFrame(rows,columns=['date','sym','f','r']); vals=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: vals.append(spearmanr(g.f,g.r).statistic); ns.append(len(g))
ic=np.array(vals); print('idea=breadth_stress_reversal dates',len(ic),'avg_n',np.mean(ns),'coverage',len(x)/(len(A)*len(x.date.unique())),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'active',regime.sum())
for h in [5,10]:
 rows=[]
 for s in A:
  d=load(s).loc[:end]; r5=d.close.pct_change(5); f=(-r5).where(regime.reindex(d.index).fillna(False),0.0); fw=d.close.shift(-h)/d.close-1
  rows += [(dt,a,b) for dt,a,b in zip(d.index,f,fw) if pd.notna(a) and pd.notna(b)]
 q=pd.DataFrame(rows,columns=['date','f','r']); z=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:z.append(spearmanr(g.f,g.r).statistic)
 z=np.array(z);print('h',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
