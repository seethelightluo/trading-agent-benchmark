import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
D={}
for s in U:
 x=load(s)
 if x is not None: D[s]=x
P=pd.concat(D,axis=1).sort_index(); r=P.pct_change()
# In weak breadth regimes, recent winners tend to mean revert; in broad risk-on regimes,
# risk-adjusted momentum is preferred. Breadth is computed cross-sectionally each day.
up=(r.rolling(20,min_periods=15).sum()>0).sum(axis=1)
valid=r.notna().rolling(20,min_periods=15).sum().sum(axis=1)
breadth=up/valid.replace(0,np.nan)
raw=r.rolling(20,min_periods=15).sum()/r.rolling(30,min_periods=20).std()
f=raw.where(breadth.shift(1)>=0.5,-raw).shift(1)
f.to_csv('scripts/miner_2_20350621_breadth_conditioned_risk_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; counts=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); counts.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     ix=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[ix]-prev[ix])))
    prev=rr
 q=np.array(vals)
 print(f'{h}D dates={len(q)} avg_n={np.mean(counts):.3f} coverage={np.mean(counts)/len(U):.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
for a,b in [(2020,2027),(2028,2031),(2032,2035)]:
 fr=P.pct_change(10).shift(-10); q=[]
 for dt in f.index:
  if not (pd.Timestamp(f'{a}-01-01')<=dt<=pd.Timestamp(f'{b}-12-31')): continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q); print(f'regime {a}-{b} dates={len(q)} IC={np.nanmean(q):.8f}')
print('dates',len(f),'instruments',len(P.columns),'cells',int(f.notna().sum().sum()),'overall_coverage',float(f.notna().sum().sum()/(len(f)*len(U))))
