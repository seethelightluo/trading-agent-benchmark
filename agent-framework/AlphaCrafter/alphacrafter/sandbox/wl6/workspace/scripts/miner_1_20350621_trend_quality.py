import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Trend quality: return over 20 days multiplied by fraction of positive sessions, risk scaled.
ret=r.rolling(20,min_periods=15).sum(); hit=(r>0).rolling(20,min_periods=15).mean(); risk=r.rolling(40,min_periods=20).std()
f=(ret*hit/risk).shift(1)
f.to_csv('scripts/miner_1_20350621_trend_quality_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[];ns=[];turn=[];prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c);ns.append(len(z));q=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     ix=q.index.intersection(prev.index);turn.append(np.mean(abs(q[ix]-prev[ix])))
    prev=q
 q=np.array(vals);print(f'{h}D dates={len(q)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turn):.5f}')
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'cell_coverage',f.notna().sum().sum()/(len(f)*15))
