import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
D={s:load(s) for s in U}; P=pd.concat({s:v for s,v in D.items() if v is not None},axis=1).sort_index(); r=P.pct_change()
# Trend-persistence quality: medium-horizon return, volatility normalized, and fraction of positive sessions.
mom=P.pct_change(20)
vol=r.rolling(40).std()*np.sqrt(252)
cons=(r.gt(0).rolling(20).mean()-0.5)*2
f=(mom/vol*cons).shift(1).replace([np.inf,-np.inf],np.nan)
out='scripts/miner_1_20350524_trend_persistence_quality_signal.csv'; f.to_csv(out,index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); ns.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     ix=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[ix]-prev[ix])))
    prev=rr
 q=np.array(vals); print(f'{h}D dates={len(q)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
fr=P.pct_change(10).shift(-10); vals=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c); dates.append(dt)
q=pd.Series(vals,index=pd.to_datetime(dates));
print('regime_10d',q.loc[q.index<'2028'].mean(),q.loc[(q.index>='2028')&(q.index<'2032')].mean(),q.loc[q.index>='2032'].mean())
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
