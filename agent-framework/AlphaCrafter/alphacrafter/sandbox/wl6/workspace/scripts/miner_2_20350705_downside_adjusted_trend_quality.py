import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<180: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
P=pd.concat(D,axis=1).sort_index(); r=P.pct_change()
# Trend quality: medium-term return, rewarded when upside volatility dominates downside volatility,
# and penalized by current drawdown from the 60-day high. All inputs are lagged one session.
ret=r.rolling(40,min_periods=25).sum()
up=r.clip(lower=0).rolling(30,min_periods=20).std()
down=(-r.clip(upper=0)).rolling(30,min_periods=20).std()
quality=(up+1e-8)/(down+1e-8)
dd=P/P.rolling(60,min_periods=40).max()-1
f=(ret/(r.rolling(30,min_periods=20).std()+1e-8))*quality*(1+dd).clip(lower=0.25).shift(1)
f.to_csv('scripts/miner_2_20350705_downside_adjusted_trend_quality_signal.csv',index_label='date')
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
 q=np.array(vals)
 print(f'{h}D dates={len(q)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/len(U):.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
for a,b in [(2020,2027),(2028,2031),(2032,2035)]:
 fr=P.pct_change(10).shift(-10); q=[]
 for dt in f.index:
  if not (pd.Timestamp(f'{a}-01-01')<=dt<=pd.Timestamp(f'{b}-12-31')): continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q); print(f'regime {a}-{b} dates={len(q)} IC={np.nanmean(q):.8f}')
print('dates',len(f),'instruments',len(P.columns),'cells',int(f.notna().sum().sum()),'overall_coverage',float(f.notna().sum().sum()/(len(f)*len(U))))
