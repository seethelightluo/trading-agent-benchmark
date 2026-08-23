import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in S:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>100: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Residual reversal: remove contemporaneous cross-asset common move from each asset's lagged 10D return.
common=r.mean(axis=1).rolling(10,min_periods=8).sum().shift(1)
asset=P.pct_change(10).shift(1)
resid=asset.sub(common,axis=0)
vol=r.rolling(30,min_periods=20).std().shift(1)
factor=-(resid/(vol*np.sqrt(252)+1e-8))
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]; ds=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z));ds.append(dt)
 a=pd.Series(vals);print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(np.array(ns)/15):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(ds).date()} end={max(ds).date()}')
ranks=factor.rank(axis=1,pct=True);print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
for n,lo,hi in [('early','2020','2025-12-31'),('mid','2026','2030-12-31'),('recent','2031','2035-07-31')]:
 y=P.shift(-10)/P-1;v=[]
 for dt in factor.loc[lo:hi].index:
  z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(v).dropna();print('regime',n,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std())
factor.to_csv('scripts/miner_2_20350816_residual_reversal_signal.csv',index_label='date')
