import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}).sort_index().ffill(); r=P.pct_change()
# Volatility premium candidate: unusually high recent risk versus its own long baseline predicts reversal; lagged.
rv=r.rolling(20).std(); baseline=rv.rolling(120,min_periods=60).median(); F=-(rv/baseline-1).shift(1)
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1;a=[];ns=[];ds=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):a.append(c);ns.append(len(z));ds.append(d)
 a=pd.Series(a); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/len(syms):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(ds).date()} end={max(ds).date()}')
F.to_csv('scripts/miner_3_20350719_volatility_regime_spread_signal.csv',index_label='date')
