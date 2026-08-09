import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=END]; R=P.pct_change();
# lagged volatility-contraction breakout: 20d trend, rewarded when recent 5d vol is below its 60d baseline
trend=R.rolling(20,min_periods=20).sum().shift(1); v5=R.rolling(5,min_periods=5).std().shift(1); v60=R.rolling(60,min_periods=40).std().shift(1); F=trend*(v60/(v5+1e-9)).clip(0.5,2.0); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_3_20261217_contraction_breakout_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); a=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): a.append(q);ns.append(len(z))
 q=pd.Series(a); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6)); print('period',F.index.min().date(),F.index.max().date())
