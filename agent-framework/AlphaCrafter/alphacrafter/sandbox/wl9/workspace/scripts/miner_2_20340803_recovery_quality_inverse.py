import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}; cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
dd=cl/cl.rolling(60,min_periods=40).max()-1; rec=cl.pct_change(10); path=r.rolling(20,min_periods=15).sum().abs()+.001; eff=r.rolling(20,min_periods=15).sum()/path
breadth=r.lt(0).rolling(5,min_periods=5).mean().mean(axis=1); gate=breadth>breadth.rolling(120,min_periods=60).quantile(.60)
sig=-((-dd.clip(upper=0))*rec.clip(lower=0)*(eff.clip(lower=0)+.05)/(r.rolling(20,min_periods=15).std()+.005)).where(gate,0).shift(1)
def calc(h):
 f=cl.shift(-h)/cl-1; x=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): x.append(q); ns.append(ok.sum())
 x=pd.Series(x); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]: print('H',h,calc(h))
print('coverage',sig.notna().sum(axis=1).mean()/15,'active',gate.mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_2_20340803_recovery_quality_inverse_signal.csv',index=False)
