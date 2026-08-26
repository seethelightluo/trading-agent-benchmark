import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
ret20=cl.pct_change(20)
# Downside-only standardized shocks: buy assets suffering large negative lagged returns,
# while avoiding positive-return names that can dominate a generic reversal rank.
sig=(-ret20.clip(upper=0)/(vol+0.05)).clip(0,8)
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
gate=(disp>disp.rolling(120,min_periods=60).quantile(.75)).astype(float)
sig=sig.mul(gate,axis=0).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,ix=None):
 f=cl.shift(-h)/cl-1; xs=[];ns=[]
 for dt in sig.index if ix is None else sig.index[ix]:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q);ns.append(ok.sum())
 x=pd.Series(xs); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)
for h in [10,20,40,60]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index.year>=2033)]: print(n,calc(60,m))
print('coverage',sig.notna().sum(axis=1).mean()/15,'active_date_fraction',float((gate>0).mean()),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_3_20340427_downside_dispersion_rebound_signal.csv',index=False)
