import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
rel20=cl.pct_change(20).sub(cl.pct_change(20).median(axis=1),axis=0)
rel60=cl.pct_change(60).sub(cl.pct_change(60).median(axis=1),axis=0)
vol20=r.rolling(20,min_periods=15).std()
sig=(-(0.6*rel20+0.4*rel60)/(vol20+0.005)).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,start=None):
 f=cl.shift(-h)/cl-1; x=[]; ns=[]
 for dt in sig.index:
  if start and dt<pd.Timestamp(start): continue
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   z=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(z): x.append(z); ns.append(ok.sum())
 x=pd.Series(x); return len(x),x.mean(),x.mean()/x.std(ddof=1), (x>0).mean(),np.mean(ns)
for h in [20,40,60,80]: print('H',h,calc(h))
for n,m in [('2020-23','2020-01-01'),('2024-26','2024-01-01'),('2027-29','2027-01-01'),('2030-32','2030-01-01'),('2033-34','2033-01-01'),('recent-3y','2031-01-01')]: print(n,calc(80,m))
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_1_20341109_crosssectional_breadth_contrarian_80d_signal.csv',index=False)
