import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
# Longer-horizon rebound: lagged 80-session peak drawdown, scaled by 30-session downside intensity and positive participation.
dd=cl/cl.rolling(80,min_periods=50).max()-1
down=(-r.clip(upper=0)).rolling(30,min_periods=15).mean()
up=(r.clip(lower=0)).rolling(30,min_periods=15).mean()
sig=((-dd)/(down+0.001)*(up+0.001)/(down+up+0.002)).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,mask=None):
 f=cl.shift(-h)/cl-1; xs=[]; ns=[]
 dates=sig.index if mask is None else sig.index[mask]
 for dt in dates:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q); ns.append(ok.sum())
 x=pd.Series(xs); return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()),float(np.mean(ns))
for h in [10,20,40,60,80]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index>=pd.Timestamp('2033-01-01'))]: print(n,calc(60,m))
print('coverage',float(sig.notna().sum(axis=1).mean()/15),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_3_20340803_drawdown_rebound_80d_signal.csv',index=False)
