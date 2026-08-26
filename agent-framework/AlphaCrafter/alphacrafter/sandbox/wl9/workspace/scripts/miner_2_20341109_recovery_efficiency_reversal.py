import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5400) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
peak=cl.rolling(120,min_periods=80).max(); dd=cl/peak-1
path=r.abs().rolling(60,min_periods=40).sum(); net=r.rolling(60,min_periods=40).sum(); eff=net.abs()/(path+1e-6)
sig=(-dd*(1-eff)).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,mask=None):
 f=cl.shift(-h)/cl-1; vals=[]; ns=[]; ix=sig.index if mask is None else sig.index[mask]
 for dt in ix:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(ok.sum())
 x=pd.Series(vals); return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()),float(np.mean(ns))
for h in [10,20,40,60,80]: print('H',h,calc(h))
y=sig.index.year
for n,m in [('2020-23',y<=2023),('2024-26',y.isin([2024,2025,2026])),('2027-29',y.isin([2027,2028,2029])),('2030-32',y.isin([2030,2031,2032])),('2033-34',y>=2033)]: print(n,calc(60,m))
print('coverage',float(sig.notna().sum(axis=1).mean()/15),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_2_20341109_recovery_efficiency_reversal_signal.csv',index=False)
