import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}; cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
r20=cl.pct_change(20); prior=cl.shift(20).pct_change(20); vol=r.rolling(60,min_periods=30).std(); curv=(r20-prior)/(vol+0.005)
breadth=(r.rolling(5).sum()<0).mean(axis=1); stress=breadth>breadth.rolling(120,min_periods=60).quantile(.6)
# Contrarian curvature only in broad stress, trend curvature otherwise; lag state and signal.
sig=(curv*(1-2*stress.astype(float))).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns),'stress_frac',stress.mean())
def calc(h,mask=None):
 f=cl.shift(-h)/cl-1; x=[]; ns=[]; idx=sig.index if mask is None else sig.index[mask]
 for dt in idx:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): x.append(q);ns.append(ok.sum())
 s=pd.Series(x); return len(s),float(s.mean()),float(s.mean()/s.std(ddof=1)),float((s>0).mean()),float(np.mean(ns))
for h in [10,20,40,60,80]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index>=pd.Timestamp('2033-01-01'))]: print(n,calc(20,m))
print('coverage',float(sig.notna().sum(axis=1).mean()/15),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_2_20340817_stress_conditional_curvature_signal.csv',index=False)
