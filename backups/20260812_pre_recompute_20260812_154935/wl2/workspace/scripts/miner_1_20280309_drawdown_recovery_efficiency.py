import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3200)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); lr=np.log(P).diff()
# Drawdown-recovery efficiency: recent return divided by the worst peak-to-trough
# drawdown over the same window. Signal is shifted one completed bar.
ret20=np.log(P).diff(20)
peak=P.rolling(20,min_periods=20).max()
dd=P/peak-1
dd_abs=(-dd).rolling(20,min_periods=20).max()
F=(ret20/(dd_abs+0.02)).shift(1)
Y={h:np.log(P).shift(-h)-np.log(P) for h in [1,3,5,10]}
def calc(y, sl=slice(None)):
 vals=[]; ns=[]
 for dt in F.loc[sl].index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman')); ns.append(len(z))
 x=pd.Series(vals).dropna(); return len(x),float(np.mean(ns)),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean())
print('symbols',len(P.columns),'range',P.index.min(),P.index.max())
for h,y in Y.items(): print('h',h,'dates avgN IC ICIR hit',calc(y))
print('coverage',F.notna().sum(axis=1).mean()/15,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]: print(name,calc(Y[1],sl))
