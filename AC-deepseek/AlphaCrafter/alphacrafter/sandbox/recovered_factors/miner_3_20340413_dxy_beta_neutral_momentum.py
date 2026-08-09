import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
base='../persistent/stock_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close']
prices=pd.DataFrame(px).sort_index(); dxy=dxy.reindex(prices.index).ffill()
r=prices.pct_change(); dr=dxy.pct_change()
# DXY-beta-neutral medium-term momentum: recent return less rolling beta * DXY return.
beta=r.rolling(60,min_periods=45).cov(dr).div(dr.rolling(60,min_periods=45).var(),axis=0)
asset20=prices.pct_change(20); dxy20=dxy.pct_change(20)
sig=asset20-beta*dxy20.values
# robustly winsorize not needed for ranks; lag one day implicit through forward return
for h in [1,5,10,20]:
  fwd=prices.shift(-h).div(prices)-1
  vals=[]; ns=[]; dates=[]
  for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
  a=np.array(vals); print(f'H{h} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit={np.mean(a>0):.4f} dates={len(a)} meanN={np.mean(ns):.2f}')
  for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2034')]:
   q=a[[str(d)[:4]>=lo and str(d)[:4]<=hi for d in dates]]
   print(' ',lo, f'{np.nanmean(q):.5f}', f'{np.nanmean(q)/np.nanstd(q,ddof=1):.5f}',len(q))
# coverage and 10d rank turnover
print('coverage',sig.notna().sum().sum()/sig.size)
ranks=sig.rank(axis=1,pct=True); print('turn10',np.nanmean((ranks-ranks.shift(10)).abs().mean(axis=1)))
# recent
for h in [1,5,10,20]:
 fwd=prices.shift(-h).div(prices)-1; aa=[]
 for dt in sig.index[sig.index>='2031-01-01']:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('recent',h,np.mean(aa),np.mean(aa)/np.std(aa,ddof=1),len(aa))
