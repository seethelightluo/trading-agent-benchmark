import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,days=5000)
 if d is None:return None
 return d[['date','close','high','low']].drop_duplicates('date').set_index('date')
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.close for s,x in D.items()}).sort_index().ffill()
H=pd.DataFrame({s:x.high for s,x in D.items()}).reindex(C.index).ffill(); L=pd.DataFrame({s:x.low for s,x in D.items()}).reindex(C.index).ffill()
C=C.loc[C.index<=pd.Timestamp('2035-09-12')];H=H.reindex(C.index);L=L.reindex(C.index)
# lagged breakout quality: distance from 60d low/high, signed by 20d trend, normalized by ATR
atr=((H-L)/C).rolling(20,min_periods=15).mean()
trend=C.shift(1)/C.shift(21)-1
hi=C.shift(1).rolling(60,min_periods=40).max(); lo=C.shift(1).rolling(60,min_periods=40).min()
loc=(C.shift(1)-lo)/(hi-lo)
sig=((2*loc-1)*trend.abs()/atr).replace([np.inf,-np.inf],np.nan)
# cross-sectional demean preserves ranking and avoids scale
sig=sig.sub(sig.median(axis=1),axis=0)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20350913_range_breakout_quality_signal.csv',index=False)
for h in [5,10,20,40]:
 f=C.shift(-h)/C-1; vals=[]; ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:
   q=sig.loc[d,ok].corr(f.loc[d,ok],method='spearman')
   if pd.notna(q):vals.append(q);ns.append(ok.sum())
 a=pd.Series(vals);print(f'h={h} dates={len(a)} avg_inst={np.mean(ns):.3f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.8f} hit={(a>0).mean():.4f}')
print('coverage',sig.notna().sum().sum()/(len(sig)*15),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean(),'instruments',len(U),'dates',len(sig),'end',C.index.max().date())
