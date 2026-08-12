import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 try:d=get_index_daily_data(s,days=4100)
 except Exception:d=get_stock_daily_data(s,days=4100)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float).sort_index()
p=pd.DataFrame(P).sort_index(); r=p.pct_change();
# Medium trend with short-horizon reversal: persistent direction but penalize recent overshoot
raw=(p/p.shift(30)-1)-0.7*(p/p.shift(3)-1)
sig=raw.shift(1).rank(axis=1,pct=True); f={h:p.shift(-h)/p-1 for h in [1,5,10,20]}; out={}
for h in f:
 vals=[]; ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt)
 x=pd.Series(vals,index=ds).dropna();out[h]=x;print('%dd dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
 if h==1:
  for yr,g in x.groupby(x.index.year):print('year',yr,'IC=%.6f n=%d'%(g.mean(),len(g)))
print('avg_names',p.notna().sum(axis=1).mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean(axis=1).dropna().mean())
sig.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20310123_trend_reversal_signal.csv',index=False)
