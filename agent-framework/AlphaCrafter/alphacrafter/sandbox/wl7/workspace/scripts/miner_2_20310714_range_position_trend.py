import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): xs.append(d[['date','close']].set_index('date').rename(columns={'close':s}))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=p.pct_change()
# Range-position trend: prior close's location in its trailing 60-session range,
# cross-sectionally demeaned and lagged to avoid using today's close.
lo=p.rolling(60,min_periods=40).min(); hi=p.rolling(60,min_periods=40).max()
f=((p-lo)/(hi-lo).replace(0,np.nan)).sub(((p-lo)/(hi-lo).replace(0,np.nan)).median(axis=1),axis=0).shift(1)
rows=[]
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; ic=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(ic,index=ds).dropna(); ir=q.mean()/q.std() if q.std()>0 else np.nan
 print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(ir,6),'hit',round((q>0).mean(),4))
 if h==1:
  print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'segments',[round(x.mean(),6) for x in np.array_split(q,3)])
out=f.copy(); out.index=out.index.astype(str); out.to_csv('scripts/miner_2_20310714_range_position_trend_signal.csv')
