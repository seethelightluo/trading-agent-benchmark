import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=cl.pct_change()
# Up-day dominance: signed absolute-return imbalance over trailing 20 sessions, lagged.
# Distinguishes persistent low-magnitude gains from a few large jumps.
f=((r.clip(lower=0).rolling(20,min_periods=16).sum()-(-r.clip(upper=0)).rolling(20,min_periods=16).sum())/(r.abs().rolling(20,min_periods=16).sum()+1e-12)).shift(1)
res={h:[] for h in [1,3,5,10]}; ns=[]
for dt in cl.index:
 for h in res:
  fut=cl.shift(-h).loc[dt]/cl.loc[dt]-1
  z=pd.concat([f.loc[dt],fut],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): res[h].append(q)
for h,a in res.items():
 q=pd.Series(a); print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
valid=[]
for dt in cl.index:
 z=pd.concat([f.loc[dt],(cl.shift(-5).loc[dt]/cl.loc[dt]-1)],axis=1).dropna(); valid.append(len(z))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',np.mean(valid),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'dates_ge8',sum(np.array(valid)>=8))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20280323_upday_dominance20_signal.csv',index=False)
