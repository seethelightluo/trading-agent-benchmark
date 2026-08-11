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
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
# Tail-risk-adjusted medium momentum: 60-session return divided by downside deviation over 30 sessions.
down=r.where(r<0,0).rolling(30,min_periods=20).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
f=(cl.pct_change(60)/down.replace(0,np.nan)).shift(1)
fut={h:cl.shift(-h)/cl-1 for h in [1,3,5,10]}
def ev(h,mask=None):
 qs=[];ns=[]
 for dt in cl.index:
  if mask is not None and not bool(mask.loc[dt]): continue
  z=pd.concat([f.loc[dt],fut[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs)
 return len(q),float(q.mean()),float(q.mean()/q.std(ddof=1)*np.sqrt(len(q))),float((q>0).mean()),float(np.mean(ns))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]:print('horizon',h,ev(h))
for name,mask in [('2020-22',pd.Series(cl.index<'2023-01-01',index=cl.index)),('2023-25',pd.Series((cl.index>='2023-01-01')&(cl.index<'2026-01-01'),index=cl.index)),('2026+',pd.Series(cl.index>='2026-01-01',index=cl.index)),('recent120',pd.Series(False,index=cl.index))]:
 if name=='recent120':mask.iloc[-120:]=True
 print('regime',name,ev(5,mask))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20280323_tailrisk_momentum_signal.csv',index=False)
