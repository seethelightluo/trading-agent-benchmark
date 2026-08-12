import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close','volume']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
raw=pd.concat(rows)
cl=raw.pivot(index='date',columns='symbol',values='close').sort_index().ffill()
vo=raw.pivot(index='date',columns='symbol',values='volume').reindex(cl.index).ffill()
r=cl.pct_change(); lv=np.log(vo.replace(0,np.nan))
# Volume-confirmation reversal: unusually high-volume recent losses are expected to mean-revert;
# unusually high-volume gains are penalized. Lagged 1 session.
vs=(lv-lv.rolling(40,min_periods=20).mean())/lv.rolling(40,min_periods=20).std()
f=(-r.rolling(3,min_periods=3).sum()*vs).shift(1)
def calc(h):
 fut=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(a): vals.append(a);ns.append(len(z))
 q=pd.Series(vals)
 return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',cl.index.max().date(),'total_dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]:
 n,ic,ir,hit,avg=calc(h); print('H',h,'obs',n,'IC',round(ic,7),'ICIR',round(ir,4),'hit',round(hit,4),'avg_n',round(avg,2))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271216_volume_confirmation_signal.csv',index=False)
