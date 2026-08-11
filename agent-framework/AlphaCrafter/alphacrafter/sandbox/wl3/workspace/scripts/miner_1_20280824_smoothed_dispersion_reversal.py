import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
disp=r.std(axis=1); cutoff=disp.rolling(60,min_periods=30).median()
# Three-session mean of lagged, dispersion-conditioned reversal; all inputs precede decision date.
raw=-r.where(disp>cutoff,0)
f=raw.shift(1).rolling(3,min_periods=3).mean()
for h in [1,3,5,10]:
 a=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q)
 q=pd.Series(a);print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:
 a=[]
 for dt in cl.index[mask]:
  z=pd.concat([f.loc[dt],cl.shift(-1).loc[dt]/cl.loc[dt]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',name,'n',len(a),'IC',np.mean(a) if a else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20280824_smoothed_dispersion_reversal_signal.csv',index=False)
