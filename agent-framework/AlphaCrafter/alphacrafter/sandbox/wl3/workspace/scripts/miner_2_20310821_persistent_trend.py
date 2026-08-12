import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,5000)
  except: x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  q=x[['date','close']].drop_duplicates('date'); q.date=pd.to_datetime(q.date); D[s]=q.set_index('date').close
cl=pd.DataFrame(D).sort_index().ffill(); r=cl.pct_change(); rv=r.rolling(20,min_periods=15).std(); cons=r.rolling(10,min_periods=8).mean()/(r.rolling(10,min_periods=8).std()+1e-12)
f=(cl.pct_change(10)*cons/(rv+1e-12)).shift(1)
def calc(h,dates):
 a=[]; ns=[]
 for dt in dates:
  z=pd.concat([f.loc[dt],(cl.shift(-h).loc[dt]/cl.loc[dt]-1)],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):a.append(v);ns.append(len(z))
 q=pd.Series(a); return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),np.mean(ns)
for h in [1,3,5,10,20]:print('H',h,'dates nIC ICIR hit avgN',calc(h,cl.index))
for label,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026-29',(cl.index>='2026-01-01')&(cl.index<'2030-01-01')),('2030+',cl.index>='2030-01-01'),('recent120',np.arange(len(cl))>=len(cl)-120)]:print('REGIME',label,calc(10,cl.index[mask]))
print('shape',cl.shape,'cutoff',cl.index.max().date(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.index.name='date';f.reset_index().to_csv('scripts/miner_2_20310821_persistent_trend_signal.csv',index=False)
