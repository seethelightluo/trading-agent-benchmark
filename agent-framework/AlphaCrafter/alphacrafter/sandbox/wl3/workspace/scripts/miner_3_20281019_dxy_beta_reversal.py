import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
m=pd.read_csv('../persistent/index_data/DXY.csv');m.date=pd.to_datetime(m.date);c='close' if 'close' in m else [x for x in m if x!='date'][0];dr=m.set_index('date')[c].reindex(cl.index).ffill().pct_change()
beta=r.rolling(60).cov(dr).div(dr.rolling(60).var(),axis=0)
# beta-weighted 5d reversal: assets most exposed to dollar get differentiated signal
f=(-r.rolling(5).sum()*beta).shift(1)
def run(h,ix):
 a=[]
 for dt in ix:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z):a.append(z)
 q=pd.Series(a);return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean()
for h in [1,3,5,10]:print('horizon',h,run(h,cl.index))
print('meta',cl.index.max().date(),len(cl),len(cl.columns),f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n,ix in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:print(n,run(5,cl.index[ix]))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20281019_dxy_beta_reversal_signal.csv',index=False)
