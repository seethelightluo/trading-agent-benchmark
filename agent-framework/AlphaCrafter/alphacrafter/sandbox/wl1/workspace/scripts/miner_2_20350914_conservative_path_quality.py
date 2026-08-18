import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is None: x=get_index_daily_data(s,5000)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); d=p.pct_change(); r20=p.pct_change(20); r60=p.pct_change(60)
acc=(r20-r20.median(axis=1).values[:,None])-(r60-r60.median(axis=1).values[:,None])
eff=(r60.abs()/d.abs().rolling(60,min_periods=40).sum()).clip(0,1)
# Less aggressive, capped path-quality amplification; lagged one completed day.
sig=(acc*(0.85+1.15*eff).clip(.85,2.0)).shift(1)
def calc(h, start=None):
 rows=[]
 for dt in sig.index:
  if start is not None and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],p.shift(-h).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 return q
q=calc(10)
print('period',q.index.min(),q.index.max(),'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC10',q.ic.mean(),'ICIRdaily',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]: print('decay',h,calc(h).ic.mean())
for start in ['2020-01-01','2027-01-01','2030-01-01','2033-01-01','2035-01-01']:
 z=calc(10,start); print('window',start,'dates',len(z),'ic',z.ic.mean(),'icir',z.ic.mean()/z.ic.std() if len(z)>1 else np.nan)
sig.to_csv('scripts/miner_2_20350914_conservative_path_quality_signal.csv',index_label='date')
q.to_csv('scripts/miner_2_20350914_conservative_path_quality_ic.csv')
