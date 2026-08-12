import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
def fetch(s):
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
      try: x=fn(s,days=5000)
      except (FileNotFoundError,KeyError): pass
      if x is not None and len(x): break
    return x[['date','close']].drop_duplicates('date').set_index('date')['close'] if x is not None and len(x) else pd.Series(dtype=float)
p={s:fetch(s) for s in U}; px=pd.DataFrame(p).sort_index().ffill()
r5=px.pct_change(5); vol=px.pct_change().rolling(20).std(); defv=r5[D].median(axis=1)
f=-(r5.sub(defv,axis=0)).div(vol); fr=px.pct_change().shift(-1)
def calc(rr):
 rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
x=calc(fr); print('period',px.index.min(),px.index.max(),'dates',len(x),'universe',len(U),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*len(U))); print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=x.loc[lo:hi]; print(lo,hi,len(q),'%.6f %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan))
for h in [1,3,5,10]:
 q=calc(px.pct_change(h).shift(-h)); print('decay',h,q.ic.mean(),len(q))
f.dropna(how='all').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20320513_defensive_relative_reversal_signal.csv',index=False)
