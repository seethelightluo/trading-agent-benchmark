import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x[['date','close']].drop_duplicates('date').set_index('date')['close']
  except Exception: pass
 return pd.Series(dtype=float)
px=pd.DataFrame({s:fetch(s) for s in U}).sort_index().ffill(); ret=px.pct_change(); r3=px.pct_change(3); vol20=ret.rolling(20).std(); defv=r3[D].median(axis=1)
vr=vol20/(ret.rolling(60).std()+1e-12); breadth=(r3.lt(0).sum(axis=1)/r3.notna().sum(axis=1)); base=-(r3.sub(defv,axis=0)).div(vol20); f=base*(1+0.5*(vr.mean(axis=1)>1.10).astype(float)); f=f.where(breadth>0.55)
def calc(rr):
 out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
def report(rr):
 x=calc(rr); print('period',px.index.min(),px.index.max(),'dates',len(x),'avg_n',round(x.n.mean(),3),'coverage',round(x.n.sum()/(len(x)*len(U)),4),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
  q=x.loc[lo:hi]; print('regime',lo,hi,len(q),'%.6f %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan))
 return x
x=report(px.pct_change(1).shift(-1))
for h in [3,5,10]: print('decay',h,calc(px.pct_change(h).shift(-h)).ic.mean())
f.dropna(how='all').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20320527_voltransition_breadth_signal.csv',index=False)
