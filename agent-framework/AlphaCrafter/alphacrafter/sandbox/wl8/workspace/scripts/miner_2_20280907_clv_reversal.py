import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index()
px=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); hi=pd.concat({s:d.high.astype(float) for s,d in D.items()},axis=1).reindex(px.index); lo=pd.concat({s:d.low.astype(float) for s,d in D.items()},axis=1).reindex(px.index)
clv=((px-lo)/(hi-lo+1e-12)-.5).shift(1); f=(-clv.rolling(3).mean()).replace([np.inf,-np.inf],np.nan)
rows=[]
for i,dt in enumerate(px.index):
 for h in [1,3,5,10]:
  if i+h>=len(px):continue
  q=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:rows.append((dt,h,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);print('assets',px.shape[1],'dates',px.index.min(),px.index.max())
for h in [1,3,5,10]:
 z=o[o.h==h];q=z.ic.dropna();print(h,len(q),z.n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
 for lab,m in [('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('2028',z.date.dt.year==2028),('recent280',z.date>=z.date.max()-pd.Timedelta(days=280))]:
  a=z[m].ic.dropna();print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
f.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_2_20280907_clv_reversal_signal.csv',index=False)
