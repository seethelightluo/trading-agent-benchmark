import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); volu=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).reindex(px.index)
lr=np.log(px).diff(); r10=lr.rolling(10).sum(); rv=lr.rolling(30).std()*np.sqrt(10)+1e-9
# Volume-confirmed medium momentum: normalized 10d trend, strengthened by abnormal volume, lagged one day.
vr=(volu.rolling(5).mean()/(volu.rolling(60).mean()+1e-12)).clip(0.5,2.0)
f=(r10/rv)*(1+0.5*(vr-1)); f=f.sub(f.median(axis=1),axis=0).shift(1)
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index[:-h]):
  q=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(q)>=8: rows.append((dt,len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()),'recent250 %.6f %.6f'%(x.tail(250).ic.mean(),x.tail(250).ic.mean()/x.tail(250).ic.std()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  q=x.loc[a:b].ic
  if len(q): print(a+'-'+b,len(q),'%.6f'%q.mean(),'%.6f'%(q.mean()/q.std()))
rr=f.rank(axis=1,pct=True); print('dates',len(px),'instruments',len(D),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(D))),'turnover %.4f'%(rr.diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300307_volume_confirmed_momentum_signal.csv',index=False)
