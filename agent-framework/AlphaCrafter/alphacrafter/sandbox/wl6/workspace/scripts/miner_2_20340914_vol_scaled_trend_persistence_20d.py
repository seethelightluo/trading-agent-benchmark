import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: d=fn(s,days=4000)
  except Exception: pass
  if d is not None and len(d)>300: break
 if d is not None:
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index(); D[s]=x.close
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
mom=P.pct_change(20); vol=R.rolling(20).std()*np.sqrt(252); breadth=R.gt(0).rolling(20).mean()
F=(mom/vol)*((breadth-0.5)*2)
rows=[]
for t in P.index:
 f=F.loc[t]; fut=P.shift(-10).loc[t]/P.loc[t]-1; z=pd.concat([f,fut],axis=1).dropna()
 if len(z)>=8: rows.append((t,z.iloc[:,0].corr(z.iloc[:,1]),len(z),z.iloc[:,0].notna().mean()))
a=pd.DataFrame(rows,columns=['date','ic','n','coverage']).set_index('date')
print('assets',len(D),'dates',len(a),'mean_n',a.n.mean(),'coverage',a['coverage'].mean())
print('IC %.8f ICIR %.5f hit %.4f turnover_proxy %.5f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean(),F.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
 q=a.loc[lo:hi].ic; print(lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if q.std()>0 else np.nan)
for h in [5,10,20,40]:
 fut=P.shift(-h)/P-1; rr=[]
 for t in P.index:
  z=pd.concat([F.loc[t],fut.loc[t]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,len(rr),np.nanmean(rr))
out=F.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20340914_vol_scaled_trend_persistence_20d_signal.csv')
