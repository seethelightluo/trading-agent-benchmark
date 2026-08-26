import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,2600)
 except Exception:pass
 if d is None:
  try:d=get_stock_daily_data(s,2600)
  except Exception:pass
 if d is not None:fs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(fs).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
cov=r.rolling(90,min_periods=45).cov(m); var=m.rolling(90,min_periods=45).var(); beta=cov.div(var.replace(0,np.nan),axis=0)
res=(r.rolling(40,min_periods=25).sum()-beta.rolling(40,min_periods=25).mean().mul(m.rolling(40,min_periods=25).sum(),axis=0))
vol=r.rolling(40,min_periods=25).std()*np.sqrt(10); f=(res/vol.replace(0,np.nan)).shift(1)
rows=[]
for i in range(130,len(p)-10):
 z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append([p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')])
a=np.array([x[2] for x in rows]); print('dates',len(a),'avg_n',np.mean([x[1] for x in rows]),'coverage',np.mean([x[1] for x in rows])/len(fs),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'period',p.index.min(),p.index.max()); pd.DataFrame(rows,columns=['date','n','ic']).to_csv('scripts/miner_1_20290618_residual_trend40_signal.csv',index=False)
for start in ['2020-01-01','2023-01-01','2026-01-01','2028-01-01']:
 q=np.array([x[2] for x in rows if str(x[0])[:10]>=start]); print(start,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
