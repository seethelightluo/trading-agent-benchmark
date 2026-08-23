import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:x=get_index_daily_data(s,days=5000)
 except Exception:x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();r=np.log(p).diff();r10=r.rolling(10).sum();dv=np.sqrt((r.clip(upper=0)**2).rolling(20).mean())+1e-8;b=(r.rolling(60).sum()<0).mean(axis=1)
f=(-r10/dv).mul(1+1.25*np.maximum(b-.40,0),axis=0)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');ic=a.ic.dropna();print('dates',len(a),'mean_n',a.n.mean(),'coverage',a.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-03-31')]:
 q=a.loc[lo:hi].ic.dropna();print('regime',lo,hi,len(q),q.mean())
for h in [1,5,10]:
 rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rr.append(z.f.corr(z.y))
 print('horizon',h,'IC %.6f n %d'%(np.nanmean(rr),len(rr)))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20320401_continuous_breadth_reversal_signal.csv',index=False)
