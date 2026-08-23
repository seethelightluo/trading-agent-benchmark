import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; O={}; C={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date'); O[s]=d.open.astype(float); C[s]=d.close.astype(float)
op=pd.DataFrame(O).sort_index().ffill(); cl=pd.DataFrame(C).sort_index().ffill()
# Multi-day close-to-open gap reversal: short-term dislocations in the tradable benchmark set.
gap=(cl/op-1).rolling(3).sum(); vol=cl.pct_change().rolling(20).std()*np.sqrt(252); f=(-gap/vol); f=f.sub(f.median(axis=1),axis=0); y=cl.shift(-1)/cl-1
A=[];N=[];D=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:A.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));N.append(len(z));D.append(dt)
ic=pd.Series(A,index=D).dropna(); print('candidate=gap_reversal_3d_1d','assets',len(C),'rows',len(cl),'dates',len(ic),'mean_n',round(np.mean(N),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(cl),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-15')]:
 q=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
 if len(q):print('regime',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
for h in [1,5,10,20]:
 yy=cl.shift(-h)/cl-1;q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(q),6),len(q))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300516_gap_reversal_1d_signal.csv',index=False)
