import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
# Lower-turnover relative reversal: 10-day losers, inverse-vol scaled and cross-sectional demeaned.
raw=-px.pct_change(10); rel=raw.sub(raw.median(axis=1),axis=0); f=(rel/vol).sub((rel/vol).mean(axis=1),axis=0); y=px.shift(-10)/px-1
A=[];N=[];D=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: A.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); N.append(len(z)); D.append(dt)
ic=pd.Series(A,index=D).dropna();
print('candidate=relative_reversal_10d','assets',len(P),'rows',len(px),'dates',len(ic),'mean_n',round(np.mean(N),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-29')]:
 q=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
 if len(q): print('regime',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(q),6),len(q))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300530_relative_reversal_10d_signal.csv',index=False)
