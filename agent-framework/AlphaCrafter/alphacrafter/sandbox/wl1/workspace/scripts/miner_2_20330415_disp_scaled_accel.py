import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-04-14'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); print('shape',px.shape,'idx',px.index.min(),px.index.max(),'dupes',px.index.duplicated().sum())
r=np.log(px).diff(); res=r.sub(r.mean(axis=1),axis=0)
base=-(res.rolling(10,min_periods=8).sum()-res.rolling(30,min_periods=20).sum()/3)/res.rolling(90,min_periods=55).std(); disp=res.std(axis=1); rank=disp.rolling(120,min_periods=60).rank(pct=True); f=(base.mul(0.65+0.70*rank,axis=0)).shift(1); fr=np.log(px.shift(-10)/px)
print('f',f.shape,f.notna().sum().sum(),'fr',fr.notna().sum().sum())
ics=[];ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):ics.append(c);ns.append(len(z))
s=pd.Series(ics); print('dates',len(s),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0))
turn=[]
for i in range(1,len(f)):
 q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
print('turn',np.mean(turn))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c):q.append(c)
 q=pd.Series(q);print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20330415_disp_scaled_accel_signal.csv',index=False);print('rows',len(out))
for h in [5,10,20]:
 fw=np.log(px.shift(-h)/px);x=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,pd.Series(x).mean(),len(x))
