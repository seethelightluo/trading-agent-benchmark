import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); A=P.to_numpy(float); R=pd.DataFrame(A).pct_change().to_numpy(); dates=P.index
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); vc=v.set_index('date').close.reindex(dates).ffill().to_numpy(); vm=pd.Series(vc).rolling(60,min_periods=40).mean().to_numpy(); vs=pd.Series(vc).rolling(60,min_periods=40).std().to_numpy(); vz=(vc-vm)/vs
sh=pd.DataFrame(R).rolling(5,min_periods=5).sum().to_numpy()/pd.DataFrame(R).rolling(20,min_periods=15).std().to_numpy(); F=(-sh*(1+.75*np.clip(vz[:,None],0,2)))*1; F=np.roll(F,1,axis=0); F[0]=np.nan
def run(h):
 out=[]; ns=[]
 for i in range(len(A)-h):
  x=F[i]; y=A[i+h]/A[i]-1; ok=np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=8: out.append(np.corrcoef(x[ok],y[ok])[0,1]); ns.append(ok.sum())
 q=np.array(out); return q.mean(),q.mean()/q.std(),len(q),np.mean(ns)
r=[]
for i in range(len(A)-10):
 x=F[i]; y=A[i+10]/A[i]-1; ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()>=8:r.append((dates[i],np.corrcoef(x[ok],y[ok])[0,1]))
r=pd.DataFrame(r,columns=['date','ic']); print('instruments',A.shape[1],'dates',len(r),'avgN',run(10)[3],'coverage',np.isfinite(F).all(2).mean()); print('H10',run(10),'hit',(r.ic>0).mean())
for h in [5,20,40,60]:print('H'+str(h),run(h))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=r[(r.date>=a)&(r.date<=b+'-12-31')];
 if len(q):print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
