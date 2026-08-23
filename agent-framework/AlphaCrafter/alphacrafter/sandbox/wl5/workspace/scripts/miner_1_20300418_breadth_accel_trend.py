import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-04-17'); base=Path('../persistent/stock_data')
px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').close.astype(float)
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r20=P.pct_change(20); r5=P.pct_change(5)
breadth=(r20>0).sum(axis=1)/r20.notna().sum(axis=1)
# Causal breadth acceleration: rising breadth activates trend; falling breadth activates short reversal.
acc=(breadth-breadth.shift(5)).clip(-1,1)
trend=r20-r20.median(axis=1).values[:,None]
rev=-(r5-r5.median(axis=1).values[:,None])
gate=acc.clip(-1,1)
sig=(gate.values[:,None]*trend+(1-gate.abs()).values[:,None]*rev)
vol=R.rolling(20).std().replace(0,np.nan)
sig=sig/vol
sig=pd.DataFrame(sig,index=P.index,columns=U)
rows=[]
for h in [5,10,20]:
 a=[]; cov=[]
 for i in range(len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));cov.append(len(z)/15)
 a=np.array(a); rows.append((h,len(a),float(np.nanmean(a)),float(np.nanmean(a)/np.nanstd(a,ddof=1)),float(np.mean(a>0)),float(np.mean(cov))))
rank=sig.rank(axis=1,pct=True); turn=float((rank.diff().abs().sum(axis=1)/15).mean())
print('cutoff',end.date(),'dates',len(P),'instruments',P.shape[1],'rows',rows,'turnover',turn,'coverage',float(sig.notna().mean().mean()))
for name,lo,hi in [('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029-now','2029-01-01','2030-04-17')]:
 a=[]
 for i,t in enumerate(P.index[:-20]):
  if pd.Timestamp(lo)<=t<=pd.Timestamp(hi):
   z=pd.concat([sig.iloc[i],P.iloc[i+20]/P.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a); print(name,len(a),float(np.nanmean(a)),float(np.nanmean(a)/np.nanstd(a,ddof=1)) if len(a)>1 else np.nan)
out=pd.DataFrame({'date':np.repeat(P.index,len(U)),'symbol':U*len(P),'signal':sig.to_numpy().ravel()})
out.to_csv('scripts/miner_1_20300418_breadth_accel_trend_signal.csv',index=False)
