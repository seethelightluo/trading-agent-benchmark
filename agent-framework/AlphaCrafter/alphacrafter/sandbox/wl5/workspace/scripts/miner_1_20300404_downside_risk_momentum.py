import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-04-03'); base=Path('../persistent/stock_data')
px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').close.astype(float)
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
r20=P.pct_change(20); down=R.where(R<0,0.0)
downvol=down.rolling(30).std()*np.sqrt(30)
# Causal asymmetric risk-adjusted momentum: reward medium trend while penalizing downside volatility.
raw=r20/downvol.replace(0,np.nan)
# Cross-sectional relative score removes broad market direction and improves comparability.
sig=raw.sub(raw.median(axis=1),axis=0)
rows=[]
for h in [5,10,20]:
 a=[]; cov=[]
 for i in range(len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15)
 a=np.array(a); rows.append((h,len(a),float(np.nanmean(a)),float(np.nanmean(a)/np.nanstd(a,ddof=1)),float(np.mean(a>0)),float(np.mean(cov))))
rank=sig.rank(axis=1,pct=True); turn=float((rank.diff().abs().sum(axis=1)/15).mean())
print('cutoff',end.date(),'dates',len(P),'instruments',P.shape[1],'rows',rows,'turnover',turn,'coverage',float(sig.notna().mean().mean()))
for name,lo,hi in [('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('recent','2029-01-01','2030-04-03')]:
 a=[]
 for i,t in enumerate(P.index[:-20]):
  if pd.Timestamp(lo)<=t<=pd.Timestamp(hi):
   z=pd.concat([sig.iloc[i],P.iloc[i+20]/P.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a); print(name,len(a),float(np.nanmean(a)) if len(a) else np.nan,float(np.nanmean(a)/np.nanstd(a,ddof=1)) if len(a)>1 else np.nan)
out=pd.DataFrame({'date':np.repeat(P.index,len(U)),'symbol':U*len(P),'signal':sig.to_numpy().ravel()})
out.to_csv('scripts/miner_1_20300404_downside_risk_momentum_signal.csv',index=False)
