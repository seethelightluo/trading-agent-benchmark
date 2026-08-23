import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); end=pd.Timestamp('2030-03-06')
px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').close.astype(float)
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); rets=P.pct_change()
# breadth-switch: follow 20d relative trend in positive breadth, fade 5d shocks in negative breadth
r20=P.pct_change(20); r5=P.pct_change(5)
breadth=(r20>0).sum(axis=1)/r20.notna().sum(axis=1)
med20=r20.median(axis=1)
sig=pd.DataFrame(index=P.index,columns=U,dtype=float)
for t in P.index:
 if breadth.loc[t]>=0.5: sig.loc[t]=r20.loc[t]-r20.loc[t].median()
 else: sig.loc[t]=-(r5.loc[t]-r5.loc[t].median())
# volatility normalize, causal
vol=rets.rolling(20).std().replace(0,np.nan)
sig=sig/vol
rows=[]
for h in [5,10,20]:
  ics=[]; cov=[]; dates=[]
  for i,t in enumerate(P.index):
   if i+h>=len(P): continue
   x=sig.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
    ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15); dates.append(t)
  a=np.array(ics); rows.append((h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),np.mean(cov)))
# turnover rank, recent regime
rank=sig.rank(axis=1,pct=True); turn=(rank.diff().abs().sum(axis=1)/15).mean()
print('cutoff',end.date(),'dates',len(P),'instruments',P.shape[1],'rows',rows,'turnover',turn,'overall_coverage',sig.notna().mean().mean())
# yearly/regime 10d
for name,lo,hi in [('2020-24','2020-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('recent','2029-01-01','2030-03-06')]:
 a=[]
 for i,t in enumerate(P.index):
  if not (pd.Timestamp(lo)<=t<=pd.Timestamp(hi)) or i+10>=len(P): continue
  z=pd.concat([sig.iloc[i],(P.iloc[i+10]/P.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a); print(name,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6) if len(a)>1 else np.nan)
# artifact
out=pd.DataFrame({'date':np.repeat(P.index,len(U)),'symbol':U*len(P),'signal':sig.to_numpy().ravel()})
out.to_csv('scripts/miner_1_20300307_breadth_switch_signal.csv',index=False)
