import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-04-25')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date'])
 px[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); p=p.loc[:cut]
r=p.pct_change(); vol=r.rolling(40,min_periods=25).std()*np.sqrt(20)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float).reindex(p.index).ffill()
# Strong-dollar regime: reversal of normalized 20d asset move; weak-dollar regime: continuation.
dxy20=dxy.pct_change(20)
reg=np.where(dxy20>0, -1.0, 1.0)
sig=(p.pct_change(20)/vol).mul(reg,axis=0)
fwd=p.shift(-10)/p-1
rows=[]; date_ic=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); date_ic.append((dt,ic,len(z)))
ics=pd.Series({d:v for d,v,n in date_ic}).dropna(); ns=pd.Series({d:n for d,v,n in date_ic})
print('cut',cut.date(),'dates',len(ics),'meanN',round(ns.mean(),3),'coverage',round(ns.mean()/15,4))
print('IC10',round(ics.mean(),6),'ICIR_ann',round(ics.mean()/ics.std()*np.sqrt(252),6),'hit',round((ics>0).mean(),4),'std',round(ics.std(),6))
for h in [5,10,20]:
 ff=p.shift(-h)/p-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(vals),6),len(vals))
for a,b in [('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-04-25')]:
 q=ics.loc[a:b]; print('regime',a[:4],round(q.mean(),6),len(q))
# turnover based on rank ordering changes
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean(); print('turnover',round(turn,6))
# save recoverable signal artifact
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20340427_dxy_conditioned_reversal_signal.csv')
