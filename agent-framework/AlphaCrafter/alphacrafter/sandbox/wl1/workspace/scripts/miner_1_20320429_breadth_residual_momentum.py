import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-28'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
# Candidate: medium-horizon residual momentum, scaled by realized volatility and gated by positive breadth.
# Positive score means continuation of relative strength; all inputs lagged one session.
ret=resid.rolling(60,min_periods=40).sum(); vol=r.rolling(40,min_periods=25).std(); breadth=(r>0).mean(axis=1).rolling(20,min_periods=15).mean()
f=(ret/vol).where(breadth>0.5).shift(1)
fr=np.log(px.shift(-10)/px); vals=[]; ns=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns),'active_frac',float(f.notna().any(axis=1).mean())); print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turns),'coverage',len(s)/len(px))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 v=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v); print(a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
print('decay')
for h in [1,5,10,20]:
 rr=np.log(px.shift(-h)/px); vv=[]
 for d in f.index:
  z=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(z)>=8: vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(h,np.nanmean(vv),len(vv))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320429_breadth_residual_momentum_signal.csv',index=False)
