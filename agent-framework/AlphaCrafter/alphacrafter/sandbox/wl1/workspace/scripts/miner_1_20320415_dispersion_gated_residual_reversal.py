import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); raw[s]=d.set_index(pd.to_datetime(d.date)).close
px=pd.DataFrame(raw).sort_index(); lr=np.log(px).diff()
r40=np.log(px/px.shift(40)); resid=r40-r40.median(axis=1).values[:,None]
down=lr.where(lr<0).rolling(40,min_periods=20).std()*np.sqrt(40)
tot=lr.rolling(40,min_periods=20).std()*np.sqrt(40)
scale=down.where(down.notna(),tot)
csdisp=lr.std(axis=1)
gate=(csdisp>csdisp.rolling(120,min_periods=60).median()).astype(float)
f=(-resid/(scale+1e-12)).mul(gate,axis=0).shift(1)
y=np.log(px.shift(-10)/px)
vals=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): vals.append((dt,c)); ns.append(len(z))
q=pd.Series(dict(vals),dtype=float); q.index=pd.to_datetime(q.index)
print('factor dispersion_gated_downside_residual_reversal_40d')
print('dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for h in [1,5,10,20]:
 yy=np.log(px.shift(-h)/px); vv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vv.append(c)
 print('decay',h,'IC',round(np.mean(vv),6),'n',len(vv))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2030-12-31'),('2031-01-01','2032-04-15')]:
 r=q[(q.index>=a)&(q.index<=b)]; print('regime',a,b,'dates',len(r),'IC',round(r.mean(),6),'ICIR',round(r.mean()/r.std(),6) if len(r)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320415_dispersion_gated_residual_reversal_signal.csv',index=False)
