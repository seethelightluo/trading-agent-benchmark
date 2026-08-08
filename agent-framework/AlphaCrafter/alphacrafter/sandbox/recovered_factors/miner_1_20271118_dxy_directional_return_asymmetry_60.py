import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; px={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
 px[a]=pd.to_numeric(d['close'],errors='coerce')
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
d=get_index_daily_data('DXY',5000).copy();d['date']=pd.to_datetime(d['date']);dxy=pd.to_numeric(d.set_index('date').sort_index()['close'],errors='coerce').reindex(P.index).ffill(); dr=dxy.pct_change()
# One interpretable idea: relative recovery in weak-dollar sessions versus resilience in strong-dollar sessions.
# Each value uses only observations ending t: mean r on DXY-down days minus mean r on DXY-up days in 60 sessions.
up=(dr>0).astype(float); down=(dr<0).astype(float); W=60
nu=R.mul(up,axis=0).rolling(W,min_periods=35).sum(); du=up.rolling(W,min_periods=35).sum()
nd=R.mul(down,axis=0).rolling(W,min_periods=35).sum(); dd=down.rolling(W,min_periods=35).sum()
F=nd.div(dd,axis=0)-nu.div(du,axis=0)
# robust rank IC; retain dates with >=8 names
print('candidate=dxy_directional_return_asymmetry_60; visible cutoff',P.index.max().date(),'cells',int(F.notna().sum().sum()),'/',F.size)
turn=F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean();print('mean_rank_turnover',float(turn))
def stats(h):
 y=P.shift(-h).div(P)-1; ics=[]; dates=[]; ns=[]
 for t in F.index:
  z=pd.concat([F.loc[t],y.loc[t]],axis=1).dropna()
  if len(z)>=8:
   ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(t);ns.append(len(z))
 x=pd.Series(ics,index=dates).dropna(); ic=x.mean(); ir=ic/x.std(ddof=1) if x.std(ddof=1)>0 else np.nan
 print('h',h,'dates',len(x),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((x>0).mean(),4),'avg_n',round(np.mean(ns),2))
 for label,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027','2027-01-01','2028-01-01')]:
  q=x.loc[lo:hi];
  if len(q): print(' regime',label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:stats(h)
print('coverage',round(F.notna().mean().mean(),6),'factor_end',F.index.max().date())
