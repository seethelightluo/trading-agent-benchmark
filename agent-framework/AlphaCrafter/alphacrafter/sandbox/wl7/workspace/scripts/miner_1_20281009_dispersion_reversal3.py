import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-10-08'); px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d[['date','close']].copy();d.date=pd.to_datetime(d.date);px[s]=d[d.date<=cut].drop_duplicates('date').set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# bounded 3-day reversal, scaled by recent volatility, activated in elevated dispersion
ret3=P/P.shift(3)-1; vol15=r.rolling(15).std(); disp=r.sub(r.median(axis=1),axis=0).abs().median(axis=1)
gate=disp>disp.rolling(252,min_periods=80).quantile(.55)
f=-(ret3/(1+ret3.abs()))/(vol15+1e-8); f=f.where(gate,f*.4)
ics=[];ds=[];cov=[];turn=[]
for i in range(len(P)-1):
 x=f.iloc[i];y=r.iloc[i+1];ok=x.notna()&y.notna()
 if ok.sum()>=8:
  v=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(v):ics.append(v);ds.append(P.index[i]);cov.append(ok.mean())
  if i and (f.iloc[i-1].notna()&x.notna()).sum()>=8:
   a=f.iloc[i-1];z=a.notna()&x.notna();turn.append(1-x[z].rank().corr(a[z].rank()))
ics=np.array(ics);print('dates',len(ics),'avg instruments',np.mean([((f.loc[d].notna()&r.loc[d+pd.Timedelta(days=1)].notna()).sum()) for d in ds if d+pd.Timedelta(days=1) in r.index]));print('coverage',np.mean(cov),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'turnover',np.mean(turn))
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2028')]:
 z=ics[(np.array(ds)>=pd.Timestamp(a+'-01-01'))&(np.array(ds)<=pd.Timestamp(b+'-12-31'))];print('regime',a,b,len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1))
for h in [1,5,10]:
 q=[]
 for i in range(len(P)-h):
  x=f.iloc[i];y=P.iloc[i+h]/P.iloc[i]-1;ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(x[ok].corr(y[ok],method='spearman'))
 q=np.array(q);print('horizon',h,'dates',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20281009_dispersion_reversal3_signal.csv',index=False)
