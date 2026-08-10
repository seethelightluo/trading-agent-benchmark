import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
 D[s]=x['close'].astype(float)
px=pd.DataFrame(D).sort_index(); r=px.pct_change()
# Volatility-transition factor: low recent volatility relative to medium-term volatility,
# combined with sign of medium-term trend. All inputs end at t, forecast return t+1.
vr=r.rolling(5,min_periods=4).std()/r.rolling(20,min_periods=15).std()
trend=r.rolling(10,min_periods=8).sum()
f=(-np.log(vr.clip(lower=1e-8))*np.sign(trend))
ics=[]; turnovers=[]; ns=[]; dates=[]
for i in range(len(px)-1):
 a=f.iloc[i]; y=r.iloc[i+1]; z=pd.concat([a,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(px.index[i])
  if i:
   prev=f.iloc[i-1].reindex(z.index).dropna(); cur=a.reindex(prev.index).dropna()
   if len(cur): turnovers.append(np.mean(np.sign(cur)!=np.sign(prev.reindex(cur.index))))
ics=np.array(ics); print('dates',len(ics),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0),'turnover',np.mean(turnovers))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=ics[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]; print(lo,hi,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
for h in [5,10]:
 yy=r.rolling(h).sum().shift(-h+1)
 q=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],yy.iloc[i+h-1]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('decay',h,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
# save signal artifact
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20261217_vol_transition_signal.csv',index=False)
