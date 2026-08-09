import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(f); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close
px=pd.concat(D,axis=1).sort_index(); px=px.loc[:'2027-03-23']
r=px.pct_change()
# volatility-normalized short reversal: negative recent 3d return, scaled by 20d realized vol
fac=-(px.pct_change(3))/(r.rolling(20).std()*np.sqrt(20))
ics=[]; rows=[]
for i in range(20,len(px)-1):
 x=fac.iloc[i]; y=px.iloc[i+1]/px.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  ics.append(ic); rows.append((px.index[i],len(z),ic))
a=np.array(ics); print('dates',len(a),'avgN',np.mean([x[1] for x in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for h in [5,10]:
 q=[]
 for i in range(20,len(px)-h):
  z=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
# turnover based on rank/normalized signals
sig=fac.rank(axis=1,pct=True); turnover=(sig.diff().abs().mean(axis=1)/2).dropna();print('turnover',turnover.mean(),'coverage',fac.notna().sum(axis=1).mean()/15)
# artifact latest signals all dates for audit
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270325_volnorm_reversal_signal.csv',index=False)
# regimes
for a0,b0 in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-07-28','2027-03-23')]:
 q=[v for dt,n,v in rows if a0<=str(dt.date())<=b0];print(a0,'-',b0,'n',len(q),'IC',np.mean(q) if q else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
