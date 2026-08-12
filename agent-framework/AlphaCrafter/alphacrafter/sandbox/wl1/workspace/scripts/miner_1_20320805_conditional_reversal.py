import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-08-04'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); b=r.mean(axis=1); x=r.sub(b,axis=0)
vol=x.rolling(20,min_periods=15).std(); disp=x.std(axis=1); high=(disp>disp.rolling(120,min_periods=60).median()).shift(1)
# Short-horizon residual reversal, conditioned on elevated dispersion and benchmark weakness, with stable volatility scaling.
f=(-x.rolling(5,min_periods=5).sum()/vol.rolling(60,min_periods=30).mean()).shift(1).where((high & (b.rolling(10,min_periods=8).sum()<0)).fillna(False),np.nan)
fr=np.log(px.shift(-10)/px); ic=[]; ns=[]; tr=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:tr.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ic);print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns),'active',f.notna().any(axis=1).mean());print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(tr))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032')]:
 v=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v);print(a,b,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20320805_conditional_reversal_signal.csv',index=False)
