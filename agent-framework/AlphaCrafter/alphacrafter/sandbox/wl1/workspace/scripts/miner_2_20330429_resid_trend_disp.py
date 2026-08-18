import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-04-28'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); res=r.sub(r.mean(axis=1),axis=0)
# Residual trend persistence, risk adjusted and conditioned on cross-asset dispersion.
trend=res.rolling(20,min_periods=14).sum()/res.rolling(60,min_periods=40).std()
disp=res.std(axis=1); dr=disp.rolling(120,min_periods=60).rank(pct=True)
f=(trend*(0.80+0.40*dr).values[:,None]).shift(1)
fr=np.log(px.shift(-10)/px); ics=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): ics.append(c); ns.append(len(z))
s=pd.Series(ics); print('shape',px.shape,'dates',len(s),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0))
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
print('turn',np.mean(turn))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q);print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
for h in [5,10,20]:
 fw=np.log(px.shift(-h)/px); x=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,pd.Series(x).mean(),len(x))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20330429_resid_trend_disp_signal.csv',index=False);print('rows',len(out))
