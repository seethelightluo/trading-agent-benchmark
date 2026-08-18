import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-03-17'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(raw).sort_index(); r=np.log(P).diff(); res=r.sub(r.mean(axis=1),axis=0)
disp=res.std(axis=1).rolling(20,min_periods=15).rank(pct=True)
# Short shock exhaustion: reverse recent residual shock only in high-dispersion sessions.
f=-(res.rolling(5,min_periods=4).sum()/res.rolling(20,min_periods=15).std()).mul((disp>0.60).astype(float),axis=0).shift(1)
fr=np.log(P.shift(-10)/P); vals=[]; ns=[]; tr=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
  if pd.notna(c): vals.append(c);ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: tr.append(z.iloc[:,0].rank(pct=True).sub(z.iloc[:,1].rank(pct=True)).abs().mean())
a=np.array(vals); print('candidate short_dispersion_gated_residual_reversal_5d','dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(tr))
for y in ['2024','2025','2026','2027','2028','2029','2030','2031','2032','2033']:
 q=[]
 for i,d in enumerate(f.index):
  if str(d)[:4]==y:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
    if pd.notna(c):q.append(c)
 print(y,len(q),np.mean(q) if q else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20330318_short_dispersion_reversal_signal.csv',index=False)
