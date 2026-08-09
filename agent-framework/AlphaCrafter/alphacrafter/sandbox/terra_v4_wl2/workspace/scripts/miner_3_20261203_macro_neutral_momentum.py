import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-03')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index();p.columns=U;r=p.pct_change()
mac=[]
for n in ['DXY','VIX']:
 x=pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut].pct_change().reindex(r.index).ffill();mac.append(x)
m=pd.concat(mac,axis=1);m.columns=['dxy','vix']
for w in [20,60,120]:
 for look in [1,3,5]:
  f=pd.DataFrame(index=r.index,columns=U,dtype=float)
  for i in range(w+look,len(r)):
   X=m.iloc[i-w:i].dropna(); Y=r.iloc[i-w:i].reindex(X.index)
   ok=Y.notna().all(axis=1); X=X.loc[ok];Y=Y.loc[ok]
   if len(X)<w//2: continue
   B=np.linalg.lstsq(np.c_[np.ones(len(X)),X],Y,rcond=None)[0]
   rr=r.iloc[i-look:i].sum(); mm=m.iloc[i-look:i].sum(); f.iloc[i]=rr-(mm.values@B[1:])
  vals=[];dates=[];ns=[]
  for i in range(len(r)-1):
   q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:vals.append(spearmanr(q.iloc[:,0],q.y).statistic);dates.append(r.index[i]);ns.append(len(q))
  a=np.array(vals); rank=f.rank(pct=True); turn=np.nanmean(np.abs(rank.diff()).mean(axis=1))
  print('w',w,'look',look,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'turn',round(turn,4))
  # recent half and annual regime diagnostics
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
   z=a[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]
   if len(z): print(' regime',lo,'-',hi,'n',len(z),'ic',round(z.mean(),5),'icir',round(z.mean()/z.std(ddof=1),5))
