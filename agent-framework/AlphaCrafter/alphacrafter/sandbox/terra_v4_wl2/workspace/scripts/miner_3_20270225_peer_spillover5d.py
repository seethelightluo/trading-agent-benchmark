import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None}).sort_index()
# Multi-day peer spillover, excluding asset i, based on lagged 5d returns.
r=px.pct_change(5); f=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for s in px.columns: f[s]=r.drop(columns=s).median(axis=1)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
# regimes h1
fr=px.shift(-1)/px-1;a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(q):a.append((dt,q))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07-01','2027-02-25')]:
 q=[x for d,x in a if str(d)>=lo and str(d)<=hi];print('REG',lo,hi,len(q),np.mean(q) if q else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'assets',len(px.columns),'dates',len(px))
f.stack().rename('signal').rename_axis(['date','symbol']).to_csv('../persistent/factor_signals_miner_3_20270225_peer_spillover5d.csv')
