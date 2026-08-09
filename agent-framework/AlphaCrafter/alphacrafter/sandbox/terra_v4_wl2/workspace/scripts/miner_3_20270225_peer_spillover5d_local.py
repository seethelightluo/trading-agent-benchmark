import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
px=pd.DataFrame(p).sort_index(); r=px.pct_change(5)
# Cross-asset peer lead-lag, excluding self; rank/median peer signal.
f=pd.DataFrame(index=px.index,columns=px.columns)
for s in px: f[s]=r.drop(columns=s).median(axis=1)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1;a=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
# one-day regime reporting
fr=px.shift(-1)/px-1;a=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(q):a.append((d,q))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07-01','2027-02-25')]:
 q=[x for d,x in a if str(d)>=lo and str(d)<=hi];print('REG',lo,hi,len(q),round(np.mean(q),6),round(np.mean(q)/np.std(q,ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'assets',len(px.columns),'dates',len(px))
f.stack().rename('signal').rename_axis(['date','symbol']).to_csv('../persistent/factor_signals_miner_3_20270225_peer_spillover5d.csv')
