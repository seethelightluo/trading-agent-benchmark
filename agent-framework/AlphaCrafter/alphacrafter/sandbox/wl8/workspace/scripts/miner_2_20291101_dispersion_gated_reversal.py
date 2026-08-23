import pandas as pd,numpy as np
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p=pd.DataFrame()
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();p[s]=d.close.astype(float)
r=p.pct_change(); disp=r.rolling(20,min_periods=15).std().mean(axis=1); base=-(.6*r.rolling(5,min_periods=4).sum()+.4*r.rolling(20,min_periods=15).sum())/r.rolling(20,min_periods=15).std();
# only trade reversal when lagged cross-asset dispersion is above its expanding median
reg=(disp.shift(1)>disp.shift(1).rolling(120,min_periods=60).median()).astype(float)
sig=base.shift(1).mul(reg,axis=0); f=p.pct_change(10).shift(-10); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for lo,hi in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-10-31')]:
 q=a.loc[lo:hi].ic;print(lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291101_dispersion_gated_reversal_signal.csv',index=False)
