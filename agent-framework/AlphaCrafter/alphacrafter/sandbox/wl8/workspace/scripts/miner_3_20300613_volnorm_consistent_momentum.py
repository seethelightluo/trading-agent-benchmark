import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-06-13'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); mom=p.pct_change(40); vol=r.rolling(20,min_periods=15).std(); consistency=(r.rolling(20,min_periods=15).mean()>0).astype(float)
f=(mom/(vol*np.sqrt(20)+1e-8))*(0.75+0.5*consistency); dates=[]; ics=[]; ns=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: dates.append(p.index[i]); ns.append(len(z)); ics.append(spearmanr(z.f,z.y).statistic)
ics=np.array(ics); dates=pd.DatetimeIndex(dates); print('factor volnorm_consistent_momentum_40d dates',len(ics),'range',dates[0],dates[-1],'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.mean(ics),'ICIR',np.mean(ics)/np.std(ics,ddof=1),'hit',np.mean(ics>0))
for label,mask in [('180d',dates>=pd.Timestamp('2029-12-13')),('360d',dates>=pd.Timestamp('2029-06-13')),('2028',(dates>=pd.Timestamp('2028-01-01'))&(dates<pd.Timestamp('2029-01-01'))),('2029',(dates>=pd.Timestamp('2029-01-01'))&(dates<pd.Timestamp('2030-01-01'))),('2030',dates>=pd.Timestamp('2030-01-01'))]:
 q=ics[mask]; print(label,len(q),np.mean(q) if len(q) else None,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else None)
for h in [5,20]:
 q=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:q.append(spearmanr(z.f,z.y).statistic)
 print('horizon',h,'dates',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/miner_3_20300613_volnorm_consistent_momentum_40d_ic.csv',index=False)
f.iloc[-1].rename('signal').to_csv('scripts/miner_3_20300613_volnorm_consistent_momentum_40d_signal.csv')
