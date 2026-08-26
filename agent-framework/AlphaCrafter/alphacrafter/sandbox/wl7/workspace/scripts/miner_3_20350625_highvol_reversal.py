import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-06-24']
r=C.pct_change(); v=r.rolling(20,min_periods=15).std(); rev=-(C/C.shift(5)-1); rev=rev.sub(rev.median(axis=1),axis=0); f=(rev/v).where(v.gt(v.median(axis=1),axis=0)).shift(1)
print('candidate=high_idio_vol_residual_reversal5_vol20')
for h in [1,5,10,20]:
 y=C.shift(-h)/C-1; A=[]; dates=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(d);ns.append(len(z))
 a=np.array(A); dates=pd.DatetimeIndex(dates)
 print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'dates',len(a),'avgN',np.mean(ns),'hit',np.mean(a>0))
 if h==20:
  print('coverage',f.notna().sum().sum()/f.size,'active_dates',f.notna().any(axis=1).mean())
  for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-06-24')]:
   q=a[(dates>=pd.Timestamp(st))&(dates<=pd.Timestamp(en))];print('REG',st,len(q),q.mean())
f.to_csv('scripts/miner_3_20350625_highvol_reversal_signal.csv',index_label='date')
