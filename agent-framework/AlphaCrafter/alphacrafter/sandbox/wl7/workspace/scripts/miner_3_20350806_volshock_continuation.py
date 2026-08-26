import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-08-05']
r=C.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
# Volatility-shock continuation: medium trend is favored when recent volatility expands,
# with cross-sectional median residualization and one-day lag.
trend=C/C.shift(20)-1
shock=(v20/v60).clip(0.25,4.0)
F=(trend/v20)*shock
F=F.sub(F.median(axis=1),axis=0).shift(1)
y=C.shift(-20)/C-1
A=[]; ds=[]; ns=[]
for d in F.index:
 q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8:A.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(d);ns.append(len(q))
a=np.asarray(A); ds=np.asarray(ds)
print('volshock_continuation IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f coverage %.4f turnover %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),len(a),np.mean(ns),np.mean(a>0),F.notna().sum().sum()/(15*len(F)),np.nanmean(np.abs(F.diff()).mean(axis=1))))
for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-08-05')]:
 q=a[(ds>=pd.Timestamp(st))&(ds<=pd.Timestamp(en))]; print('regime',st,len(q),'IC',q.mean() if len(q) else None,'hit',np.mean(q>0) if len(q) else None)
for h in [1,5,10,20]:
 yy=C.shift(-h)/C-1; aa=[]
 for d in F.index:
  q=pd.concat([F.loc[d],yy.loc[d]],axis=1).dropna()
  if len(q)>=8:aa.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 aa=np.asarray(aa); print('H',h,'IC %.8f ICIR %.8f dates %d'%(aa.mean(),aa.mean()/aa.std(ddof=1)*np.sqrt(252),len(aa)))
F.to_csv('scripts/miner_3_20350806_volshock_continuation_signal.csv',index_label='date')
