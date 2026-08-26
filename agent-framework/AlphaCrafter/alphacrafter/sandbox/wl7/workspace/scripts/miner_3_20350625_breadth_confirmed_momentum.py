import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-06-24']
r=C.pct_change(); v=r.rolling(40,min_periods=25).std()
# Candidate: breadth-conditioned intermediate momentum. 60d return is cross-sectionally
# residualized, scaled by volatility, and activated only when the lagged market breadth
# confirms (at least 60% assets above their 20d moving average). Signal is lagged one day.
ret=C/C.shift(60)-1
raw=ret-ret.median(axis=1).values[:,None]
f=(raw/v).where((C>C.rolling(20,min_periods=15).mean()).mean(axis=1).shift(1)>=.60).shift(1)
print('candidate=breadth_confirmed_momentum60_vol40')
y=C.shift(-20)/C-1; A=[]; dates=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(d);ns.append(len(z))
a=np.array(A); dates=np.array(dates)
print('IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f coverage %.4f active_dates %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),len(a),np.mean(ns),np.mean(a>0),f.notna().sum().sum()/f.size,np.mean(f.notna().any(axis=1))))
for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-06-24'),('2034-06-24','2035-06-24')]:
 q=a[(dates>=pd.Timestamp(st))&(dates<=pd.Timestamp(en))]; print('REG',st,len(q),q.mean() if len(q) else np.nan, q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 yy=C.shift(-h)/C-1; aa=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8:aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.array(aa); print('H',h,'IC',aa.mean(),'ICIR',aa.mean()/aa.std(ddof=1)*np.sqrt(252),'dates',len(aa))
f.to_csv('scripts/miner_3_20350625_breadth_confirmed_momentum_signal.csv',index_label='date')
