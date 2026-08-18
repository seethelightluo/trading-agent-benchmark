import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];end=pd.Timestamp('2028-07-03');b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:end].ffill();R=P.pct_change();f=-(P.pct_change(3)/(R.rolling(20).std()+1e-8));y=P.shift(-10)/P-1
A=[];N=[];D=[]
for d in f.index:
 q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8:A.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);N.append(len(q));D.append(pd.Timestamp(d))
a=np.array(A);D=np.array(D,dtype='datetime64[ns]');print('ALL',len(a),np.mean(N),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-07-03')]:
 q=(D>=np.datetime64(lo))&(D<=np.datetime64(hi));z=a[q];print('REG',lo,len(z),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0))
r=f.rank(pct=True);print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turn',(r-r.shift()).abs().mean(axis=1).dropna().mean());f.to_csv('scripts/miner_1_20280704_volnorm_reversal3_signal.csv')
