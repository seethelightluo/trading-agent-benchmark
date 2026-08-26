import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(p).sort_index().ffill(); r=p.pct_change()
# Cross-asset dispersion, fully lagged before forming conditional contrarian signal.
disp=r.T.rolling(20,min_periods=15).std().T.shift(1)
cut=disp.rolling(120,min_periods=60).median()
f=(-r.rolling(5,min_periods=5).sum()).shift(1).where(disp>cut,0.0)
y=p.pct_change(10).shift(-10)
D=[]; A=[]; ns=[]; turns=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  D.append(d); A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  turns.append(np.mean(np.sign(f.loc[d]).values != np.sign(f.shift(1).loc[d]).values))
A=np.array(A);D=pd.DatetimeIndex(D)
print('all dates',len(A),'avgN',np.mean(ns),'coverage',f.notna().sum(1).mean()/15,'turnover',np.mean(turns),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',np.mean(A>0))
for name,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31'),('2029','2029','2029-09-09')]:
 q=A[(D>=lo)&(D<=hi)];print(name,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
f.to_csv('scripts/miner_2_20290924_dispersion_high_reversal5_signal.csv',index_label='date')
