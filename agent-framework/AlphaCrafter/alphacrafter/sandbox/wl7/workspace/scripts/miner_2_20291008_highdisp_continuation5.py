import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
# High-dispersion continuation: lagged 5d return, activated in high cross-asset dispersion.
disp=r.rolling(20,min_periods=15).std().mean(1); gate=disp.shift(1)>disp.shift(1).rolling(120,min_periods=80).median()
f=r.rolling(5,min_periods=4).sum().shift(1).where(gate)
y=p.pct_change(10).shift(-10);D=[];A=[];ns=[];active=[]
for d in f.index:
 if not gate.loc[d]: continue
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:
  D.append(d);A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));active.append(1)
A=np.array(A);D=pd.DatetimeIndex(D)
print('factor=high_dispersion_continuation5 horizon=10')
print('active_dates',len(A),'avgN',round(np.mean(ns),2),'active_fraction',round(np.mean(gate.dropna()),4),'coverage_active',round(np.mean(ns)/15,4),'IC',round(A.mean(),6),'ICIR',round(A.mean()/A.std(ddof=1),6),'hit',round(np.mean(A>0),4))
for name,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31'),('2029','2029','2029-10-08')]:
 q=A[(D>=lo)&(D<=hi)];print(name,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round(np.mean(q>0),4) if len(q) else None)
f.to_csv('scripts/miner_2_20291008_highdisp_continuation5_signal.csv',index_label='date')
