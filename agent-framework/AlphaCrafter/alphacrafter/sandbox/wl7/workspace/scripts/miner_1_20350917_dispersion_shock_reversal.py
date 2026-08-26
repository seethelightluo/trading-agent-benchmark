import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-09-12']
r=C.pct_change(); cs=r.mean(axis=1); disp=r.sub(cs,axis=0).abs().mean(axis=1)
rollhi=disp.rolling(120,min_periods=60).median(); gate=(disp.rolling(5,min_periods=5).mean()>rollhi)
shock=r.rolling(3,min_periods=3).sum()/r.rolling(20,min_periods=15).std()
raw=(-shock).where(gate,np.nan); F=raw.sub(raw.median(axis=1),axis=0).shift(1)
y=C.shift(-5)/C-1
A=[]; ds=[]; ns=[]
for d in F.index:
 q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): A.append(z);ds.append(pd.Timestamp(d));ns.append(len(q))
a=np.asarray(A); ds=np.asarray(ds,dtype='datetime64[ns]'); ir=a.mean()/a.std(ddof=1)*np.sqrt(252) if len(a)>1 and a.std(ddof=1)>0 else np.nan
print('dates',len(a),'avgN',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),ir,np.mean(a>0)))
for lo,hi in [('2026-07-16','2029-12-31'),('2030-01-01','2033-12-31'),('2034-01-01','2035-09-12')]:
 q=a[(ds>=np.datetime64(lo))&(ds<=np.datetime64(hi))];print('regime',lo,hi,'n',len(q),'IC',q.mean() if len(q) else np.nan,'hit',np.mean(q>0) if len(q) else np.nan)
print('coverage %.4f active_rate %.4f turnover %.6f'%(F.notna().sum().sum()/(15*len(F)),gate.mean(),np.nanmean(np.abs(F.diff()).mean(axis=1))))
F.to_csv('scripts/miner_1_20350917_dispersion_shock_reversal_signal.csv',index_label='date')
