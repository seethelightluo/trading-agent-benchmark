import pandas as pd,numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2026-12-17');syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for s in syms}).ffill();r=p.pct_change()
# low volatility, lagged 60d, with momentum tie-breaker
vol=r.shift(1).rolling(60).std(); mom=p.shift(1)/p.shift(21)-1
sig=(-vol + 0.15*mom).sub((-vol+0.15*mom).mean(axis=1),axis=0); f=r.shift(-1)
def run(fr):
 a=[];ds=[];ns=[]
 for d in sig.index:
  x=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
  if len(x)>=8:
   z=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
   if np.isfinite(z):a.append(z);ds.append(d);ns.append(len(x))
 return np.array(a),np.array(ds),ns
a,ds,ns=run(f);print('dates',len(a),'avgN',np.mean(ns),'coverage',np.sum(ns)/(15*len(a)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.nanmean(np.abs(sig.diff()).sum(axis=1)/(np.abs(sig.shift(1)).sum(axis=1)+1e-12)))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-17')]:
 z=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print(lo,len(z),z.mean(),z.mean()/z.std(ddof=1))
for h in [5,10]:
 z,_,_=run(p.shift(-h)/p-1);print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
 sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_2_20261217_lowvol_signal.csv',index=False)
