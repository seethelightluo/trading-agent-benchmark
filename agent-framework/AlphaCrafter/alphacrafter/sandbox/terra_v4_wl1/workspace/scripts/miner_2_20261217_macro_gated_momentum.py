import pandas as pd, numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for s in syms}
p=pd.DataFrame(px).ffill(); r=p.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].reindex(p.index).ffill()
mom=p.shift(1)/p.shift(21)-1; vol=r.shift(1).rolling(20).std(); base=mom/vol.replace(0,np.nan)
reg=pd.Series(np.where(vix.shift(1).pct_change(10)<0,1.0,0.35),index=p.index); sig=base.mul(reg,axis=0); sig=sig.sub(sig.mean(axis=1),axis=0); fwd=r.shift(-1)
def calc(fr):
 a=[];ds=[];ns=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   q=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(x))
 return np.array(a),np.array(ds),ns
a,ds,ns=calc(fwd); print('dates',len(a),'avgN',np.mean(ns),'coverage',np.sum(ns)/(len(a)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.nanmean(np.abs(sig.diff()).sum(axis=1)/(np.abs(sig.shift(1)).sum(axis=1)+1e-12)))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]:
 z=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print(lo,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [5,10]:
 aa,_,_=calc(p.shift(-h)/p-1);print('decay',h,aa.mean(),aa.mean()/aa.std(ddof=1))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal');out.to_csv('scripts/miner_2_20261217_macro_gated_momentum_signal.csv',index=False)
