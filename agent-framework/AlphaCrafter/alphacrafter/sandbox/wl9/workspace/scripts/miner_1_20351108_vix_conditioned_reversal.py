import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# lagged signal, all inputs through t; forward returns t to t+10
r=p.pct_change()
vix_z=(vix-vix.rolling(60).mean())/(vix.rolling(60).std()+1e-12)
gate=(1+0.35*np.clip(vix_z, -1.5, 1.5))
sig=(-p.pct_change(20)/(r.rolling(20).std()*np.sqrt(20)+0.05)).shift(1).mul(gate.shift(1),axis=0)
fwd=p.shift(-10)/p-1
ics=[]; turns=[]; ns=[]
for i,d in enumerate(p.index):
 x=sig.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
  if i>=10:
   prev=sig.iloc[i-10]; oo=prev.notna()&x.notna()
   if oo.sum()>=8: turns.append((x[oo].rank().sub(prev[oo].rank()).abs().mean()/(oo.sum())))
arr=np.array(ics); print('dates',len(arr),'avgN',np.mean(ns),'minN',min(ns),'coverage',np.mean([n/15 for n in ns]));print('IC',arr.mean(),'ICIR',arr.mean()/arr.std(ddof=1),'hit',np.mean(arr>0),'turn',np.mean(turns));
for h in [5,10,20,40,60]:
 yy=p.shift(-h)/p-1; aa=[]
 for d in p.index:
  x=sig.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: aa.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.mean(aa),len(aa))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=arr[(p.index[-len(arr):]>=lo)&(p.index[-len(arr):]<=hi)] if False else []
 # use keyed recomputation simpler
 vals=[]
 for d in p.index:
  if str(d.year) < lo or str(d.year)>hi: continue
  x=sig.loc[d];y=fwd.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 print('regime',lo,hi,np.mean(vals) if vals else None,len(vals))
# artifact
out=pd.DataFrame(sig,columns=assets);out.index.name='date';out.to_csv('scripts/miner_1_20351108_vix_conditioned_reversal_signal.csv')
