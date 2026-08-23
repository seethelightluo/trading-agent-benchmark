import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
P=pd.concat(D,axis=1).sort_index(); R=P.pct_change()
# Cross-asset residual reversal: negate 20d return relative to contemporaneous equal-weight benchmark,
# with 20d residual volatility normalization. All values use data through date t.
bench=R.mean(axis=1); resid=R.sub(bench,axis=0)
raw=resid.rolling(20).sum(); scale=resid.rolling(60).std()*np.sqrt(20)
sig=(-raw/scale).replace([np.inf,-np.inf],np.nan)
ics=[]; ns=[]; turns=[]
for i in range(65,len(P)-10):
 z=pd.concat([sig.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  if i>65:
   q=pd.concat([sig.iloc[i],sig.iloc[i-1]],axis=1).dropna(); turns.append(np.mean(np.sign(q.iloc[:,0])!=np.sign(q.iloc[:,1])))
a=np.asarray(ics); print('dates',len(a),'avg_inst',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0),'turnover',np.mean(turns))
for n in [180,360]:
 b=a[-n:]; print('recent',n,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std())
for h in [1,5,10]:
 q=[]
 for i in range(65,len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.asarray(q); print('decay',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
# signal artifact for deterministic audit
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20310417_residual_reversal_20d_signal.csv',index=False)
