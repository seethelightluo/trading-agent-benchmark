import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Volatility-scaled cross-sectional 40d reversal: fade relative return, normalized by
# each asset's trailing 20d volatility; use only information lagged one session.
rel=p.pct_change(40).sub(p.pct_change(40).median(axis=1),axis=0)
vol=r.rolling(20,min_periods=15).std()
sig=(-rel/vol.replace(0,np.nan)).shift(1)
ff={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
def calc(x):
 out=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 return pd.Series(out),np.mean(ns)
v,n=calc(ff[10]); v=v.loc[v.index>=0] if False else v
print('dates',len(v),'avgN',n,'coverage',sig.notna().mean().mean(),'IC',v.mean(),'dailyICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for h in [1,5,10,20]:
 q,_=calc(ff[h]); print('decay',h,q.mean())
for w in [365,750,1260]:
 q,_=calc(ff[10]);q=q.tail(w);print('recent',w,'ICIR',q.mean()/q.std(ddof=1),'IC',q.mean())
ranks=sig.rank(axis=1,pct=True); tt=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8:tt.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(tt))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20341221_volscaled_relative_reversal_signal.csv',index=False)
