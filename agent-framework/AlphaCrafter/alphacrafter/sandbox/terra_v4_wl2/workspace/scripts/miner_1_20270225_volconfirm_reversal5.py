import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-02-25')
def g(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None:return x
  except: pass
P={};V={}
for s in U:
 x=g(s).set_index('date').sort_index();x=x[x.index<=cutoff];P[s]=x.close;V[s]=x.volume
px=pd.DataFrame(P).sort_index();vol=pd.DataFrame(V).reindex(px.index); fwd=px.shift(-1)/px-1
sig=-px.pct_change(5)*np.log1p((vol/vol.rolling(20,min_periods=10).median()).clip(.1,10))
a=[];n=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z))
a=np.array(a);to=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()
print('dates',len(a),'avgN',np.mean(n),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',sig.notna().sum().sum()/(len(U)*len(sig)),'turnover',to)
for h in [5,10]:
 q=[];fw=px.shift(-h)/px-1
 for d in sig.index:
  z=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q);print('horizon',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
sig.to_csv('../persistent/factor_signals_miner_1_20270225_volconfirm_reversal5.csv',index_label='date')
