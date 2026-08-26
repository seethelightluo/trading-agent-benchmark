import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(D).sort_index().loc['2020-01-01':'2034-08-05'].ffill();r=P.pct_change();v=r.rolling(20,min_periods=15).std();disp=r.std(axis=1)
for qtile in [.6,.75,.8]:
 gate=(disp>disp.rolling(60,min_periods=30).quantile(qtile)).astype(float);F=(-(P.pct_change(5)/(v*np.sqrt(20)+1e-12))).mul(gate,axis=0).shift(1);R=P.shift(-10)/P-1;a=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):a.append(c);ns.append(len(z))
 x=np.asarray(a);print('Q',qtile,'active',gate.mean(),'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'recent500',x[-500:].mean(),x[-500:].mean()/x[-500:].std(ddof=1))
