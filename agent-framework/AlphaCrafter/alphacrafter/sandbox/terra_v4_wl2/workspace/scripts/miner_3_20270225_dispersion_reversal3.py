import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data/'
px=pd.DataFrame({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2027-02-24']
r=px.pct_change(); disp=r.std(axis=1); extreme=(disp>disp.rolling(60,min_periods=30).quantile(.75)).shift(1)
# Reversal signal is formed from the prior completed 3-day return and activated only after elevated cross-asset dispersion.
sig=(-r.rolling(3).sum()).where(extreme); sig=sig.sub(sig.median(axis=1),axis=0)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',np.mean(ns) if len(ns) else np.nan,'IC',np.mean(a) if len(a) else np.nan,'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0) if len(a) else np.nan)
a=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],(px.shift(-1)/px-1).loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for lab,aa,bb in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-24')]:
 q=[x for d,x in a if str(d)>=aa and str(d)<=bb];print(lab,len(q),np.mean(q) if q else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('coverage',sig.notna().sum().sum()/(len(U)*len(sig)),'active_dates',int(extreme.sum()),'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_dispersion_reversal3.csv',index=False)
