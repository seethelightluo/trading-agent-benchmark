import pandas as pd, numpy as np, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
px={}
for a in assets:
 p=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(p): px[a]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rel=R.rolling(5).sum(); F=(-rel.sub(rel.median(axis=1),axis=0)).rolling(3).mean()
for h in [1,5,10]:
 fwd=sum(R.shift(-k) for k in range(1,h+1)); vals=[]; dates=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=dates).dropna(); print('h',h,'dates',len(s),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',np.mean(s>0))
rank=F.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 fwd=R.shift(-1); s=[]
 for dt in F.loc[lo:hi].index:
  z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:s.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=pd.Series(s).dropna();print(lo,'n',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1) if len(s)>1 else np.nan)
out=pd.DataFrame(F.stack(),columns=['signal']);out.index.names=['date','symbol'];out.to_csv('scripts/miner_1_20260730_relative_dispersion_5d_signal.csv')
