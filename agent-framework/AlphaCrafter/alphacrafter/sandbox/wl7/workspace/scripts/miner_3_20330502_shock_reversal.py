import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-05-01'); D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; ret=px.pct_change(); vol=ret.rolling(20,min_periods=15).std()
# Contrarian short-horizon shock, normalized by prevailing volatility; lag prevents lookahead.
sig=(-ret.rolling(5,min_periods=5).sum()/(vol+1e-12)).shift(1)
print('assets',len(D),'dates',len(px))
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 a=np.array(vals); print('H',h,'valid_dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
 a3=np.array_split(a,3); print(' thirds ',[round(float(x.mean()),6) for x in a3])
print('coverage',round(sig.notna().mean().mean(),4),'turnover',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4))
sig.to_csv('scripts/miner_3_20330502_shock_reversal_signal.csv')
