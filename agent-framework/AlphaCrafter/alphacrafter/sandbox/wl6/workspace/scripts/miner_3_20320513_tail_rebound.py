import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-05-12')
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float) for s in U},axis=1).sort_index().loc[:cut]; r=P.pct_change()
# Tail-rebound: short shock is rewarded only when it is extreme relative to each asset's own recent volatility.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std(); shock=(v20/(v60+1e-12)).clip(.5,3)
f=-P.pct_change(5)/(v20+1e-12)*shock
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1],'coverage',f.notna().stack().mean())
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; vals=[];ns=[];ds=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt].rename('x'),fr.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.x.corr(z.y,method='spearman')
   if np.isfinite(q): vals.append(q);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('h',h,'valid_dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 q=pd.Series(a,index=ds); print('regimes',q.groupby(q.index.year).mean().round(6).to_dict())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
