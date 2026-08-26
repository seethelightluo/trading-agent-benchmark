import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-07-10'); D={}
for a in A:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change()
# Intermediate-horizon momentum divided by trailing volatility, lagged one session.
sig=(r.rolling(20,min_periods=15).sum()/(r.rolling(40,min_periods=25).std()+1e-12)).shift(1)
sig.to_csv('scripts/miner_3_20330711_mom20_vol40_signal.csv')
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'thirds',[round(x.mean(),6) for x in np.array_split(a,3)])
print('coverage',round(sig.notna().mean().mean(),4),'assets',len(D),'dates',len(px))
