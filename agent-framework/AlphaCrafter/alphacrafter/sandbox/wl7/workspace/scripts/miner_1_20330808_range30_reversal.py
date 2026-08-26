import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-08-07']; r=px.pct_change()
lo=px.rolling(30,min_periods=20).min(); hi=px.rolling(30,min_periods=20).max()
sig=((.5-(px-lo)/(hi-lo+1e-12))/(r.rolling(15,min_periods=10).std()+1e-12)).shift(1)
sig.to_csv('scripts/miner_1_20330808_range30_reversal_signal.csv')
print('assets',len(D),'dates',len(px),'coverage',round(sig.notna().mean().mean(),4))
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 a=np.array(vals)
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'thirds',[round(float(x.mean()),6) for x in np.array_split(a,3)])
for n in [180,500,750]:
 s=sig.iloc[-n:]; f=px.iloc[-n:].shift(-10)/px.iloc[-n:]-1; v=[]
 for dt in s.index:
  q=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): v.append(z)
 a=np.array(v); print('recent',n,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
print('turnover',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4))
