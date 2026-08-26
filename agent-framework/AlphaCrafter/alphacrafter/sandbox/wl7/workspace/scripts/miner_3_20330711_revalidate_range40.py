import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-07-10'); D={}
for a in A:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change(); lo=px.rolling(40,min_periods=30).min(); hi=px.rolling(40,min_periods=30).max()
sig=((.5-(px-lo)/(hi-lo+1e-12))/(r.rolling(20,min_periods=15).std()+1e-12)).shift(1)
sig.to_csv('scripts/miner_3_20330711_range40_reversal_signal.csv')
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'thirds',[round(x.mean(),6) for x in np.array_split(a,3)])
for label, sub in [('recent180',px.iloc[-180:]),('recent500',px.iloc[-500:])]:
 s=sig.loc[sub.index]; f=sub.shift(-10)/sub-1; v=[]
 for dt in sub.index:
  q=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): v.append(z)
 a=np.array(v); print(label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),4),'assets',len(D),'dates',len(px))
