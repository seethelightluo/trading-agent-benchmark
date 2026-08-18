import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-12-06'); C={};V={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.set_index('date'); C[s]=d.close.astype(float)
  if 'volume' in d: V[s]=d.volume.astype(float)
P=pd.DataFrame(C).sort_index().loc[:cutoff].ffill(); Vol=pd.DataFrame(V).reindex(P.index).ffill(); R=P.pct_change()
# volume-confirmed trend: lagged 20d return scaled by contemporaneous (lagged) volume participation ratio
F=(R.rolling(20,min_periods=20).sum()*(Vol.rolling(20,min_periods=20).mean()/Vol.rolling(60,min_periods=60).mean())).shift(1)
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]
 for i in range(65,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna(); ns.append(len(z))
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);dates.append(P.index[i])
 a=pd.Series(vals,index=dates); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recentIC',round(a.tail(252).mean(),6),'recentIR',round(a.tail(252).mean()/a.tail(252).std(ddof=1),6))
# turnover proxy and coverage
print('coverage',round(F.notna().mean().mean(),4),'turnover',round((F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
