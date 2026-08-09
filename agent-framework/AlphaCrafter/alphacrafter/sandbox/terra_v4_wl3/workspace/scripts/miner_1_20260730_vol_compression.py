import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; R={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x.date); x=x.sort_values('date').set_index('date'); p=x.close.astype(float).loc[:'2026-07-15']; P[s]=p; R[s]=p.pct_change()
px=pd.concat(P,axis=1,sort=True); r=pd.concat(R,axis=1,sort=True)
rv5=r.rolling(5,min_periods=5).std(); rv40=r.rolling(40,min_periods=30).std(); f=-(rv5/rv40)
for h in [1,5,10]:
 fr=pd.concat({s:P[s].shift(-h)/P[s]-1 for s in U},axis=1,sort=True); vals=[]; dates=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(a))
 z=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); print('h',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(),6),'hit',round((z>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.index.year): print('regime',yr,'IC',round(g.mean(),5),'n',len(g))
rr=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum().sum()/f.size,4),'rank turnover',round(rr.diff().abs().mean(axis=1).mean(),4))
