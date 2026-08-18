import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-11-08'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill();R=P.pct_change(); r10=P.pct_change(10).shift(1); v=R.rolling(30,min_periods=20).std().shift(1)
# Cross-asset breadth-conditioned trend, with risk normalization and lagged regime.
breadth=(P.pct_change(10)>0).mean(axis=1).shift(1); regime=np.where(breadth>=0.5,1.,-1.)
F=r10.div(v*np.sqrt(10)+1e-12).mul(regime,axis=0)
for h in [5,10,20]:
 vals=[]; cov=[]
 for i in range(45,len(P)-h):
  z=pd.concat([F.iloc[i],P.shift(-h).iloc[i]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);cov.append(len(z)/15)
 a=pd.Series(vals);print('h',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'coverage',round(np.mean(cov),4))
F.to_csv('scripts/miner_3_20351109_breadth_trend_signal.csv')
