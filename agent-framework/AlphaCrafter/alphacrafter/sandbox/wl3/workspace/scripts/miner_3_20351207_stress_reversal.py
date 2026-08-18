import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-12-06'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
ret3=P.pct_change(3).shift(1); vol20=R.rolling(20,min_periods=15).std().shift(1)
breadth=(R.rolling(5,min_periods=5).sum()>0).mean(axis=1).shift(1)
disp=R.rolling(10,min_periods=8).std().mean(axis=1).shift(1)
meddisp=disp.rolling(120,min_periods=60).median().shift(1)
stress=((breadth<0.45)&(disp>meddisp)).astype(float)
F=(-ret3/(vol20*np.sqrt(3)+1e-12)).mul(stress,axis=0)
for h in [1,3,5,10]:
 vals=[]; cov=[]
 for i in range(25,len(P)-h):
  z=pd.concat([F.iloc[i],P.shift(-h).iloc[i]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q); cov.append(len(z)/15)
 a=pd.Series(vals)
 print('h',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'coverage',round(np.mean(cov),4))
h=5; vals=[]; dates=[]
for i in range(25,len(P)-h):
 z=pd.concat([F.iloc[i],P.shift(-h).iloc[i]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(q): vals.append(q); dates.append(P.index[i])
a=pd.Series(vals,index=dates)
for label,lo,hi in [('early','2020-01-01','2025-01-01'),('middle','2025-01-01','2030-01-01'),('recent','2030-01-01','2035-12-06')]:
 x=a.loc[lo:hi]; print('slice',label,'IC',round(x.mean(),6),'dates',len(x))
F.to_csv('scripts/miner_3_20351207_stress_reversal_signal.csv')
