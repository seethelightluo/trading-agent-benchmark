import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-10-11');D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill();R=P.pct_change(); n=len(D)
for w in [10,20,40,60]:
 F=-R.rolling(w,min_periods=max(5,w//2)).std().shift(1)
 vals=[];cov=[]
 for i in range(65,len(P)-10):
  z=pd.concat([F.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);cov.append(len(z)/15)
 a=pd.Series(vals);rr=a.iloc[-252:]
 print('w',w,'dates',len(a),'inst',n,'ic',round(a.mean(),6),'icir',round(a.mean()/a.std(),4),'recent',round(rr.mean(),6),round(rr.mean()/rr.std(),4),'hit',round((a>0).mean(),4),'coverage',round(np.mean(cov),4))
 if w==20:F.to_csv('scripts/miner_1_20351026_lowvol_signal.csv')
