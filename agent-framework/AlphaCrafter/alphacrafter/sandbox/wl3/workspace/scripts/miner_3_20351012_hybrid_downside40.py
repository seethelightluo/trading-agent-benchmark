import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-10-11');D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill();R=P.pct_change()
for w in [20,25,30,40,60]:
 down=R.where(R<0).rolling(w,min_periods=max(8,w//3)).std(); total=R.rolling(w,min_periods=max(10,w//2)).std(); den=down.fillna(total)
 F=P.pct_change(20).shift(1).div(den.shift(1)*np.sqrt(20)+1e-12); vals=[];cov=[]
 for i in range(65,len(P)-10):
  z=pd.concat([F.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);cov.append(len(z)/15)
 a=pd.Series(vals);r=a.iloc[-252:]
 print('w',w,'dates',len(a),'inst',len(D),'ic',round(a.mean(),6),'icir',round(a.mean()/a.std(),4),'recent',round(r.mean(),6),round(r.mean()/r.std(),4),'coverage',round(np.mean(cov),4))
 if w==40:F.to_csv('scripts/miner_3_20351012_hybrid_downside40_signal.csv')
