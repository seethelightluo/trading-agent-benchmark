import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-10-11'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change(); n=len(D)
# smoothed breadth is lagged, and smoothly changes trend amplitude around neutral
for bw in [5,10,20]:
 breadth=(R.rolling(20,min_periods=10).sum()>0).mean(axis=1).rolling(bw,min_periods=3).mean().shift(1)
 F=P.pct_change(20).shift(1).mul((0.5+breadth).values,axis=0)
 vals=[]; cov=[]
 for i in range(65,len(P)-10):
  z=pd.concat([F.iloc[i],(P.iloc[i+10]/P.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);cov.append(len(z)/15)
 a=pd.Series(vals); rr=a.iloc[-252:]
 print('bw',bw,'dates',len(a),'inst',n,'ic',a.mean(),'icir',a.mean()/a.std(),'recent',rr.mean(),rr.mean()/rr.std(),'coverage',np.mean(cov))
 if bw==10:F.to_csv('scripts/miner_1_20351026_smoothed_breadth_signal.csv')
