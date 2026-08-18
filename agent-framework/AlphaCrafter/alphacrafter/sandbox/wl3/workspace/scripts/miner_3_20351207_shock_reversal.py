import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-12-06'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill();R=P.pct_change(); v=R.rolling(20,min_periods=15).std().shift(1)
# Volatility-scaled short-term reversal with an extreme-move confidence weight.
r1=P.pct_change(1).shift(1); z=(r1/(v+1e-12)).abs(); F=(-r1/(v+1e-12))*(1+0.5*z.clip(upper=3))
for h in [1,3,5,10]:
 vals=[];cov=[]
 for i in range(22,len(P)-h):
  x=pd.concat([F.iloc[i],P.shift(-h).iloc[i]/P.iloc[i]-1],axis=1).dropna()
  if len(x)>=8:
   q=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);cov.append(len(x)/15)
 a=pd.Series(vals);print('h',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'coverage',round(np.mean(cov),4))
F.to_csv('scripts/miner_3_20351207_shock_reversal_signal.csv')
