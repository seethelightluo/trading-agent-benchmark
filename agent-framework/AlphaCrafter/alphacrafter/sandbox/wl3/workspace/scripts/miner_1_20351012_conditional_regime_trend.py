import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-10-11');D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); r=P.pct_change(); ret=P.pct_change(20).shift(1); br=(ret>0).sum(1)/ret.notna().sum(1)
# Conditional regime factor: trend-follow relative strength in broad positive regimes,
# mean-reversion in broad negative regimes; continuous confidence avoids hard threshold.
F=ret.mul(np.where(br.values>=.5,1.0,-1.0)*(0.5+2*abs(br.values-.5)),axis=0)
for h in [5,10,20]:
 v=[];c=[]
 for i in range(65,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):v.append(q);c.append(len(z)/15)
 a=pd.Series(v); rec=a.iloc[-252:]
 print('h',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),4),'recentIC',round(rec.mean(),6),'recentIR',round(rec.mean()/rec.std(),4),'hit',round((rec>0).mean(),4),'cov',round(np.mean(c),4))
F.index.name='date';F.to_csv('scripts/miner_1_20351012_conditional_regime_trend_signal.csv')
