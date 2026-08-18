import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-10-11'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
# Hybrid downside-risk denominator: downside deviation when available, otherwise total volatility.
# This preserves the downside-risk interpretation while avoiding sparse coverage in short histories.
down=R.where(R<0).rolling(30,min_periods=8).std()
total=R.rolling(30,min_periods=15).std()
den=down.fillna(total)
F=P.pct_change(20).shift(1).div(den.shift(1)*np.sqrt(20)+1e-12)
print('dates',len(P),'instruments',len(D),'cutoff',P.index.max(),'factor hybrid_downside_total_30')
for h in [1,3,5,10,20]:
 vals=[]; cov=[]
 for i in range(65,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q); cov.append(len(z)/len(U))
 a=pd.Series(vals); recent=a.iloc[-252:] if len(a)>252 else a
 print('h',h,'n',len(a),'all_ic',round(a.mean(),6),'all_icir',round(a.mean()/a.std(),4),'recent_ic',round(recent.mean(),6),'recent_icir',round(recent.mean()/recent.std(),4),'hit',round((recent>0).mean(),4),'cov',round(np.mean(cov),4))
F.index.name='date'; F.to_csv('scripts/miner_3_20351012_hybrid_downside30_signal.csv')
