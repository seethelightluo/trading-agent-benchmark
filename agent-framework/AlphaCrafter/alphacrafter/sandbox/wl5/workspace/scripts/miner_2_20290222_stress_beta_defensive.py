import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-02-21'
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut].astype(float)
r=P.pct_change()
# defensive stress-beta: negative rolling beta to common stress proxy (cross-asset downside breadth),
# rewarded together with recent relative strength, and risk scaled.
stress=(-r.clip(upper=0).mean(axis=1)).rolling(5).mean()
beta=r.rolling(60).cov(stress).div(stress.rolling(60).var(),axis=0)
rel=r.rolling(20).sum().sub(r.rolling(20).sum().median(axis=1),axis=0)
vol=r.rolling(30).std()
F=(-beta*rel/vol).replace([np.inf,-np.inf],np.nan)
# lag signal one day, forward 10 trading day return
rows=[]; ics=[]; dates=[]
for i in range(60,len(P)-10):
 d=P.index[i]; x=F.iloc[i-1]; y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.concat([x,y],axis=1).dropna();
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(d)
  rows += [(d,s,float(x[s])) for s in z.index]
a=pd.Series(ics,index=pd.DatetimeIndex(dates)).dropna()
print('dates',len(a),'rows',len(rows),'mean_n',len(rows)/len(a),'coverage',len(rows)/(len(a)*15))
print('IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
for label,mask in [('2020-24',a.index<'2025-01-01'),('2025-26',(a.index>='2025-01-01')&(a.index<'2027-01-01')),('2027-28',(a.index>='2027-01-01')&(a.index<'2029-01-01')),('recent',a.index>='2028-01-01')]:
 q=a[mask]; print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan,(q>0).mean() if len(q) else np.nan)
# artifact for provenance
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290222_stress_beta_defensive_signal.csv',index=False)
