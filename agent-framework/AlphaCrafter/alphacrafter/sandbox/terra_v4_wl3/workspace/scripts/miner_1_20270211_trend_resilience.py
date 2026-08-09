import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# Trend resilience: 20-day return divided by absolute worst peak-to-trough drawdown in lookback.
rollmax=P.rolling(20,min_periods=15).max(); dd=P/rollmax-1; mdd=(-dd.rolling(20,min_periods=15).min()).replace(0,np.nan)
F=(P/P.shift(20)-1)/mdd
ics=[]; counts=[]; turns=[]; regs=[]
for i in range(20,len(P)-1):
 z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');ics.append(c);counts.append(len(z)); regs.append((P.index[i],c)); turns.append(np.mean(np.sign(z.iloc[:,0])!=np.sign(F.iloc[i-1].reindex(z.index).fillna(0))))
a=np.array(ics); print('candidate=trend_resilience_20d dates',len(a),'avg_names',round(np.mean(counts),2),'coverage',round(sum(counts)/(len(a)*15),4));print('IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round(np.mean(a>0),4),'turnover',round(np.mean(turns),4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=np.array([c for d,c in regs if lo<=str(d)[:4]<=hi]);print(lo+'-'+hi,'n',len(q),'ic',round(float(np.nanmean(q)),6) if len(q) else None)
for h in [1,5,10]:
 z=[]
 for i in range(20,len(P)-h):
  q=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('decay',h,len(z),round(float(np.nanmean(z)),6),round(float(np.nanmean(z)/np.nanstd(z)),6))
