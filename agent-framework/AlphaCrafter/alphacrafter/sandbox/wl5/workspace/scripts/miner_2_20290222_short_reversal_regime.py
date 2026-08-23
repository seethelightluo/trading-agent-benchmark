import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2029-02-21'
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut].astype(float);r=P.pct_change()
# short-horizon reversal, amplified when the asset is below its 60d trend (downtrend bounce), risk scaled
short=r.rolling(3).sum(); trend=r.rolling(60).sum(); vol=r.rolling(20).std(); gate=(trend<0).astype(float)*1.0+(trend>=0).astype(float)*0.25
F=(-short*gate/vol).replace([np.inf,-np.inf],np.nan); a=[];ds=[];rows=[]
for i in range(65,len(P)-10):
 x=F.iloc[i-1];y=P.iloc[i+10]/P.iloc[i]-1;z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(P.index[i]);rows += [(P.index[i],s,float(x[s])) for s in z.index]
a=pd.Series(a,index=pd.DatetimeIndex(ds)).dropna();print('dates',len(a),'rows',len(rows),'mean_n',len(rows)/len(a),'coverage',len(rows)/(len(a)*15));print('IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
for n,m in [('2020-24',a.index<'2025-01-01'),('2025-26',(a.index>='2025-01-01')&(a.index<'2027-01-01')),('2027-28',(a.index>='2027-01-01')&(a.index<'2029-01-01')),('recent',a.index>='2028-01-01')]:
 q=a[m];print(n,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan,(q>0).mean() if len(q) else np.nan)
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290222_short_reversal_regime_signal.csv',index=False)
