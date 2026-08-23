import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2029-02-21'
D={s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
C=pd.DataFrame({s:d.close for s,d in D.items()}); V=pd.DataFrame({s:d.volume for s,d in D.items()}); r=C.pct_change()
# breakout continuation only when recent volume is above its own medium-term baseline;
# normalize by volatility to make cross-asset comparable.
ret=r.rolling(20).sum(); vol=r.rolling(40).std(); vz=np.log1p(V).rolling(10).mean()-np.log1p(V).rolling(60).mean(); F=(ret/vol)*(1+0.5*np.tanh(vz)).replace([np.inf,-np.inf],np.nan)
a=[]; ds=[]; rows=[]
for i in range(65,len(C)-10):
 x=F.iloc[i-1]; y=C.iloc[i+10]/C.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); a.append(q); ds.append(C.index[i]); rows += [(C.index[i],s,float(x[s])) for s in z.index]
a=pd.Series(a,index=pd.DatetimeIndex(ds)).dropna(); print('dates',len(a),'rows',len(rows),'mean_n',len(rows)/len(a),'coverage',len(rows)/(len(a)*15)); print('IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
for n,m in [('2020-24',a.index<'2025-01-01'),('2025-26',(a.index>='2025-01-01')&(a.index<'2027-01-01')),('2027-28',(a.index>='2027-01-01')&(a.index<'2029-01-01')),('recent',a.index>='2028-01-01')]:
 q=a[m];print(n,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan,(q>0).mean() if len(q) else np.nan)
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290222_volume_confirmed_breakout_signal.csv',index=False)
