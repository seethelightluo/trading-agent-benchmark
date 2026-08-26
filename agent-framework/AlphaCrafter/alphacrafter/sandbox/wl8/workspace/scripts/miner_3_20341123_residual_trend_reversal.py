import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date');D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); m=r.mean(axis=1)
# Rolling beta-neutral residual returns; reverse short EWMA of residual medium/long trend.
beta=r.rolling(60,min_periods=30).cov(m)/m.rolling(60,min_periods=30).var()
res=r-beta.multiply(m,axis=0); rv=res.rolling(30,min_periods=20).std()
trend=.6*res.rolling(20,min_periods=15).mean()+.4*res.rolling(60,min_periods=40).mean()
raw=-trend/(rv+1e-8); sig=raw.ewm(span=3,min_periods=2).mean().shift(1)
f=p.shift(-10)/p-1;rows=[];cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)));cov.append(sig.loc[dt].notna().mean())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna();q=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',np.mean(cov),'IC',q.mean(),'dailyICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
rk=sig.rank(axis=1,pct=True);tt=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:tt.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(tt))
for w in [365,750,1260]:
 z=q.tail(w);print('recent',w,'ICIR',z.mean()/z.std(ddof=1))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1;a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
out=sig.loc[ic.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20341123_residual_trend_reversal_signal.csv',index=False);ic.reset_index().to_csv('scripts/miner_3_20341123_residual_trend_reversal_ic.csv',index=False)
