import pandas as pd,numpy as np,os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
P=pd.DataFrame(D).sort_index(); ret=P.pct_change(); lag=P.shift(1)
high=lag.rolling(90,min_periods=60).max(); v=ret.shift(1).rolling(20,min_periods=15).std()
sig=(-(1-lag/high)/(v*np.sqrt(20))).rank(axis=1,pct=True)-.5
fwd=P.shift(-10)/P-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=q.ic
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'annICIR',x.mean()/x.std(ddof=1)*np.sqrt(252))
for w in [365,750,1260]:
 z=x.tail(w); print('recent',w,z.mean()/z.std(ddof=1))
for h in [1,5,10,20]:
 ff=P.shift(-h)/P-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr))
print('turnover',sig.diff().abs().mean().mean())
sig.tail(500).to_csv('scripts/miner_3_20341109_drawdown_unconditional_signal.csv'); q.to_csv('scripts/miner_3_20341109_drawdown_unconditional_ic.csv')
