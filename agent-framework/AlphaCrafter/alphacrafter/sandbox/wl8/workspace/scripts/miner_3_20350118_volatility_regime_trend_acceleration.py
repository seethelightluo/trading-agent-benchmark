import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Trend acceleration is recent 10d pace relative to prior 60d pace. Condition it on
# each asset's own volatility regime: emphasize acceleration when current 20d vol is
# below its trailing 120d median, and damp it in stressed vol. Lag all inputs one day.
v20=r.rolling(20,min_periods=15).std(); vmed=v20.rolling(120,min_periods=80).median()
acc=(p.pct_change(10)-p.pct_change(60)/6)/(v20*np.sqrt(20))
reg=(vmed/(v20+1e-12)).clip(.5,2.0)
sig=(acc*reg).shift(1)
rows=[]
for dt in sig.index:
 fwd=p.shift(-10).loc[dt]/p.loc[dt]-1
 z=pd.concat([sig.loc[dt],fwd],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); q=ic.ic
print('dates',len(ic),'avgN',round(ic.n.mean(),2),'coverage',round(sig.notna().mean().mean(),4),'IC',q.mean(),'dailyICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
rk=sig.rank(axis=1,pct=True); tr=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: tr.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(tr))
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),'n',len(a))
out=sig.loc[ic.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20350118_volatility_regime_trend_acceleration_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_3_20350118_volatility_regime_trend_acceleration_ic.csv',index=False)
