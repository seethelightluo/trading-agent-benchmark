import pandas as pd, numpy as np
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame()
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 p[s]=d['close'].astype(float)
r=p.pct_change()
# A reversal signal scaled by downside risk: recent 10d loss is more informative
# when downside semivolatility is low; all inputs are shifted to avoid lookahead.
down=r.where(r<0,0.0).rolling(20,min_periods=15).std()
sig=(-r.rolling(10,min_periods=8).sum()/down.replace(0,np.nan)).shift(1)
fwd=p.pct_change(10).shift(-10)
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit', (a.ic>0).mean(),'turnover',sig.rank(pct=True).diff().abs().mean().mean()/2)
for h in [1,5,10,20]:
 ff=p.pct_change(h).shift(-h); q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for lo,hi in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-10-31')]:
 q=a.loc[lo:hi].ic; print(lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# artifact
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291101_downside_vol_reversal_signal.csv',index=False)
