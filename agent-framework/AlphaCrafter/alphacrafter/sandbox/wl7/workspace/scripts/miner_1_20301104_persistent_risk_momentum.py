import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s, days=5000) for s in U}
px={s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>100}
p=pd.DataFrame(px).sort_index().ffill()
# no current/future leakage: signal at t predicts t+1..t+10; all rolling features naturally through t
r=np.log(p).diff()
ret40=np.log(p/p.shift(40)); vol60=r.rolling(60).std()*np.sqrt(252)
# persistence fraction, dampen noisy trends; cross-sectional rank is not required for IC
persist=(r>0).rolling(40).mean()
f=(ret40/vol60)*persist
f=f.shift(1) # conservative decision lag
fr=np.log(p.shift(-10)/p)
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt, z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(out),'instruments_avg',out.n.mean(),'coverage',len(out)/(len(p)-10))
print('IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(ddof=1),'hit', (out.ic>0).mean())
print('turnover', f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20]:
 frh=np.log(p.shift(-h)/p); a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],frh.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('h',h,'ic',np.nanmean(a),'icir',np.nanmean(a)/np.nanstd(a,ddof=1),'n',len(a))
# regime split
for name,q in [('early',out.iloc[:len(out)//3]),('middle',out.iloc[len(out)//3:2*len(out)//3]),('late',out.iloc[2*len(out)//3:])]: print(name,q.ic.mean(),len(q))
# artifact
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20301104_persistent_risk_momentum_signal.csv',index=False)
out.reset_index().to_csv('scripts/miner_1_20301104_persistent_risk_momentum_ic.csv',index=False)
