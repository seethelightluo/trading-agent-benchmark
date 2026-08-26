import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv'); d['date']=pd.to_datetime(d['date']); px[a]=d.set_index('date')['close'].sort_index()
P=pd.DataFrame(px).sort_index().ffill()
# candidate: 20d trend divided by 20d realized vol, demeaned cross-sectionally
ret=P.pct_change()
sig=(P.shift(1)/P.shift(21)-1)/(ret.rolling(20).std().shift(1)*np.sqrt(20))
# reduce common beta: cross-sectional demean, retain direction (higher = better)
sig=sig.sub(sig.mean(axis=1),axis=0)
fwd=P.shift(-20)/P-1
ics=[]; turnovers=[]; cov=[]; rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],y[ok]).statistic
  ics.append(ic); cov.append(ok.mean()); rows.append((dt,ic,ok.sum()))
  if len(rows)>1:
   prev=sig.loc[rows[-2][0]]; z=prev.notna()&x.notna()
   turnovers.append(np.mean(np.abs(x[z].rank(pct=True)-prev[z].rank(pct=True))))
ics=np.array(ics)
print('dates',len(ics),'avg_n',np.mean([r[2] for r in rows]),'coverage',np.mean(cov),'turnover',np.mean(turnovers))
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0))
for name, lo, hi in [('2020-23','2020','2024'),('2024-26','2024','2027'),('2027-28','2027','2029')]:
 z=np.array([v for d,v,n in rows if lo<=str(d)[:4]<hi]); print(name,len(z),z.mean() if len(z) else None,z.mean()/z.std(ddof=1) if len(z)>1 else None)
# decay
for h in [5,10,20,40]:
 yy=P.shift(-h)/P-1; q=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(sig.loc[dt][ok],yy.loc[dt][ok]).statistic)
 q=np.array(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
# signal artifact
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20281204_volnorm_momentum20_signal.csv',index=False)
