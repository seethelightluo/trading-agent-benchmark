import os, sys
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_stock_daily_data,get_index_daily_data):
        try:
            x=f(s,5000)
            if x is not None and len(x)>100: return x
        except Exception: pass
    return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Candidate: volatility-breakout continuation, lagged: 5d return, activated when short vol > long vol; cross-sectional rank
rows=[]
for s,x in D.items():
 x=x.copy(); x['date']=pd.to_datetime(x.date); x=x.sort_values('date').set_index('date')
 r=x.close.pct_change(); vol5=r.rolling(5).std(); vol20=r.rolling(20).std()
 sig=(r.rolling(5).sum() * (vol5/vol20).clip(0.5,3.0)).shift(1)
 for d,v in sig.items(): rows.append((d,s,v))
F=pd.DataFrame(rows,columns=['date','symbol','factor']).pivot(index='date',columns='symbol',values='factor')
P=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.pct_change() for s,x in D.items()})
# forward 10d return from close d to d+10
R=P.shift(-10).rolling(10).sum().shift(-9) # likely wrong; direct close future / close -1
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close for s,x in D.items()})
R=C.shift(-10)/C-1
ics=[]; counts=[]
for d in F.index:
 z=pd.concat([F.loc[d],R.reindex(F.index).loc[d]],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); counts.append(len(z))
ic=pd.Series(ics,index=F.index[-len(ics):]).dropna()
for n in [180,365]:
 q=ic.tail(n); print('recent',n,q.mean(),q.mean()/q.std() if q.std()>0 else np.nan)
print('dates',len(ic),'avgN',np.mean(counts),'coverage',F.notna().sum().sum()/(F.shape[0]*len(U)))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
print('decay')
for h in [1,5,10,20]:
 rr=C.shift(-h)/C-1; a=[]
 for d in F.index:
  z=pd.concat([F.loc[d],rr.reindex(F.index).loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(h,np.nanmean(a),len(a))
# artifacts
os.makedirs('scripts',exist_ok=True)
F.to_csv('scripts/miner_1_20311002_volbreak_continuation_signal.csv'); ic.rename('ic').to_csv('scripts/miner_1_20311002_volbreak_continuation_ic.csv')
