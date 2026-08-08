# Single-idea research: peer downside/upside capture asymmetry, tested without forward leakage
import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; P={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']).dt.normalize()
 P[a]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(P).sort_index(); p=p.loc[:p.index.max()]
r=p.pct_change(); market=r.median(axis=1); cut=p.index.max()
# Candidate: relative resilience in broadly negative market sessions versus beta in positive sessions.
# Per asset, 60-session mean excess return on negative-market days less positive-market days;
# at least 10 observations in each state; cross-sectional median centering; one-day signal lag.
raw=pd.DataFrame(index=p.index,columns=A,dtype=float)
for i in range(59,len(p)):
 rr=r.iloc[i-59:i+1]; mm=market.iloc[i-59:i+1]
 neg=rr.loc[mm<0]; pos=rr.loc[mm>0]
 if len(neg)>=10 and len(pos)>=10:
  raw.iloc[i]=(neg.sub(mm.loc[neg.index],axis=0).mean()-pos.sub(mm.loc[pos.index],axis=0).mean())
F=raw.sub(raw.median(axis=1),axis=0).shift(1)
def metric(h,lo=None,hi=None,sign=1):
 y=p.shift(-h).div(p).sub(1); ics=[]; ns=[]; dates=[]
 for dt in F.index:
  if (lo and dt<pd.Timestamp(lo)) or (hi and dt>pd.Timestamp(hi)): continue
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())*sign
   if np.isfinite(q): ics.append(q);ns.append(len(z));dates.append(dt)
 x=np.array(ics); return {'dates':len(x),'mean_n':round(float(np.mean(ns)),2) if ns else None,'min_n':min(ns) if ns else None,'ic':round(float(x.mean()),6) if len(x) else None,'icir':round(float(x.mean()/x.std(ddof=1)),6) if len(x)>1 and x.std(ddof=1)>0 else None,'hit':round(float((x>0).mean()),4) if len(x) else None}
print('CUT',cut.date(),'PRICE_DATES',len(p),'UNIVERSE',len(A),'COVERAGE',int(F.notna().sum().sum()),'/',F.size,round(F.notna().mean().mean(),6))
for orient,s in [('direct',1),('inverse',-1)]:
 print('ORIENTATION',orient)
 for h in [1,5,10,20]: print('H',h,metric(h,sign=s))
print('DIRECT_10_REGIMES')
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print(n,metric(10,lo,hi,1))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_SD',round(float(F.std(axis=1).mean()),6))
print('NOTE novelty correlation against all admitted signals is not computable from JSON definitions alone; candidate cannot be admitted absent reproducible aligned library panels.')
