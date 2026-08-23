import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2033-08-17'); xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change()
 vol=r.ewm(span=20,min_periods=20,adjust=False).std()*np.sqrt(10)
 a=c.pct_change(10)*(2*r.gt(0).rolling(10).mean()-1)/(vol+1e-12)
 b=c.pct_change(20)*(2*r.gt(20).rolling(20).mean()-1)/(vol+1e-12)
 fac=((a+b)/2).ewm(span=5,min_periods=5,adjust=False).mean()
 xs[s]=pd.DataFrame({'fac':fac,'f10':c.shift(-10)/c-1})
ics=[]; ns=[]; turns=[]; prev=None
for dt in sorted(set().union(*[x.index for x in xs.values()])):
 if dt>end: continue
 cur={s:x.loc[dt,'fac'] for s,x in xs.items() if dt in x.index and pd.notna(x.loc[dt,'fac']) and pd.notna(x.loc[dt,'f10'])}
 if len(cur)>=8:
  ics.append((dt,spearmanr(list(cur.values()),[xs[s].loc[dt,'f10'] for s in cur]).statistic)); ns.append(len(cur))
  if prev:
   co=set(prev)&set(cur)
   if len(co)>=8:
    a1=pd.Series({s:prev[s] for s in co}).rank(); b1=pd.Series({s:cur[s] for s in co}).rank(); turns.append(np.mean(abs(a1-b1))/(len(co)-1))
  prev=cur
z=np.array([q for _,q in ics]); print('dates',len(z),'instruments_mean',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for y in range(2020,2034):
 v=[q for d,q in ics if d.year==y]
 if v: print('year',y,'n',len(v),'ic',np.mean(v),'hit',np.mean(np.array(v)>0))
