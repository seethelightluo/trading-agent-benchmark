import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-07-20')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change()
 ewvol=r.ewm(span=20,min_periods=20,adjust=False).std()*np.sqrt(20)
 fac=c.pct_change(20)*(2*r.gt(0).rolling(20).mean()-1)/(ewvol+1e-12)
 xs[s]=pd.DataFrame({'fac':fac,'f10':c.shift(-10)/c-1})
dates=sorted(set().union(*[x.index for x in xs.values()])); ics=[]; ns=[]; ranks=[]; prev=None
for dt in dates:
 if dt>end: continue
 cur={s:x.loc[dt,'fac'] for s,x in xs.items() if dt in x.index and pd.notna(x.loc[dt,'fac']) and pd.notna(x.loc[dt,'f10'])}
 if len(cur)>=8:
  ic=spearmanr(list(cur.values()),[xs[s].loc[dt,'f10'] for s in cur]).statistic; ics.append((dt,ic)); ns.append(len(cur))
  if prev:
   common=set(prev)&set(cur)
   if len(common)>=8:
    a=pd.Series({s:prev[s] for s in common}).rank(); b=pd.Series({s:cur[s] for s in common}).rank(); ranks.append(np.mean(abs(a-b))/(len(common)-1))
 prev=cur
z=np.array([v for _,v in ics]); print('dates',len(z),'instruments_mean',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(ranks))
for yr in range(2020,2034):
 v=[q for d,q in ics if d.year==yr]
 if v: print('year',yr,'n',len(v),'ic',np.mean(v),'hit',np.mean(np.array(v)>0))
