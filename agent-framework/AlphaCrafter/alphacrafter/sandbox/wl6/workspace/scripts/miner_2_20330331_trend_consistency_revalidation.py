import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-03-30')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change(); fac=c.pct_change(20)*(2*r.gt(0).rolling(20).mean()-1); xs[s]=pd.DataFrame({'fac':fac,'fwd':c.shift(-10)/c-1})
dates=sorted(set().union(*[x.index for x in xs.values()])); ics=[]; ns=[]; prev=None; turns=[]
for dt in dates:
 if dt>end: continue
 cur={s:x.loc[dt,'fac'] for s,x in xs.items() if dt in x.index and pd.notna(x.loc[dt,'fac']) and pd.notna(x.loc[dt,'fwd'])}
 if len(cur)>=8:
  q=spearmanr(list(cur.values()),[xs[s].loc[dt,'fwd'] for s in cur]).statistic
  if np.isfinite(q): ics.append(q); ns.append(len(cur))
 rankcur=pd.Series({s:x.loc[dt,'fac'] for s,x in xs.items() if dt in x.index and pd.notna(x.loc[dt,'fac'])})
 if prev is not None:
  common=prev.index.intersection(rankcur.index)
  if len(common)>=8: turns.append(np.mean(abs(prev[common].rank()-rankcur[common].rank()))/(len(common)-1))
 prev=rankcur
z=np.array(ics); print('horizon 10 dates',len(z),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4),'turnover',round(np.mean(turns),4),'through',end.date())
for y in range(2026,2034):
 a=[]
 for dt,q in zip([d for d in dates if d<=end][-len(z):],z):
  if dt.year==y:a.append(q)
 if a: print(y,round(float(np.mean(a)),6),len(a))
