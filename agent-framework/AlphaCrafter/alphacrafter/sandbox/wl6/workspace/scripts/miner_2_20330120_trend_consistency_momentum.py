import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-01-19')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change()
 fac=c.pct_change(20)*(r.gt(0).rolling(20).mean()*2-1)
 xs[s]=pd.DataFrame({h:c.shift(-h)/c-1 for h in [5,10,20,40]}).assign(fac=fac)
for h in [5,10,20,40]:
  ics=[]; ns=[]; turns=[]; prev={}
  dates=sorted(set().union(*[x.index for x in xs.values()]))
  for dt in dates:
   if dt>end: continue
   a=[];b=[]
   for s,x in xs.items():
    if dt in x.index and pd.notna(x.loc[dt,'fac']) and pd.notna(x.loc[dt,h]): a.append(x.loc[dt,'fac']);b.append(x.loc[dt,h])
   if len(a)>=8:
    q=spearmanr(a,b).statistic
    if np.isfinite(q):ics.append(q);ns.append(len(a))
   cur={s:xs[s].loc[dt,'fac'] for s in U if dt in xs[s].index and pd.notna(xs[s].loc[dt,'fac'])}
   if prev:
    common=set(prev)&set(cur)
    if len(common)>=8:
     ra=pd.Series({s:prev[s] for s in common}).rank();rb=pd.Series({s:cur[s] for s in common}).rank();turns.append(np.mean(abs(ra-rb))/(len(common)-1))
   prev=cur
  z=np.array(ics); print('h',h,'dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(252),'hit',np.mean(z>0),'turnover',np.mean(turns))
