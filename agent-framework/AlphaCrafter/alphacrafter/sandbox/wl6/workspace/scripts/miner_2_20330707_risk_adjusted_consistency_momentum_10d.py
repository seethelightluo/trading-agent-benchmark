import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-07-06'); xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20).std()*np.sqrt(20); f=c.pct_change(20)*(2*r.gt(0).rolling(20).mean()-1)/(vol+1e-12)
 xs[s]=pd.DataFrame({'fac':f,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1,'f20':c.shift(-20)/c-1,'f40':c.shift(-40)/c-1})
dates=sorted(set().union(*[x.index for x in xs.values()])); prev={}; out={h:[] for h in [5,10,20,40]}; ns=[]; turns=[]
for dt in dates:
 if dt>end: continue
 cur={s:x.loc[dt,'fac'] for s,x in xs.items() if dt in x.index and pd.notna(x.loc[dt,'fac'])}
 if prev:
  co=set(prev)&set(cur)
  if len(co)>=8:
   ra=pd.Series({s:prev[s] for s in co}).rank(); rb=pd.Series({s:cur[s] for s in co}).rank(); turns.append(np.mean(abs(ra-rb))/(len(co)-1))
 prev=cur
 for h in out:
  a=[];b=[]
  for s,x in xs.items():
   if dt in x.index and pd.notna(x.loc[dt,'fac']) and pd.notna(x.loc[dt,'f'+str(h)]): a.append(x.loc[dt,'fac']);b.append(x.loc[dt,'f'+str(h)])
  if len(a)>=8: out[h].append(spearmanr(a,b).statistic)
for h,z in out.items():
 z=np.array(z); print('h',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'avg_n approx',15,'turnover',np.mean(turns))
 if h==10:
  for yr in range(2020,2034):
   v=[q for d,q in zip([d for d in dates if d<=end],z) if d.year==yr]
   if v: print('year',yr,'n',len(v),'ic',np.mean(v))
