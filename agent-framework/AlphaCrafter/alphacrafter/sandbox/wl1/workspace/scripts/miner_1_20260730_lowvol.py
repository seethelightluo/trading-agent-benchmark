import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-07-15')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close.astype(float) for s in U};r={s:x.pct_change() for s,x in p.items()};f={s:1/r[s].rolling(20,min_periods=15).std() for s in U}
dates=sorted(set().union(*[x.index for x in p.values()]));out={1:[],5:[],10:[]};ns={1:[],5:[],10:[]}
for d in dates:
 for h in [1,5,10]:
  vals=[];ys=[]
  for s in U:
   if d not in p[s].index or pd.isna(f[s].loc[d]):continue
   ix=p[s].index.get_loc(d)
   if ix+h>=len(p[s]):continue
   vals.append(f[s].loc[d]);ys.append(p[s].iloc[ix+h]/p[s].iloc[ix]-1)
  if len(vals)>=8 and len(set(vals))>1:out[h].append(spearmanr(vals,ys).statistic);ns[h].append(len(vals))
for h in [1,5,10]:
 a=np.array(out[h]);print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns[h]),2),'coverage',round(np.mean(ns[h])/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
