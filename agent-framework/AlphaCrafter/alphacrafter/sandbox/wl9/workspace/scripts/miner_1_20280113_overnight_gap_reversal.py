import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root=Path('../persistent/stock_data'); ds={}
for a in assets:
 d=pd.read_csv(root/(a+'.csv'),parse_dates=['date']).sort_values('date').set_index('date'); ds[a]=d

def test(h):
 out=[]; cov=[]
 dates=sorted(set().union(*[set(d.index) for d in ds.values()]))
 for dt in dates:
  va=[]; fw=[]
  for a,d in ds.items():
   if dt not in d.index: continue
   i=d.index.get_loc(dt)
   if i<1 or i+h>=len(d): continue
   gap=d.iloc[i].open/d.iloc[i-1].close-1
   y=d.iloc[i+h].close/d.iloc[i].close-1
   if np.isfinite(gap) and np.isfinite(y): va.append(-gap); fw.append(y)
  if len(va)>=8: out.append(spearmanr(va,fw).statistic); cov.append(len(va))
 z=pd.Series(out).dropna(); return z,cov
for h in [1,5,10]:
 z,c=test(h); print('horizon',h,'dates',len(z),'avgN',np.mean(c),'meanIC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
z,c=test(1); print('coverage',np.mean(c)/15); print('years',pd.DataFrame({'ic':z.values}).assign(y=pd.to_datetime(sorted(set().union(*[set(d.index) for d in ds.values()]))[:len(z)]).year).groupby('y').ic.mean())
