import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-06-10')
xs={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 c=d.close.astype(float)
 # acceleration: short return relative to medium return; lagged one session
 f=(c.shift(1).pct_change(5)-c.shift(1).pct_change(20))
 # contrarian acceleration, robust cross-section rank handled by IC
 xs[s]=pd.DataFrame({'f':-f,'r5':c.pct_change(5).shift(-5),'r10':c.pct_change(10).shift(-10),'r20':c.pct_change(20).shift(-20)})
dates=sorted(set().union(*[x.index for x in xs.values()]))
for h in [5,10,20]:
 vals=[]; ns=[]
 for dt in dates:
  if dt>cut: continue
  z=[]; y=[]
  for s in U:
   if dt in xs[s].index:
    a=xs[s].loc[dt,'f']; b=xs[s].loc[dt,'r'+str(h)]
    if np.isfinite(a) and np.isfinite(b): z.append(a);y.append(b)
  if len(z)>=8 and np.std(z)>0 and np.std(y)>0:
   vals.append(spearmanr(z,y).statistic);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5),'hit',round(np.mean(a>0),4))
for label,lo,hi in [('early','2020-01-01','2023-12-31'),('mid','2024-01-01','2027-12-31'),('recent','2028-01-01','2032-06-10')]:
 vals=[]
 for dt in dates:
  if not (pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)): continue
  z=[];y=[]
  for s in U:
   if dt in xs[s].index and np.isfinite(xs[s].loc[dt,'f']) and np.isfinite(xs[s].loc[dt,'r10']):z.append(xs[s].loc[dt,'f']);y.append(xs[s].loc[dt,'r10'])
  if len(z)>=8 and np.std(z)>0 and np.std(y)>0: vals.append(spearmanr(z,y).statistic)
 a=np.array(vals); print(label,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5) if len(a)>1 else None)
print('coverage',np.mean([np.isfinite(xs[s]['f']).mean() for s in U]))
