import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-06-10');D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();c=d.close.astype(float);v=d.volume.astype(float).replace(0,np.nan)
 # high volume amplifies short-term reversal; lag all features one day
 shock=np.log(v.shift(1)/v.shift(1).rolling(20).median())
 rev=-c.shift(1).pct_change(5)
 D[s]=pd.DataFrame({'f':rev*shock,'r5':c.pct_change(5).shift(-5),'r10':c.pct_change(10).shift(-10),'r20':c.pct_change(20).shift(-20)})
dates=sorted(set().union(*[x.index for x in D.values()]))
for h in [5,10,20]:
 q=[];ns=[]
 for dt in dates:
  if dt>cut:continue
  z=[];y=[]
  for s in U:
   if dt in D[s].index:
    a=D[s].loc[dt,'f'];b=D[s].loc[dt,'r'+str(h)]
    if np.isfinite(a) and np.isfinite(b):z.append(a);y.append(b)
  if len(z)>=8 and np.std(z)>0 and np.std(y)>0:q.append(spearmanr(z,y).statistic);ns.append(len(z))
 a=np.array(q);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4))
for label,lo,hi in [('early','2020-01-01','2023-12-31'),('mid','2024-01-01','2027-12-31'),('recent','2028-01-01','2032-06-10')]:
 q=[]
 for dt in dates:
  if not(pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)):continue
  z=[];y=[]
  for s in U:
   if dt in D[s].index and np.isfinite(D[s].loc[dt,'f']) and np.isfinite(D[s].loc[dt,'r10']):z.append(D[s].loc[dt,'f']);y.append(D[s].loc[dt,'r10'])
  if len(z)>=8 and np.std(z)>0 and np.std(y)>0:q.append(spearmanr(z,y).statistic)
 a=np.array(q);print(label,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5))
print('coverage',round(np.mean([np.isfinite(D[s].f).mean() for s in U]),5))
