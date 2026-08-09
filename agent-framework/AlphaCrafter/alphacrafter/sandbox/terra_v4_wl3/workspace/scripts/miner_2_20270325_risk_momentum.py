import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x.date=pd.to_datetime(x.date); x=x.sort_values('date'); r=x.close.pct_change()
  # medium-term trend, scaled by recent risk: interpretable risk-adjusted momentum
  x['sig']=x.close.pct_change(20)/r.rolling(20).std(); x['fwd']=x.close.shift(-1)/x.close-1
  D[s]=x[['date','sig','fwd']]

def run(h):
 rows=[]
 dates=sorted(set().union(*[set(x.date) for x in D.values()]))
 for dt in dates:
  a=[]
  for x in D.values():
   z=x[x.date==dt]
   if len(z) and np.isfinite(z.sig.iloc[0]) and len(x):
    # calculate forward h already via indexed group externally unavailable
    pass
  
 # rebuild fwd by each series
 for s,x in D.items():
  raw=pd.read_csv('../persistent/stock_data/'+s+'.csv'); raw.date=pd.to_datetime(raw.date); raw=raw.sort_values('date'); raw['sig']=raw.close.pct_change(20)/raw.close.pct_change().rolling(20).std(); raw['fwd']=raw.close.shift(-h)/raw.close-1; D[s]=raw[['date','sig','fwd']]
 for dt in dates:
  a=[]
  for x in D.values():
   z=x[x.date==dt]
   if len(z) and np.isfinite(z.sig.iloc[0]) and np.isfinite(z.fwd.iloc[0]): a.append((z.sig.iloc[0],z.fwd.iloc[0]))
  if len(a)>=8:
   q=spearmanr(*zip(*a)).statistic
   if np.isfinite(q): rows.append(q)
 return np.array(rows)
for h in [1,5,10]:
 r=run(h); print(h,len(r),r.mean(),r.mean()/r.std(ddof=1),(r>0).mean())
