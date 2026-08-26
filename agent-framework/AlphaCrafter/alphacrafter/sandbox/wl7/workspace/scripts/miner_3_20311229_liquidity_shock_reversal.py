import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').set_index('date')
Ds=sorted(set().union(*[set(x.index) for x in D.values()])); R={h:[] for h in [1,5,10,20]}; cov=[];turn=[];prev=None
for t in Ds:
 a={}; f={h:{} for h in R}
 for s,x in D.items():
  if t not in x.index:continue
  z=x.loc[:t];c=z.close;v=z.volume.replace(0,np.nan)
  if len(z)<45 or pd.isna(c.iloc[-1]):continue
  vz=np.log(v).iloc[-1]-np.log(v).rolling(30).median().iloc[-1]
  if pd.isna(vz):continue
  a[s]=-(c.iloc[-1]/c.iloc[-6]-1)*(1+np.clip(vz,-1,2))
  fut=x.loc[x.index>t].close
  for h in R:
   if len(fut)>=h:f[h][s]=fut.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15);rr=pd.Series(a).rank(pct=True);turn.append(0 if prev is None else (rr-prev).abs().mean());prev=rr
  for h in R:
   com=[s for s in a if s in f[h] and np.isfinite(f[h][s])]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[f[h][s] for s in com]).statistic
    if np.isfinite(q):R[h].append((t,q,len(com)))
print('dates',len(R[5]),'avgN',np.mean([z[2] for z in R[5]]),'coverage',np.mean(cov),'turnover',np.mean(turn))
for h,a in R.items():
 q=pd.Series([z[1] for z in a]);print('H',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean());print('thirds',[q.iloc[i*len(q)//3:(i+1)*len(q)//3].mean() for i in range(3)])
