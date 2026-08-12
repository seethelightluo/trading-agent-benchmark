import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in A:
 x=get_stock_daily_data(s,days=2600)
 if x is not None:D[s]=x.sort_values('date').drop_duplicates('date').set_index('date')
rows=[]
for s,x in D.items():
 c=x.close.astype(float); r=c.pct_change(); v=r.rolling(20).std()
 for i in range(65,len(x)-10):
  if not np.isfinite(v.iloc[i]) or v.iloc[i]<1e-5:continue
  # stable trend: average of 20/60 momentum, penalize disagreement and volatility
  m20=c.iloc[i]/c.iloc[i-20]-1; m60=c.iloc[i]/c.iloc[i-60]-1
  agree=1 if m20*m60>=0 else -1
  f=(.4*m20+.6*m60)*(.5+.5*agree)/(v.iloc[i]+.01)
  rows.append((x.index[i],s,f,c.iloc[i+1]/c.iloc[i]-1,c.iloc[i+5]/c.iloc[i]-1))
z=pd.DataFrame(rows,columns=['d','s','f','r1','r5']); I={1:[],5:[]}; ns=[]
for d,g in z.groupby('d'):
 if len(g)>=8:
  ns.append(len(g));
  for k in [1,5]: I[k].append(g.f.corr(g['r'+str(k)],method='spearman'))
for k in I:
 a=np.array(I[k],float);print(k,'IC %.5f ICIR %.5f hit %.3f n %d'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),len(a)))
print('avg n',np.mean(ns),'coverage',np.mean(ns)/15)
