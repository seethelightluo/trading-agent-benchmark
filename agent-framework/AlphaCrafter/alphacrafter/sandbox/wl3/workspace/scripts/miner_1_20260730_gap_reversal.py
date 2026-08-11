import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15'); D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=END].copy(); d['date']=pd.to_datetime(d.date); D[s]=d.set_index('date')
# Novel: gap reversal, prior overnight gap (open/prev close), normalized by 20d vol; predicts next close return
cl=pd.concat({s:d.close for s,d in D.items()},axis=1).sort_index().ffill(); op=pd.concat({s:d.open for s,d in D.items()},axis=1).sort_index().reindex(cl.index).ffill(); r=cl.pct_change(); v=r.rolling(20,min_periods=10).std()
gap=op/cl.shift(1)-1
f=(-gap/v).replace([np.inf,-np.inf],np.nan)
out=[]; cov=[]; turn=[]; dec5=[]; dec10=[]
for i in range(len(cl)-10):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  out.append(spearmanr(q.f,q.y).statistic); cov.append(len(q)/len(U))
  for h,a in [(5,dec5),(10,dec10)]:
   z=pd.concat([f.iloc[i].rename('f'),cl.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
   if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(spearmanr(z.f,z.y).statistic)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8:turn.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.asarray(out); print('dates',len(x),'assets',len(D),'avg_names',np.mean(cov)*len(U),'coverage',np.mean(cov),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turn',np.mean(turn),'d5',np.mean(dec5),'d10',np.mean(dec10),'start',cl.index.min(),'end',cl.index.max())
for name,z in [('early',x[:len(x)//2]),('late',x[len(x)//2:]),('recent250',x[-250:])]: print(name,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
