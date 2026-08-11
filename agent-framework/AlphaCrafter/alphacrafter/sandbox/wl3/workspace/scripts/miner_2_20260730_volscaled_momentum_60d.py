import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15'); px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=END]
  if len(d)>120:px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20,min_periods=10).std()
f=p.pct_change(60)/vol
ics=[]; cov=[]; turns=[]; d5=[]; d10=[]
for i in range(len(p)-11):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  ics.append(spearmanr(q.f,q.y).statistic);cov.append(len(q)/15)
  for h,out in [(5,d5),(10,d10)]:
   z=pd.concat([f.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()
   if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna(); turns.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.array(ics);print('candidate=volscaled_momentum_60d dates',len(x),'avg_names',np.mean(cov)*15,'coverage',np.mean(cov),'IC %.8f ICIR %.8f hit %.5f turnover %.5f decay5 %.8f decay10 %.8f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(turns),np.mean(d5),np.mean(d10)))
for label,z in [('2020-22',x[:len(x)//3]),('2023-24',x[len(x)//3:2*len(x)//3]),('2025-26',x[2*len(x)//3:]),('recent250',x[-250:])]:print(label,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
