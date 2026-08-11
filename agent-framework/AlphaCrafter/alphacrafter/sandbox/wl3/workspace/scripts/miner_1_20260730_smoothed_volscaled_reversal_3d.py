import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15'); px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=END]
  if len(d)>120:px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20,min_periods=10).std()
raw=-p.pct_change(3)/v
# causal cross-sectional signal smoothing: only information available at date t.
f=raw.rolling(3,min_periods=2).mean()
ics=[]; cov=[]; turn=[]; d5=[]
for i in range(len(p)-5):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  ics.append(spearmanr(q.f,q.y).statistic); cov.append(len(q)/len(U))
  q5=pd.concat([f.iloc[i].rename('f'),p.pct_change(5).iloc[i+5].rename('y')],axis=1).dropna()
  if len(q5)>=8 and q5.f.nunique()>1 and q5.y.nunique()>1:d5.append(spearmanr(q5.f,q5.y).statistic)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8:turn.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.array(ics); print('candidate=smoothed_volscaled_reversal_3d dates',len(x),'avg_names',np.mean(cov)*len(U),'coverage',np.mean(cov),'IC %.8f ICIR %.8f hit %.5f turnover %.5f decay5 %.8f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(turn),np.mean(d5)))
for label,z in [('early',x[:len(x)//2]),('late',x[len(x)//2:]),('recent250',x[-250:])]:print(label,'dates',len(z),'IC %.8f ICIR %.8f hit %.5f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
print('assets',len(px),'period',p.index.min(),p.index.max())
