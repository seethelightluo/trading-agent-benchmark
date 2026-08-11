import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15'); px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=END]
  if len(d)>120:px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1); beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0); resid=r-beta.mul(m,axis=0); rv=resid.rolling(20,min_periods=10).std()
f=-resid.rolling(4,min_periods=4).sum()/rv
ics=[]; cov=[]; turns=[]; d5=[]; d10=[]; d20=[]
for i in range(len(p)-21):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  ics.append(spearmanr(q.f,q.y).statistic); cov.append(len(q)/len(U))
  for h,out in [(5,d5),(10,d10),(20,d20)]:
   qh=pd.concat([f.iloc[i].rename('f'),p.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
   if len(qh)>=8 and qh.f.nunique()>1 and qh.y.nunique()>1:out.append(spearmanr(qh.f,qh.y).statistic)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8: turns.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.asarray(ics)
print('candidate=residual_reversal_4d dates',len(x),'avg_names %.3f coverage %.5f IC %.8f ICIR %.8f hit %.5f turnover %.5f decay5 %.8f decay10 %.8f decay20 %.8f'%(np.mean(cov)*len(U),np.mean(cov),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(turns),np.mean(d5),np.mean(d10),np.mean(d20)))
for label,z in [('early',x[:len(x)//3]),('middle',x[len(x)//3:2*len(x)//3]),('late',x[2*len(x)//3:]),('recent250',x[-250:])]:print(label,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
print('assets',len(px),'period',p.index.min(),p.index.max())
