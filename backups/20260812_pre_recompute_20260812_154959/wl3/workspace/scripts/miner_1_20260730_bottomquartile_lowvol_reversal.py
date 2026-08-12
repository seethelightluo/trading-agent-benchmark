import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15'); px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=END]
  if len(d)>120: px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20,min_periods=10).std()
# Novel candidate: stronger low-vol conditioning (bottom quartile), 3-day reversal.
f=(-p.pct_change(3)).where(vol.le(vol.quantile(.25,axis=1),axis=0)); out=[]; cov=[]; turns=[]; d5=[]; d10=[]
for i in range(len(p)-10):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  out.append(spearmanr(q.f,q.y).statistic); cov.append(len(q)/len(U))
  q5=pd.concat([f.iloc[i].rename('f'),p.pct_change(5).iloc[i+5].rename('y')],axis=1).dropna()
  q10=pd.concat([f.iloc[i].rename('f'),p.pct_change(10).iloc[i+10].rename('y')],axis=1).dropna()
  if len(q5)>=8 and q5.f.nunique()>1 and q5.y.nunique()>1:d5.append(spearmanr(q5.f,q5.y).statistic)
  if len(q10)>=8 and q10.f.nunique()>1 and q10.y.nunique()>1:d10.append(spearmanr(q10.f,q10.y).statistic)
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8: turns.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
x=np.asarray(out); print('candidate=bottom_quartile_lowvol_reversal_3d dates',len(x),'avg_names',np.mean(cov)*len(U),'coverage',np.mean(cov),'IC %.8f ICIR %.8f hit %.5f turnover %.5f decay5 %.8f decay10 %.8f period %s %s'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(turns),np.mean(d5),np.mean(d10),p.index.min(),p.index.max()))
for label,z in [('early',x[:len(x)//2]),('late',x[len(x)//2:]),('recent250',x[-250:])]: print(label,'dates',len(z),'IC %.8f ICIR %.8f hit %.5f'%(len(z) and z.mean(),len(z) and z.mean()/z.std(ddof=1),(z>0).mean()))
print('assets',len(px),'valid_dates',len(x),'avg_valid_cross_section',np.mean(cov)*len(U))
