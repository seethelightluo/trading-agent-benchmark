import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
# cross-sectional dispersion regime, with continuous z-score intensity; factor is 5d reversal scaled by market dispersion
csdisp=r.sub(r.mean(1),axis=0).abs().mean(1); base=csdisp.rolling(60,min_periods=30).median()
intensity=(csdisp/base).clip(.5,2.0)
f=-p.pct_change(5).mul(intensity,axis=0)
for h in [1,5,10]:
 y=r.rolling(h).sum().shift(-h+1); xs=[]; cov=[]; tr=[]; prev=None
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y));cov.append(len(q)/15)
  z=f.iloc[i].rank(pct=True)
  if prev is not None:tr.append(np.nanmean(abs(z-prev)))
  prev=z
 x=np.array(xs);print('h',h,'dates',len(x),'names',np.mean(cov)*15,'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(cov),'turn',np.nanmean(tr))
# dispersion terciles
for label,m in [('low',csdisp<=csdisp.rolling(252,min_periods=60).quantile(.33)),('high',csdisp>=csdisp.rolling(252,min_periods=60).quantile(.67))]:
 x=[]
 for i in range(len(p)-1):
  if not bool(m.iloc[i]):continue
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8:x.append(q.f.corr(q.y))
 print(label,'dates',len(x),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1))
