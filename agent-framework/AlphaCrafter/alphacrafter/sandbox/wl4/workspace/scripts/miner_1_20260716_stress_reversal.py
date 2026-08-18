import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
v=get_index_daily_data('VIX',days=1800)
if v is None: raise RuntimeError('no VIX')
v=v.set_index('date').close.astype(float).reindex(p.index).ffill()
# Stress-conditioned cross-asset reversal: recent losses are favored, with intensity rising
# only when VIX is above its trailing 60d median; no future data.
stress=(v>v.rolling(60,min_periods=30).median()).astype(float)
intensity=(v/v.rolling(252,min_periods=60).median()).clip(0.5,2.5)
f=-p.pct_change(5).mul(stress*intensity,axis=0)
for h in [1,5,10]:
  ys=r.rolling(h).sum().shift(-h+1)
  ics=[]; cov=[]; turns=[]
  prev=None
  for i in range(len(p)-h):
   q=pd.concat([f.iloc[i].rename('f'),ys.iloc[i].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:
    ics.append(q.f.corr(q.y));cov.append(len(q)/15)
   if prev is not None:
    z=f.iloc[i].rank(pct=True); turns.append(np.nanmean(abs(z-prev)))
   prev=f.iloc[i].rank(pct=True)
  x=np.array(ics,dtype=float); x=x[np.isfinite(x)]
  print('horizon',h,'dates',len(x),'mean_names',round(np.mean(cov)*15,2) if cov else 0,'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(cov) if cov else 0,'turnover',np.nanmean(turns))
# regime detail and correlation with existing simple reversal
valid=pd.concat([f.stack(),(-p.pct_change(5)).stack()],axis=1).dropna();print('library_corr_reversal5',valid.iloc[:,0].corr(valid.iloc[:,1]))
for label,m in [('stress',stress>0),('calm',stress==0)]:
 x=[]
 for i in range(len(p)-1):
  if not m.iloc[i]: continue
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y))
 print(label,'dates',len(x),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1))
