import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=2400)
 if d is not None and len(d)>120: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
# Candidate: volatility-normalized short-term reversal, with a continuous (not binary)
# market-stress multiplier. All rolling quantities are lagged at decision date.
v=get_index_daily_data('VIX',days=2400)
v=v.set_index('date').close.astype(float).reindex(p.index).ffill()
assetvol=r.rolling(20,min_periods=15).std()
volnorm=(-p.pct_change(5)).div(assetvol*np.sqrt(5))
vr=(v/v.rolling(252,min_periods=60).median()).clip(0.5,2.5)
# keep a nonzero calm signal; stress increases reversal exposure continuously
f=volnorm.mul((0.75+0.5*vr).clip(0.5,2.0),axis=0)
print('assets',len(px),'dates',len(p),'candidate','volnorm_reversal_stress_intensity')
for h in [1,5,10]:
 y=r.rolling(h).sum().shift(-h+1); ics=[]; ns=[]; tr=[]; prev=None
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   ics.append(q.f.corr(q.y));ns.append(len(q))
  z=f.iloc[i].rank(pct=True)
  if prev is not None: tr.append(np.nanmean(abs(z-prev)))
  prev=z
 x=np.asarray(ics); x=x[np.isfinite(x)]
 print('horizon',h,'dates',len(x),'avg_names',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/15,'turnover',np.nanmean(tr))
# rolling recent and annual regime diagnostics at admission horizon
h=1;y=r.shift(-1); rows=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((p.index[i],q.f.corr(q.y)))
z=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
for w in [60,120,252]:
 x=z.ic.tail(w);print('recent',w,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
for yr,g in z.groupby(z.index.year): print('year',yr,'dates',len(g),'IC',g.ic.mean(),'ICIR',g.ic.mean()/g.ic.std(ddof=1))
print('corr_simple_reversal',pd.concat([f.stack(),(-p.pct_change(5)).stack()],axis=1).dropna().corr().iloc[0,1])
