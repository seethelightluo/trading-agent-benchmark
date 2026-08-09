import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
# Relative-strength factor: asset trailing return minus contemporaneous cross-sectional median.
for look in [10,20,40]:
 raw=p.pct_change(look); f=raw.sub(raw.median(axis=1),axis=0)
 out={h:[] for h in [1,5,10]}; turns=[]; cov=[]; dates=[]
 for i in range(len(p)-10):
  for h in out:
   q=pd.concat([f.iloc[i].rename('f'),p.pct_change(h).iloc[i+h].rename('y')],axis=1).dropna()
   if len(q)>=8: out[h].append(q.f.corr(q.y))
  if i:
   z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
   if len(z)>=8: turns.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
  q=f.iloc[i].dropna(); cov.append(len(q)/15)
 print('look',look,'dates',len(out[1]),'n',len(px),'IC',*(round(np.nanmean(out[h]),5) for h in out),'ICIR',round(np.nanmean(out[1])/np.nanstd(out[1],ddof=1),5),'hit',round(np.mean(np.array(out[1])>0),5),'cov',round(np.mean(cov),5),'turn',round(np.mean(turns),5))
 print('decay_n',*(len(out[h]) for h in out))
 # regime split by cross-sectional dispersion
 disp=raw.std(axis=1); med=disp.median(); lo=[];hi=[]
 for i,x in enumerate(out[1]):
  pass
 for i in range(len(p)-10):
  q=pd.concat([f.iloc[i].rename('f'),p.pct_change(1).iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8:
   (lo if disp.iloc[i]<=med else hi).append(q.f.corr(q.y))
 print('regime',round(np.nanmean(lo),5),len(lo),round(np.nanmean(hi),5),len(hi))
