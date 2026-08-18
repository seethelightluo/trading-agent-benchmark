import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try: d=get_stock_daily_data(s, days=6000)
 except Exception: d=None
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=np.log(p).diff()
m20=r.rolling(20).sum(); m60=r.rolling(60).sum(); vol20=r.rolling(20).std()
f=(m20-m60/3)/(vol20*np.sqrt(20)); f=f.shift(1); fw=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok]),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(px),list(px),'dates',len(a),'avgN',a.n.mean(),'coverage',a.n.sum()/(len(a)*len(px)))
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
print('turnover',np.nanmean([((f.loc[d].rank(pct=True)-f.shift(1).loc[d].rank(pct=True)).abs().mean()) for d in f.index[1:] if f.loc[d].notna().sum()>=8]))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=a.loc[lo:hi].ic; print(lo,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [5,10,20,40]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in f.index:
  x=f.loc[dt]; y=yy.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(x[ok].corr(y[ok]))
 print('decay',h,np.nanmean(rr),len(rr))
