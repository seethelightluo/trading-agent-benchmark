import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=None
    for fn in (get_stock_daily_data,get_index_daily_data):
        try: d=fn(s,days=3000)
        except Exception: pass
        if d is not None: break
    if d is not None and len(d)>180:
        q=d.copy(); q['date']=pd.to_datetime(q['date']); D[s]=q.set_index('date')['close'].astype(float).sort_index()
p=pd.concat(D,axis=1).sort_index(); r=np.log(p).diff(); mu=r.mean(axis=1); res=r.sub(mu,axis=0)
rel=res.rolling(60,min_periods=45).sum(); consistency=(res>0).rolling(60,min_periods=45).mean()*2-1
sig=(rel*consistency).shift(1)
sig.rename_axis('date').to_csv('scripts/miner_2_20310904_relative_trend_persistence_signal.csv')
print('assets',len(D),'rows',len(p),'period',p.index.min(),p.index.max())
for h in [5,10,20,40,60]:
 fwd=np.log(p.shift(-h)/p); vals=[]; ns=[]; cov=[]; dates=[]
 for dt in sig.index:
  z=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.s,z.f).statistic); ns.append(len(z)); cov.append(len(z)/len(U)); dates.append(dt)
 a=np.asarray(vals); ic=np.nanmean(a); sd=np.nanstd(a,ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
 prev=None; turns=[]
 for dt in sig.index:
  x=sig.loc[dt].rank(pct=True)
  if prev is not None: turns.append(np.nanmean(abs(x-prev)))
  prev=x
 print('H',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(cov),'IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic,icir,np.mean(a>0),np.nanmean(turns)))
 if h==20:
  for name,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-09-04')]:
   vv=a[[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in dates]]
   print(name,len(vv),'IC %.6f ICIR %.6f'%(np.nanmean(vv),np.nanmean(vv)/np.nanstd(vv,ddof=1)*np.sqrt(252) if len(vv)>1 else np.nan))
