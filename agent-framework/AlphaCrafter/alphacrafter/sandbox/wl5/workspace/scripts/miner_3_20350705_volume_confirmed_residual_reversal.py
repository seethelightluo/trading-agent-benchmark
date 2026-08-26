import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=120:
  x=d.set_index('date'); px[s]=x.close.astype(float); vol[s]=x.volume.astype(float)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change(); M=R.mean(axis=1)
beta=R.rolling(60,min_periods=40).cov(M).div(M.rolling(60,min_periods=40).var(),axis=0)
ret20=P.pct_change(20); market20=M.rolling(20).sum(); res=ret20-beta.mul(market20,axis=0)
idvol=R.sub(beta.mul(M,axis=0),axis=0).rolling(20,min_periods=15).std()*np.sqrt(20)
base=-res/(idvol+1e-12)
# causal relative activity; reversal is stronger when the recent selloff occurs on declining activity
lv=np.log(V.replace(0,np.nan)); activity=lv.rolling(5,min_periods=3).mean()-lv.rolling(40,min_periods=20).mean()
# negative activity gets multiplier above one, positive activity below one, bounded
mult=(1-0.35*activity.clip(-3,3)).clip(0.25,2.0)
sig=base*mult
print('assets',len(P.columns),'rows',len(P),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 q=P.shift(-h)/P-1; aa=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c); dates.append(dt); ns.append(len(z))
 aa=np.array(aa); dates=pd.DatetimeIndex(dates); ns=np.array(ns)
 if len(aa): print('horizon',h,'dates',len(aa),'start',dates[0].date(),'end',dates[-1].date(),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'hit',round((aa>0).mean(),6))
 if h==10 and len(aa):
  for a,b in [('2023-11-13','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-07-05')]:
   w=aa[(dates>=a)&(dates<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates],index=dates); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20350705_volume_confirmed_residual_reversal_signal.csv',index=False)
