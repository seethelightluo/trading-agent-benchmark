import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); common=r.mean(axis=1); res=r.sub(common,axis=0)
shock=res.rolling(10,min_periods=8).sum(); vol=res.rolling(40,min_periods=25).std()
base=(-shock/(vol*np.sqrt(10)+1e-12)).clip(-8,8)
disp=res.std(axis=1).rolling(20,min_periods=12).mean()
# normalized dispersion, causal and computationally simple
norm=disp/disp.rolling(252,min_periods=80).median()
sig=base*(0.75+0.5*norm.clip(0.5,1.5))
q=P.shift(-10)/P-1
ics=[]; dates=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(z))
a=np.array(ics); dates=pd.DatetimeIndex(dates)
print('assets',len(P.columns),'rows',len(P),'dates',len(a),'start',dates[0].date() if len(a) else None,'end',dates[-1].date() if len(a) else None,'mean_n',round(np.mean(ns),3) if ns else 0,'coverage',round(np.mean(ns)/15,6) if ns else 0)
if len(a):
 print('IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
 for x,y in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-01-03')]:
  z=a[(dates>=x)&(dates<=y)]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
 ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates],index=dates)
 print('turnover',round(ranks.diff().abs().mean().mean(),6))
 for h in [5,20]:
  f=P.shift(-h)/P-1; aa=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if np.isfinite(c): aa.append(c)
  print('decay',h,round(np.mean(aa),6),'dates',len(aa))
out=pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value'])
out.to_csv('scripts/miner_2_20350104_dispersion_residual_reversal_signal.csv',index=False)
