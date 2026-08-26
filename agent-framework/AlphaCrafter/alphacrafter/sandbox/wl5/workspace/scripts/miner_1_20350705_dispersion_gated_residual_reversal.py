import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=120: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); M=R.mean(axis=1)
mr=R.rolling(60,min_periods=40).mean(); mm=M.rolling(60,min_periods=40).mean(); cov=R.mul(M,axis=0).rolling(60,min_periods=40).mean()-mr.mul(mm,axis=0); beta=cov.div(M.rolling(60,min_periods=40).var(),axis=0)
ret20=P.pct_change(20); market20=M.rolling(20).sum(); res=ret20-beta.mul(market20,axis=0)
idvol=R.sub(beta.mul(M,axis=0),axis=0).rolling(20,min_periods=15).std()*np.sqrt(20)
base=-res/(idvol+1e-12)
disp=R.std(axis=1).rolling(20,min_periods=15).mean(); z=(disp-disp.rolling(120,min_periods=60).mean())/(disp.rolling(120,min_periods=60).std()+1e-12)
sig=base*(1+0.5*z.clip(-1,2)).clip(0.25,2.0)
print('assets',len(P.columns),'rows',len(P),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 q=P.shift(-h)/P-1; aa=[]; dates=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c);dates.append(dt);ns.append(len(a))
 aa=np.array(aa); dates=pd.DatetimeIndex(dates); ns=np.array(ns)
 print('horizon',h,'dates',len(aa),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'hit',round((aa>0).mean(),6))
 if h==10:
  for x,y in [('2023-10-25','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-06-25')]:
   w=aa[(dates>=x)&(dates<=y)];print('regime',x,y,len(w),round(w.mean(),6))
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates]);print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20350705_dispersion_gated_residual_reversal_signal.csv',index=False)
