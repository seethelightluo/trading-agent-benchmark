import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
r60=P/P.shift(60)-1
# Reward persistent appreciation, penalizing only downside variation (risk-aware trend)
down=r.where(r<0,0).rolling(60,min_periods=40).std()*np.sqrt(60)
sig=(r60/(down+1e-8)).clip(-10,10)
rows=[]
for h in [5,10,20]:
 Q=P.shift(-h)/P-1; vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],Q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); dates.append(dt); ns.append(len(z))
 a=np.asarray(vals); di=pd.DatetimeIndex(dates)
 print('horizon',h,'dates',len(a),'start',di[0].date(),'end',di[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
 if h==10:
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20341012_downside_vol_momentum_signal.csv',index=False)
  for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-10-11')]:
   z=a[(di>=pd.Timestamp(x))&(di<=pd.Timestamp(y))]
   if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in di],index=di)
  print('turnover',round(ranks.diff().abs().mean().mean(),6))
