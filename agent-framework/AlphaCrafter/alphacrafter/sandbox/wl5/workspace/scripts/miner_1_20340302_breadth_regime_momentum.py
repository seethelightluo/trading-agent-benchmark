import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=160: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r20=P/P.shift(20)-1
v40=R.rolling(40,min_periods=25).std(); mom=r20/(v40*np.sqrt(20)+1e-8)
# Causal cross-sectional breadth regime: favor continuation in broad positive tape,
# and reversal when the majority of assets are negative.
breadth=(r20>0).mean(axis=1)
sign=pd.Series(np.where(breadth>=0.5,1.0,-1.0),index=P.index)
f=mom.mul(sign,axis=0).clip(-6,6)
fwds={h:P.shift(-h)/P-1 for h in [5,10,20]}; ics={h:[] for h in fwds}; dates={h:[] for h in fwds}; rows=[]
for dt in f.index:
 for h in fwds:
  z=pd.concat([f.loc[dt],fwds[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): ics[h].append(c); dates[h].append(dt)
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20340302_breadth_regime_momentum_signal.csv',index=False)
for h in fwds:
 a=np.array(ics[h]); ds=pd.DatetimeIndex(dates[h]); ns=[len(pd.concat([f.loc[d],fwds[h].loc[d]],axis=1).dropna()) for d in dates[h]]
 print('horizon',h,'dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
 for x,y in [('2020-07-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-03-01')]:
  z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1)*np.sqrt(252),6))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in dates[10]],index=dates[10]); print('turnover',round(S.diff().abs().mean().mean(),6))
