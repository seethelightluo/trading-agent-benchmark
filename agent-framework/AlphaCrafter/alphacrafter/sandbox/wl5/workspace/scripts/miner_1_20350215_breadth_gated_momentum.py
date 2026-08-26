import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); cs=r.sub(r.mean(axis=1),axis=0)
# Medium-horizon relative momentum, volatility-normalized and attenuated when market breadth is weak.
raw=cs.rolling(60,min_periods=40).sum()
vol=cs.rolling(60,min_periods=40).std()*np.sqrt(60)+1e-12
mom=(raw/vol).clip(-8,8)
breadth=(cs.rolling(20,min_periods=15).mean()>0).mean(axis=1)
# Trend is more reliable when at least half of the cross-section has positive recent relative returns.
gate=(0.5+0.5*((breadth-0.5)/0.25).clip(-1,1)).fillna(0.5)
sig=mom.mul(gate,axis=0)
def evaluate(h):
 q=P.shift(-h)/P-1; out=[]; ds=[]; ns=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   c=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
   if np.isfinite(c): out.append(c); ds.append(dt); ns.append(len(x))
 return pd.Series(out,index=pd.DatetimeIndex(ds)),pd.Series(ns,index=pd.DatetimeIndex(ds))
for h in [5,10,20]:
 ic,n=evaluate(h)
 print('horizon',h,'dates',len(ic),'start',ic.index[0].date(),'end',ic.index[-1].date(),'mean_n',round(n.mean(),3),'coverage',round(n.mean()/15,6),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),6))
 if h==10:
  for a,b in [('2023-06-30','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2030-12-31'),('2031-01-01','2035-02-01')]:
   z=ic.loc[a:b]; print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None)
  print('turnover',round(sig.rank(pct=True).loc[ic.index].diff().abs().mean().mean(),6))
  rows=[(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])]
  pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20350215_breadth_gated_momentum_signal.csv',index=False)
