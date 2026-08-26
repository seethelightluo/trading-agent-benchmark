import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; H={}; L={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None:
  x=d.set_index('date'); P[s]=x.close.astype(float); H[s]=x.high.astype(float); L[s]=x.low.astype(float)
P=pd.DataFrame(P).sort_index(); H=pd.DataFrame(H).reindex(P.index); L=pd.DataFrame(L).reindex(P.index)
r=P.pct_change(); cs=r.sub(r.mean(axis=1),axis=0)
# Fade cumulative idiosyncratic move more strongly after an unusually wide recent range.
move=cs.rolling(10,min_periods=8).sum(); risk=cs.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-12
base=(-move/risk).clip(-8,8)
tr=(H/L-1).replace([np.inf,-np.inf],np.nan); rz=(tr-tr.rolling(40,min_periods=25).mean())/(tr.rolling(40,min_periods=25).std()+1e-12)
sig=(base*(1+0.30*np.tanh(rz.rolling(3,min_periods=2).mean()))).clip(-8,8)
def ev(h):
 q=P.shift(-h)/P-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   c=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c);ds.append(dt);ns.append(len(x))
 return pd.Series(vals,index=pd.DatetimeIndex(ds)),pd.Series(ns,index=pd.DatetimeIndex(ds))
for h in [5,10,20]:
 ic,n=ev(h); print('horizon',h,'dates',len(ic),'start',ic.index[0].date(),'end',ic.index[-1].date(),'mean_n',round(n.mean(),3),'coverage',round(n.mean()/15,6),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),6))
 if h==10:
  for a,b in [('2023-09-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2030-12-31'),('2031-01-01','2035-02-01')]:
   z=ic.loc[a:b]; print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None)
  print('turnover',round(sig.rank(pct=True).loc[ic.index].diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20350215_range_shock_reversal_signal.csv',index=False)
