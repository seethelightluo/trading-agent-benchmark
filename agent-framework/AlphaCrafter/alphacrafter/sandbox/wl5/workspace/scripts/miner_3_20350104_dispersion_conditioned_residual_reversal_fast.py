import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); common=r.mean(axis=1); res=r.sub(common,axis=0)
base=(-res.rolling(10,min_periods=8).sum()/(res.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-12)).clip(-8,8)
disp=res.std(axis=1).rolling(20,min_periods=12).mean(); dr=disp.rolling(252,min_periods=80).rank(pct=True); sig=base*(.5+dr).clip(.5,1.5)
def evalh(h):
 q=P.shift(-h)/P-1; out=[]; ns=[]; dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a=z.iloc[:,0].rank(); b=z.iloc[:,1].rank(); c=a.corr(b)
   if np.isfinite(c): out.append(c);ns.append(len(z));dates.append(dt)
 return pd.Series(out,index=pd.DatetimeIndex(dates)),pd.Series(ns,index=pd.DatetimeIndex(dates))
print('assets',len(P.columns),'rows',len(P))
ic,n=evalh(10); print('dates',len(ic),'start',ic.index[0].date(),'end',ic.index[-1].date(),'mean_n',round(n.mean(),3),'coverage',round(n.mean()/15,6),'IC',round(ic.mean(),6),'ICIR_daily',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),6))
for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-01-03')]:
 z=ic.loc[x:y]; print('regime',x,len(z),round(z.mean(),6))
print('turnover',round(sig.rank(pct=True).diff().abs().mean().mean(),6))
for h in [5,20]:
 z,_=evalh(h);print('decay',h,round(z.mean(),6),'dates',len(z))
pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20350104_dispersion_conditioned_residual_reversal_signal.csv',index=False)
