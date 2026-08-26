import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=100: cl[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(cl).sort_index(); r=P.pct_change(); cs=r.sub(r.mean(axis=1),axis=0)
base=(-cs.rolling(10,min_periods=8).sum()/(cs.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-12)).clip(-8,8)
breadth=(r<0).mean(axis=1); bz=(breadth-breadth.rolling(60,min_periods=30).mean())/(breadth.rolling(60,min_periods=30).std()+1e-12)
mult=(1+0.45*np.tanh(bz)).clip(.55,1.45); sig=base.mul(mult,axis=0)
print('assets',len(P.columns),'rows',len(P),'date_start',P.index.min().date(),'date_end',P.index.max().date())
def ev(h):
 q=P.shift(-h)/P-1; a=[]; ds=[]; ns=[]
 for dt in sig.index:
  zz=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(zz)>=8:
   c=zz.iloc[:,0].corr(zz.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(zz))
 return np.array(a),pd.DatetimeIndex(ds),np.array(ns)
for h in [5,10,20]:
 a,ds,ns=ev(h)
 if not len(a): print('horizon',h,'no valid observations'); continue
 print('horizon',h,'dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
 if h==10:
  for x,y in [('2023-08-23','2025-12-31'),('2026-01-01','2030-12-31'),('2031-01-01','2035-03-14')]:
   zz=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]; print('regime',x,y,'dates',len(zz),'IC',round(zz.mean(),6) if len(zz) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20350329_breadth_conditioned_reversal_signal.csv',index=False)
