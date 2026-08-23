import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); S[s]=d.set_index('date').sort_index().close.astype(float).replace([np.inf,-np.inf],np.nan)
px=pd.DataFrame(S).sort_index(); r=px.pct_change(); r10=px.pct_change(10); vol20=r.rolling(20).std()*np.sqrt(20)
disp=r.std(axis=1).rolling(20).mean(); mult=(disp/disp.rolling(120).median()).clip(0.5,2.0)
peer=r10.sub(r10.mean(axis=1),axis=0)
factor=(-peer/(vol20*np.sqrt(20))).mul(mult,axis=0).shift(1)
factor.to_csv('scripts/miner_3_20340427_dispersion_residual_reversal_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],list(px.columns))
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy(); m=a.mean(); ir=m/a.std(ddof=1)*np.sqrt(len(a))
 print('h',h,'dates',len(a),'avg_names',q.n.mean(),'coverage',q.n.mean()/15,'IC',m,'ICIR',ir,'hit',(a>0).mean())
 if h==20:
  for aa,bb in [('2025','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
   x=q[(q.date.astype(str)>=aa)&(q.date.astype(str)<=bb)].ic; print('regime',aa,bb,len(x),x.mean())
print('turnover',factor.rank(pct=True).diff().abs().stack().mean())
