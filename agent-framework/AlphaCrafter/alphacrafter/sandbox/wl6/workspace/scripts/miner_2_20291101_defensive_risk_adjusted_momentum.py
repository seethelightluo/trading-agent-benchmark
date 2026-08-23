import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=4000)
   if x is not None and len(x): return x
  except Exception: pass
S={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date); S[s]=pd.to_numeric(x.set_index('date').sort_index().close,errors='coerce')
P=pd.DataFrame(S).sort_index(); r=P.pct_change(20); vol=P.pct_change().rolling(20).std()*np.sqrt(252); fw=P.shift(-10)/P-1
# One interpretable idea: medium-term return relative to defensive median, scaled by own realized risk.
defensive=r[[x for x in D if x in P]].median(axis=1)
sig=(r.sub(defensive,axis=0)).div(vol.clip(lower=.05))
rows=[]
for dt in P.index:
 z=pd.DataFrame({'s':sig.loc[dt],'f':fw.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,z.s.rank().corr(z.f.rank()),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']); print('instruments',len(P.columns),'dates',len(P),'observations',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15)
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; a=[]
 for dt in P.index:
  z=pd.DataFrame({'s':sig.loc[dt],'f':f.loc[dt]}).dropna()
  if len(z)>=8:a.append(z.s.rank().corr(z.f.rank()))
 a=np.array(a); print('horizon',h,'dates',len(a),'IC=%.6f ICIR=%.6f hit=%.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252),np.mean(a>0)))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,'dates',len(y),'IC=%.6f'%y.mean() if len(y) else 'none')
print('turnover_proxy',sig.rank(axis=1).diff().abs().mean().mean(),'max_abs_library_correlation',None)
