import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
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
P=pd.DataFrame(S).sort_index(); ret=P.pct_change(); r20=P.pct_change(20)
# Downside-risk normalized reversal: losers measured relative to their recent downside volatility.
down=ret.where(ret<0).rolling(20,min_periods=10).std()*np.sqrt(252)
sig=(-r20).div(down.clip(lower=.05))
rows=[]
for dt in P.index:
 z=pd.DataFrame({'s':sig.loc[dt],'f':(P.shift(-10)/P-1).loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,z.s.rank().corr(z.f.rank()),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('instruments',len(P.columns),'dates',len(P),'observations',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15)
print('10d IC=%.6f ICIR=%.4f hit=%.4f turnover=%.6f'%(m,m/sd*np.sqrt(252),(q.ic>0).mean(),q.ic.diff().abs().mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,'dates',len(y),'IC=%.6f'%y.mean() if len(y) else 'none')
for h in [1,5,20]:
 f=P.shift(-h)/P-1; a=[]
 for dt in P.index:
  z=pd.DataFrame({'s':sig.loc[dt],'f':f.loc[dt]}).dropna()
  if len(z)>=8:a.append(z.s.rank().corr(z.f.rank()))
 print('decay',h,'IC=%.6f dates=%d'%(np.mean(a),len(a)))
