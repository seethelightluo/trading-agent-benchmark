import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D=['XAU','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None and len(d): return d
        except Exception: pass
S={}
for s in U:
    d=fetch(s)
    if d is None: continue
    d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
    c=pd.to_numeric(d.close,errors='coerce')
    S[s]=c
P=pd.DataFrame(S).sort_index()
ret20=P.pct_change(20); ret5=P.pct_change(5)
defwd=P.shift(-10)/P-1
rows=[]
for dt in P.index:
    if dt not in ret20.index: continue
    vals=ret20.loc[dt]; fw=defwd.loc[dt]
    defensive=ret20.loc[dt,[x for x in D if x in P.columns]].median()
    # relative medium-term strength versus a defensive benchmark; neutralizes common regime beta
    sig=vals-defensive
    z=pd.DataFrame({'sig':sig,'fw':fw}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)>=8:
        ic=z.sig.rank().corr(z.fw.rank())
        rows.append((dt,ic,len(z),defensive))
q=pd.DataFrame(rows,columns=['date','ic','n','def_ret'])
print('instruments',len(P.columns),list(P.columns),'dates',len(P),'observations',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15)
m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('10d IC=%.6f ICIR=%.4f hit=%.4f turnover_proxy=%.6f'%(m,m/sd*np.sqrt(252), (q.ic>0).mean(), q.ic.diff().abs().mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic
 print('regime',a,'dates',len(y),'IC=%.6f'%y.mean() if len(y) else 'none')
for h in [1,5,20]:
 f=P.shift(-h)/P-1; rr=[]
 for dt in P.index:
  z=pd.DataFrame({'sig':ret20.loc[dt],'fw':f.loc[dt]}).dropna()
  if len(z)>=8: rr.append(z.sig.rank().corr(z.fw.rank()))
 print('decay',h,'d IC=%.6f'%np.mean(rr),'dates',len(rr))
