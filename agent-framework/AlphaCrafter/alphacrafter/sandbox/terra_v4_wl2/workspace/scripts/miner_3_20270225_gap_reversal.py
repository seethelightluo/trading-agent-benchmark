import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
px={}; op={}
for s in U:
 d=get(s).set_index('date'); px[s]=d['close'];op[s]=d['open']
C=pd.DataFrame(px).sort_index(); O=pd.DataFrame(op).reindex(C.index)
# Gap reversal: today's open-close gap, rank reversal predicts next close return.
gap=O/C.shift(1)-1
sig=-gap.rolling(2).mean() # smooth two-session opening gap shock
fwd=C.shift(-1)/C-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(a),'avgN',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit', (a.ic>0).mean())
print('coverage',sig.notna().sum().sum()/(len(U)*len(sig)),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=a[(a.index.astype(str)>=lo)&(a.index.astype(str)<=hi)].ic
 print(name,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
for h in [5,10]:
 F=C.shift(-h)/C-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],F.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=np.array(vals);print('H',h,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
